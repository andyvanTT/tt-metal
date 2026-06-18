# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K-parallel flash-decode attention for the pi0.5 denoise step (blitz-style custom op).

Shape this op targets (single chip, MQA):
  Q:    (1, NH=8, Sq=32, D=256)      — small query (10 real rows, padded to 32)
  K/V:  (1, 1,  Sk=1056, D=256)      — one shared KV head (MQA), long sequence
  mask: (1, 1,  Sq, Sk) additive     — blocks phantom KV columns (non-causal)
  out:  (1, NH, Sq, D)

The stock prefill SDPA op parallelizes only over batch×heads×q_chunks → ~8 cores
for this shape. This op splits the Sk dimension across many cores (flash-decode)
with an online-softmax cross-core combine, keeping K/V resident in L1 (no batch
replication). See tests/perf/_sdpa_decode_qfold_repro.py for the PCC/perf refs.

BUILT INCREMENTALLY (see git history of this file):
  M1  (current) — plumbing only: single-core copy through ttnn.generic_op, to
                  validate the andy_kernels packaging + CB/kernel compile path.
  M2  — single-core full attention (QK^T → mask → softmax → AV) for correctness.
  M3  — per-head fan-out (8 cores), then Sk-split + online-softmax reduce (the win).
"""


import torch
import ttnn

_KDIR = "models/experimental/pi0_5/andy_kernels/flash_decode_mqa/kernels"


class FlashDecodeMQA:
    @staticmethod
    def golden(q, k, v, mask, scale):
        """PyTorch reference. q (1,NH,Sq,D), k/v (1,1,Sk,D), mask (1,1,Sq,Sk) additive.
        MQA: the single KV head is shared across all NH query heads."""
        nh = q.shape[1]
        kr = k.expand(1, nh, k.shape[2], k.shape[3])
        vr = v.expand(1, nh, v.shape[2], v.shape[3])
        return torch.nn.functional.scaled_dot_product_attention(
            q, kr, vr, attn_mask=mask.expand(1, nh, mask.shape[2], mask.shape[3]), scale=scale, is_causal=False
        )

    # ------------------------------------------------------------------ M1
    @staticmethod
    def op_passthrough(in_tensor: ttnn.Tensor, out_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Milestone 1: single-core copy in→out via generic_op. Validates the
        packaging, CB descriptors, and reader/compute/writer kernel compile path
        before any attention math is added. in/out must be identically-sharded
        L1 TILE tensors on a single core."""
        cb_in, cb_out = 0, 16
        num_tiles = (in_tensor.shape[-2] // 32) * (in_tensor.shape[-1] // 32)
        cores = in_tensor.memory_config().shard_spec.grid

        cb_in_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in, in_tensor)
        cb_out_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_tensor)

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_in, num_tiles],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_in, cb_out, num_tiles],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, num_tiles],
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute],
            cbs=[cb_in_desc, cb_out_desc],
            semaphores=[],
        )
        ttnn.generic_op([in_tensor, out_tensor], prog)
        return out_tensor

    # ------------------------------------------------------------------ M2a
    @staticmethod
    def op_qkt(q: ttnn.Tensor, k: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """Milestone 2a: S = Q @ K^T for one (32x32) score block, single core,
        contracting over head_dim. Validates matmul+transpose in-kernel before
        softmax/PV. q,k: (32, head_dim) L1 TILE single-core shards; out: (32,32)."""
        cb_q, cb_k, cb_out = 0, 1, 16
        dt = q.shape[-1] // 32  # contraction tiles (head_dim / 32)
        cores = q.memory_config().shard_spec.grid

        cbq = ttnn.cb_descriptor_from_sharded_tensor(cb_q, q)
        cbk = ttnn.cb_descriptor_from_sharded_tensor(cb_k, k)
        cbo = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out)

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_q, cb_k, dt],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_q, cb_k, cb_out, dt],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, 1],
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=[cbq, cbk, cbo], semaphores=[])
        ttnn.generic_op([q, k, out], prog)
        return out

    # ------------------------------------------------------------------ M2b
    @staticmethod
    def _scratch_cb(idx, cores, n_tiles=1, dtype=ttnn.bfloat16):
        """A non-tensor-backed L1 scratch circular buffer of n_tiles 32x32 tiles."""
        tile_bytes = 32 * 32 * 2  # bf16
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=tile_bytes)
        return ttnn.CBDescriptor(total_size=tile_bytes * n_tiles, core_ranges=cores, format_descriptors=[fmt])

    @staticmethod
    def op_softmax(s: ttnn.Tensor, scaler: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """Milestone 2b: row-softmax of one (32x32) score tile, single core.
        s: (32,32) scores; scaler: (32,32) of 1.0 (reduce multiplier); out: (32,32)."""
        cb_s, cb_scaler, cb_out = 0, 1, 16
        cb_max, cb_exp, cb_sum, cb_recip = 24, 25, 26, 27
        cores = s.memory_config().shard_spec.grid

        cbs = ttnn.cb_descriptor_from_sharded_tensor(cb_s, s)
        cbsc = ttnn.cb_descriptor_from_sharded_tensor(cb_scaler, scaler)
        cbo = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out)
        scratch = [
            FlashDecodeMQA._scratch_cb(cb_max, cores),
            FlashDecodeMQA._scratch_cb(cb_exp, cores),
            FlashDecodeMQA._scratch_cb(cb_sum, cores),
            FlashDecodeMQA._scratch_cb(cb_recip, cores),
        ]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_softmax_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_s, cb_scaler],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_softmax_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, 1],
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=[cbs, cbsc, cbo, *scratch], semaphores=[])
        ttnn.generic_op([s, scaler, out], prog)
        return out

    # ------------------------------------------------------------------ M2d
    @staticmethod
    def op_attn(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, scaler: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """Milestone 2d: full single-block attention O = softmax(Q@K^T)@V, single
        core, Sk = one 32-tile. Q must be PRE-SCALED by 1/sqrt(d) on the host.
        q,k: (32, d); v: (32, dv); scaler: (32,32) of 1.0; out: (32, dv)."""
        cb_q, cb_k, cb_v, cb_scaler, cb_out = 0, 1, 2, 3, 16
        dt = q.shape[-1] // 32
        vt = v.shape[-1] // 32
        cores = q.memory_config().shard_spec.grid

        cbq = ttnn.cb_descriptor_from_sharded_tensor(cb_q, q)
        cbk = ttnn.cb_descriptor_from_sharded_tensor(cb_k, k)
        cbv = ttnn.cb_descriptor_from_sharded_tensor(cb_v, v)
        cbsc = ttnn.cb_descriptor_from_sharded_tensor(cb_scaler, scaler)
        cbo = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out)
        scratch = [FlashDecodeMQA._scratch_cb(i, cores) for i in (24, 25, 26, 27, 28, 29)]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_attn_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_q, cb_k, cb_v, cb_scaler, dt, vt],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_attn_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, vt],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, vt],
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute], cbs=[cbq, cbk, cbv, cbsc, cbo, *scratch], semaphores=[]
        )
        ttnn.generic_op([q, k, v, scaler, out], prog)
        return out

    # ------------------------------------------------------------------ M2e
    @staticmethod
    def op_attn_full(q, k, v, scaler, mask, out, stage=0) -> ttnn.Tensor:
        """Milestone 2e: single-core full-row attention over the real Sk (Skt tiles)
        with additive mask. Q pre-scaled by 1/sqrt(d). The per-core kernel M3 K-splits.
        q (32,d); k (Sk,d); v (Sk,dv); scaler (32,32) 1.0; mask (32,Sk); out (32,dv)."""
        cb_q, cb_k, cb_v, cb_scaler, cb_mask, cb_out = 0, 1, 2, 3, 4, 16
        dt = q.shape[-1] // 32
        vt = v.shape[-1] // 32
        Skt = k.shape[-2] // 32
        cores = q.memory_config().shard_spec.grid

        cbq = ttnn.cb_descriptor_from_sharded_tensor(cb_q, q)
        cbk = ttnn.cb_descriptor_from_sharded_tensor(cb_k, k)
        cbv = ttnn.cb_descriptor_from_sharded_tensor(cb_v, v)
        cbsc = ttnn.cb_descriptor_from_sharded_tensor(cb_scaler, scaler)
        cbm = ttnn.cb_descriptor_from_sharded_tensor(cb_mask, mask)
        cbo = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out)
        scratch = [
            FlashDecodeMQA._scratch_cb(24, cores, Skt),  # cb_qk
            FlashDecodeMQA._scratch_cb(25, cores, 1),  # cb_max
            FlashDecodeMQA._scratch_cb(26, cores, Skt),  # cb_exp
            FlashDecodeMQA._scratch_cb(27, cores, 1),  # cb_sum
            FlashDecodeMQA._scratch_cb(28, cores, 1),  # cb_recip
            FlashDecodeMQA._scratch_cb(29, cores, Skt),  # cb_p
            FlashDecodeMQA._scratch_cb(30, cores, Skt),  # cb_qkm
        ]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_attn_mt_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_q, cb_k, cb_v, cb_scaler, cb_mask, dt, vt, Skt],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_attn_mt_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, vt, Skt, stage],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/fd_qkt_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, vt],
            config=ttnn.WriterConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute], cbs=[cbq, cbk, cbv, cbsc, cbm, cbo, *scratch], semaphores=[]
        )
        ttnn.generic_op([q, k, v, scaler, mask, out], prog)
        return out
