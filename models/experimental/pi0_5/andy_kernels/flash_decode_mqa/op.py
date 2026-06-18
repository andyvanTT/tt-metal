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
