# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase-0 GO/NO-GO harness: K-parallel denoise SDPA via the decode op (q-fold).

The pi0.5 denoise step's attention runs on the PREFILL sdpa op
(ttnn.transformer.scaled_dot_product_attention), which parallelizes over
batch×heads×q_chunks only. For the denoise shape (B=1, NQH=8, Q=32→1 q-chunk,
MQA num_kv_heads=1, KV≈1056) that pins it to ~8 cores → ~67 µs/call, 34% of the
step (Tracy-verified).

The DECODE op (scaled_dot_product_attention_decode) flash-decodes by splitting K
across cores. It requires Q=[1, B, NH, D] (one query position per batch elem) and
K/V in DRAM+TILE. The denoise joint-attention is NON-CAUSAL and all 32 suffix
queries attend the same KV, so we fold the 32 query positions into the decode
batch dim and share one KV cache (share_cache=True).

This script validates the composed decode path against (a) the current prefill
SDPA op and (b) a torch reference, then measures device time of the full composed
path (permute-in + KV DRAM staging + decode + permute-out) vs the current single
SDPA op via trace replay. GO only if PCC holds AND it's a net device-time win.

Run (single chip):
  python_env/bin/python models/experimental/pi0_5/tests/perf/_sdpa_decode_qfold_repro.py
"""

import math
import statistics
import time

import torch
import ttnn

# ---- denoise SDPA shape (upstream pi05_libero: action_horizon=10, 3-cam prefix) ----
D = 256  # head_dim
NH = 8  # query heads
NKV = 1  # MQA
PREFIX = 1024  # prefix KV tokens (3*256 img + 256 lang, tile-aligned)
AH = 10  # real action tokens (upstream checkpoint)
Q_PAD = 32  # suffix padded to tile
KV = PREFIX + Q_PAD  # 1056 total KV
SCALE = 1.0 / math.sqrt(D)
SEED = 0
TRACE_REGION = 134_217_728

NUM_WARMUP = 3
NUM_ITERS = 30


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_host():
    torch.manual_seed(SEED)
    # Q in prefill layout (1, NH, Q_PAD, D); K/V single MQA head (1, 1, KV, D).
    q = torch.randn(1, NH, Q_PAD, D, dtype=torch.float32)
    k = torch.randn(1, NKV, KV, D, dtype=torch.float32)
    v = torch.randn(1, NKV, KV, D, dtype=torch.float32)
    # Additive mask: block phantom suffix-K columns [PREFIX+AH : KV]. Same for
    # every query row and head (prefix is all-real + tile-aligned here).
    blocked = torch.zeros(KV, dtype=torch.float32)
    blocked[PREFIX + AH : KV] = -1e4
    return q, k, v, blocked


def _torch_ref(q, k, v, blocked):
    # Repeat MQA KV head across the 8 query heads, additive mask, non-causal.
    kr = k.repeat(1, NH, 1, 1)  # (1, NH, KV, D)
    vr = v.repeat(1, NH, 1, 1)
    mask = blocked.view(1, 1, 1, KV).expand(1, NH, Q_PAD, KV)
    out = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, attn_mask=mask, scale=SCALE, is_causal=False)
    return out  # (1, NH, Q_PAD, D)


INNER = 25  # chain N calls inside one trace so device time dominates host launch overhead


def _time_trace(device, fn, label):
    for _ in range(NUM_WARMUP):
        out = fn()
        ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(INNER):
        out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times.append((time.perf_counter() - t0) * 1e6 / INNER)  # µs per call
    ttnn.release_trace(device, tid)
    avg = statistics.mean(times)
    print(f"   {label:42s} {avg:8.2f} µs/call   (min {min(times):.2f}, {INNER} chained)")
    return avg, out


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=TRACE_REGION)
    try:
        n_dev = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        assert n_dev == 1, f"need single chip, got {n_dev}"
        grid = device.compute_with_storage_grid_size()
        grid_size = (grid.x, grid.y)
        print(f"\n🔒 single chip; grid={grid_size}; shape Q(1,{NH},{Q_PAD},{D}) KV(1,{NKV},{KV},{D})")

        q, k, v, blocked = _build_host()
        ref = _torch_ref(q, k, v, blocked)

        # ---- device tensors (production dtypes: Q bf16, K/V bf8_b) ----
        q_prefill = ttnn.from_torch(
            q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        k_l1 = ttnn.from_torch(
            k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        v_l1 = ttnn.from_torch(
            v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # prefill additive mask (1,1,Q_PAD,KV)
        prefill_mask_t = blocked.view(1, 1, 1, KV).expand(1, 1, Q_PAD, KV).contiguous()
        prefill_mask = ttnn.from_torch(
            prefill_mask_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        from models.experimental.pi0_5.tt.ttnn_common import get_sdpa_compute_kernel_config

        ckc = get_sdpa_compute_kernel_config()

        # ---------- current prefill SDPA ----------
        from models.experimental.pi0_5.tt.ttnn_common import sdpa_prefill_chunk_sizes

        qc, kc = sdpa_prefill_chunk_sizes(Q_PAD, KV)
        prefill_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )

        def run_prefill():
            return ttnn.transformer.scaled_dot_product_attention(
                q_prefill,
                k_l1,
                v_l1,
                attn_mask=prefill_mask,
                is_causal=False,
                scale=SCALE,
                program_config=prefill_cfg,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        out_prefill = run_prefill()
        ttnn.synchronize_device(device)
        prefill_host = ttnn.to_torch(out_prefill).float()  # (1,NH,Q_PAD,D)

        # ---------- decode HEAD-fold (fold positions×heads -> num_heads; K batch=1, NO replication) ----------
        # q_prefill (1,NH,Q_PAD,D) --reshape--> (1,1,NH*Q_PAD,D): row-major m=h*Q_PAD+i,
        # so reshaping the output straight back to (1,NH,Q_PAD,D) recovers [h,i].
        # K/V stay batch=1, MQA 1 head — NO replication. q_chunk_size (heads/chunk)
        # is the L1 knob: default (=padded heads, 256) overflows CBs, so chunk it.
        import os as _os

        POS_PER_CALL = int(_os.environ.get("QFOLD_POS", "8"))  # positions folded per decode call
        N_CALLS = (Q_PAD + POS_PER_CALL - 1) // POS_PER_CALL
        NH_FOLD = NH * POS_PER_CALL  # folded heads per call (must keep CBs < L1)
        Q_CHUNK = int(_os.environ.get("QFOLD_QCHUNK", str(NH_FOLD)))
        k_dram = ttnn.from_torch(
            k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # (1,1,KV,D)
        v_dram = ttnn.from_torch(
            v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        K_CHUNK = 32  # 1056 % 32 == 0
        decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size, q_chunk_size=Q_CHUNK, k_chunk_size=K_CHUNK, exp_approx_mode=False
        )
        dec_mask_t = blocked.view(1, 1, 1, KV).expand(1, NH_FOLD, 1, KV).contiguous().transpose(1, 2).contiguous()
        dec_mask = ttnn.from_torch(
            dec_mask_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(
            f"   chunked head-fold: {N_CALLS} calls × (NH_FOLD={NH_FOLD} heads, {POS_PER_CALL} pos) q_chunk={Q_CHUNK}"
        )

        _dbg = {}

        def run_decode():
            # q_prefill (1, NH, Q_PAD, D). For each position-chunk: slice (1,NH,POS,D),
            # reshape→(1,1,NH*POS,D), decode (K batch=1, no replication), reshape back.
            outs = []
            for c in range(N_CALLS):
                p0 = c * POS_PER_CALL
                q_c = ttnn.slice(q_prefill, [0, 0, p0, 0], [1, NH, p0 + POS_PER_CALL, D])  # (1,NH,POS,D)
                q_c = ttnn.reshape(q_c, (1, 1, NH_FOLD, D))
                q_c = ttnn.to_memory_config(q_c, ttnn.DRAM_MEMORY_CONFIG)
                o = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_c,
                    k_dram,
                    v_dram,
                    is_causal=False,
                    attn_mask=dec_mask,
                    scale=SCALE,
                    program_config=decode_cfg,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                outs.append(ttnn.reshape(o, (1, NH, POS_PER_CALL, D)))
            _dbg["out_raw"] = tuple(outs[0].shape)
            return ttnn.concat(outs, dim=2)  # (1, NH, Q_PAD, D)

        out_decode = run_decode()
        ttnn.synchronize_device(device)
        decode_host = ttnn.to_torch(out_decode).float()
        print(
            f"\n   [shapes] q_prefill={tuple(q_prefill.shape)} per_call_out={_dbg['out_raw']} "
            f"decode_out={tuple(decode_host.shape)} prefill_out={tuple(prefill_host.shape)} "
            f"torch_ref={tuple(ref.shape)}"
        )

        # ---- PCC on REAL rows only [0:AH] (phantom rows discarded downstream) ----
        ref_r = ref[:, :, :AH, :]
        pre_r = prefill_host[:, :, :AH, :]
        dec_r = decode_host[:, :, :AH, :]
        print("\n--- PCC (real rows [0:%d]) ---" % AH)
        print(f"   prefill vs torch : {_pcc(pre_r, ref_r):.6f}")
        if dec_r.shape == ref_r.shape:
            print(f"   decode  vs torch : {_pcc(dec_r, ref_r):.6f}")
            print(f"   decode  vs prefill: {_pcc(dec_r, pre_r):.6f}   (gate >= 0.999)")
        else:
            print(f"   !! decode shape {tuple(dec_r.shape)} != ref {tuple(ref_r.shape)} — fix layout before PCC")

        # ---- device-time comparison ----
        print("\n--- device time (trace replay) ---")
        t_pre, _ = _time_trace(device, run_prefill, "current prefill SDPA")
        t_dec, _ = _time_trace(device, run_decode, f"chunked head-fold ({N_CALLS} calls, no replication)")
        print(f"\n   full-path speedup:   {t_pre / t_dec:.2f}x   ({'GO' if t_dec < t_pre else 'NO-GO'})")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
