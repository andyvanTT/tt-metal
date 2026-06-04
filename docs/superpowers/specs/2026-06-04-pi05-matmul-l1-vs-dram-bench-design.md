# pi0.5 matmul L1-vs-DRAM micro-bench — design

**Date:** 2026-06-04
**Author:** Andy Van (with Claude)

## Goal

Answer one question per matmul: *does the model's matmul fit in L1, and is it
faster on L1 than on DRAM?* Do this by **replicating the actual matmul shapes the
pi0.5 model runs** (taken from a real tracy profile), running each with all
operands on DRAM vs all operands on L1, and reporting timing stats.

This is a **matmul-only micro-bench** — no norms, attention, RoPE, or real
weights. Random tensors of the profiled shape/dtype are sufficient because we
measure kernel time, not correctness.

## Source of the shapes

Shapes were extracted from `~/avan/ops_perf_results_pi05_matmuls_l1_vs_dram.xlsx`
(annotated tracy `ops_perf_results`, sheet `ops_annotated`). Filtering to
`MatmulDeviceOperation` and collapsing the 18-layer / 10-step repeats yields **14
distinct large matmuls** across three stages. In the profile every weight (in1)
is `bfloat8_b` on DRAM and every activation (in0) is on L1; output dtype is
`bfloat16`; math fidelity is `LoFi`.

### SigLIP vision block — M=256 (×27 layers)
| name | M | K | N | in0 dt | in1 dt | act |
|------|---|---|---|--------|--------|-----|
| qkv     | 256 | 1152 | 4608 | bf16 | bf8 | — |
| mlp_fc1 | 256 | 1152 | 4608 | bf16 | bf8 | gelu |
| o_proj  | 256 | 1536 | 1152 | bf8  | bf8 | — |
| mlp_fc2 | 256 | 4608 | 1152 | bf8  | bf8 | — |

### VLM prefill — Gemma-2B, M=512, width=2048 (×18 layers)
| name | M | K | N | in0 dt | in1 dt | act |
|------|---|---|---|--------|--------|-----|
| qkv     | 512 | 2048 | 2560  | bf16 | bf8 | — |
| o_proj  | 512 | 2048 | 2048  | bf8  | bf8 | — |
| gate    | 512 | 2048 | 16384 | bf16 | bf8 | gelu |
| up      | 512 | 2048 | 16384 | bf16 | bf8 | — |
| down    | 512 | 16384 | 2048 | bf16 | bf8 | — |  ← max size (17.2 GFLOP) & max time (~181 µs)

### Expert denoise — Gemma-300M, M=32, width=1024 (×18 layers ×10 steps)
| name | M | K | N | in0 dt | in1 dt | act |
|------|---|---|---|--------|--------|-----|
| qkv     | 32 | 1024 | 2560 | bf16 | bf8 | — |
| o_proj  | 32 | 2048 | 1024 | bf8  | bf8 | — |
| gate    | 32 | 1024 | 4096 | bf16 | bf8 | gelu |
| up      | 32 | 1024 | 4096 | bf16 | bf8 | — |
| down    | 32 | 4096 | 1024 | bf16 | bf8 | — |

Tiny n=1 edge matmuls (velocity proj `32×32×1024`, SigLIP projector
`256×1152×2048`, etc.) are **excluded** — not "large".

## Design

**Files (all in `models/experimental/pi0_5/tests/perf/`):**
- `matmul_bench_common.py` — shared, non-collected helper: `MatmulCfg` dataclass,
  `time_matmul(...)`, `run_stage(...)`, dtype map, reporting.
- `test_prefill_matmul_l1_vs_dram.py` — VLM-prefill configs + test.
- `test_siglip_matmul_l1_vs_dram.py` — SigLIP configs + test.
- `test_expert_matmul_l1_vs_dram.py` — expert-denoise configs + test.

**Per matmul, the bench:**
1. Builds random `in0` (M×K) and `in1` (K×N) at the profiled dtype, on the target
   memory (interleaved), once — kept resident so we time only the matmul.
2. Runs `ttnn.linear(in0, in1, dtype=bfloat16, memory_config=MEM,
   compute_kernel_config=LoFi, activation="gelu" if act else None)`. The matmul
   program config is left to ttnn's auto-selection so each placement runs in its
   best config (the realistic comparison).
3. Times two placements — **all-DRAM** and **all-L1** — with `NUM_WARMUP` +
   `NUM_ITER` host-timed iterations (`perf_counter` + `synchronize_device` both
   sides), deallocating the output each iteration.
4. Catches L1 OOM (`RuntimeError`) → reports **"does not fit"** instead of crashing.
5. Prints a per-stage table: `matmul | DRAM mean | L1 mean | speedup | L1 fits?`,
   flagging the max-size and max-time rows.

**Controlled variable:** only operand/output **memory placement** changes between
the two runs. Shape, dtype, fused activation, and compute config are held fixed
(matching the profile). Block-sharding is *not* replicated — both runs use
interleaved, for a clean L1-vs-DRAM comparison.

**Timing note:** host timing includes dispatch/sync overhead, so absolute numbers
are larger than the profile's device-kernel µs; the **L1-vs-DRAM ratio** is the
signal. Running under tracy still emits the per-op device-kernel CSV
(`generated/profiler/.logs/tracy_ops_*.csv`) for device-time comparison against
the source profile.

**Skip-gating:** each test gated behind its env var
(`PI0_MATMUL_BENCH_PREFILL/SIGLIP/EXPERT=1`) and tracy-friendly, consistent with
the existing perf benches. `PI0_BENCH_WARMUP` / `PI0_BENCH_ITER` knobs reused.

## Out of scope
- Real `pi05_base` weights (timing doesn't need them).
- Full block forward / norms / attention.
- The two `test_blocks_all_*.py` aggregators — **kept on disk, untouched**; they
  remain dormant (still import the never-written block-helper modules).
- The n=1 edge matmuls.

## Run
```
PI0_MATMUL_BENCH_PREFILL=1 pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_prefill_matmul_l1_vs_dram.py
# (and _siglip_ / _expert_ with their env vars)
```
