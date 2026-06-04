# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the pi0.5 matmul L1-vs-DRAM micro-benches.

NOT a test module (no ``test_`` prefix) so pytest does not collect it.

Each large matmul that the pi0.5 model runs (shapes taken from a real tracy
``ops_perf_results`` profile) is replicated here as a :class:`MatmulCfg` and
timed with EVERY operand+output on DRAM vs EVERY operand+output on L1. Only the
memory placement changes between the two runs — shape, dtype, fused activation
and compute config are held fixed, matching the profile. This isolates one
question per matmul: does it fit in L1, and is L1 faster than DRAM?

Design doc:
docs/superpowers/specs/2026-06-04-pi05-matmul-l1-vs-dram-bench-design.md
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import ttnn


NUM_WARMUP = int(os.environ.get("PI0_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_BENCH_ITER", "100"))

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# The source profile ran LoFi math fidelity with packer-L1 accumulation; match
# it so kernel times line up with the ops_perf_results export we replicate.
_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

_DTYPES = {
    "bf16": ttnn.bfloat16,
    "bf8": ttnn.bfloat8_b,
}

Stats = Tuple[float, float, float, float]  # mean, stdev, min, max  (ms)


@dataclass(frozen=True)
class MatmulCfg:
    """One large matmul: out[M,N] = in0[M,K] @ in1[K,N], optional fused act.

    in0 is the activation, in1 the weight — dtypes named as in the profile
    ("bf16" / "bf8"). ``activation`` is a ttnn fused-activation string (e.g.
    "gelu") or None.
    """

    name: str
    M: int
    K: int
    N: int
    in0_dtype: str = "bf16"
    in1_dtype: str = "bf8"
    activation: Optional[str] = None

    @property
    def flops(self) -> int:
        return self.M * self.K * self.N

    def shape_str(self) -> str:
        a = f" +{self.activation}" if self.activation else ""
        return f"({self.M}x{self.K})@({self.K}x{self.N}){a}"


def _make_operands(cfg: MatmulCfg, device, mem) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    in0 = ttnn.from_torch(
        torch.randn(cfg.M, cfg.K) * 0.1,
        dtype=_DTYPES[cfg.in0_dtype],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
    )
    try:
        in1 = ttnn.from_torch(
            torch.randn(cfg.K, cfg.N) * 0.1,
            dtype=_DTYPES[cfg.in1_dtype],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem,
        )
    except RuntimeError:
        ttnn.deallocate(in0)
        raise
    return in0, in1


def _run_once(in0, in1, cfg: MatmulCfg, mem) -> ttnn.Tensor:
    return ttnn.linear(
        in0,
        in1,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        compute_kernel_config=_COMPUTE_KERNEL_CONFIG,
        activation=cfg.activation,
    )


def time_matmul(device, cfg: MatmulCfg, mem) -> Optional[Stats]:
    """Time one matmul with all operands+output on ``mem``.

    Returns (mean, stdev, min, max) in ms, or None if the placement does not
    fit (L1 OOM). Operands are built once and kept resident, so only the matmul
    op is timed.
    """
    try:
        in0, in1 = _make_operands(cfg, device, mem)
    except RuntimeError:
        return None

    try:
        for _ in range(NUM_WARMUP):  # compile + populate program cache
            out = _run_once(in0, in1, cfg, mem)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        samples: List[float] = []
        for _ in range(NUM_ITER):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = _run_once(in0, in1, cfg, mem)
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
    except RuntimeError:
        ttnn.deallocate(in0)
        ttnn.deallocate(in1)
        return None

    ttnn.deallocate(in0)
    ttnn.deallocate(in1)
    return (
        statistics.mean(samples),
        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min(samples),
        max(samples),
    )


def run_stage(device, stage: str, configs: List[MatmulCfg]) -> None:
    """Bench every config on DRAM and L1, print a live line then a summary table."""
    print("\n" + "=" * 96)
    print(f"  {stage}   matmul L1-vs-DRAM   (warmup={NUM_WARMUP}, iter={NUM_ITER})")
    print("  speedup = DRAM_mean / L1_mean   (>1.0 means L1 is faster)")
    print("=" * 96)

    rows: List[Tuple[MatmulCfg, Optional[Stats], Optional[Stats]]] = []
    for cfg in configs:
        dram = time_matmul(device, cfg, DRAM)
        l1 = time_matmul(device, cfg, L1)
        rows.append((cfg, dram, l1))
        d = f"{dram[0]:.3f}" if dram else "n/a"
        if l1 is None:
            live = f"L1={'OOM':>9}     speedup={'—':>6}"
        elif dram is None:
            live = f"L1={l1[0]:>9.3f} ms  speedup={'—':>6}"
        else:
            live = f"L1={l1[0]:>9.3f} ms  speedup={dram[0] / l1[0]:>5.2f}x"
        print(f"  {cfg.name:<10} {cfg.shape_str():<30}  DRAM={d:>9} ms  {live}")
        ttnn.synchronize_device(device)

    max_flops = max(c.flops for c in configs)
    fitting_dram = [r[1][0] for r in rows if r[1] is not None]
    max_dram_time = max(fitting_dram) if fitting_dram else None

    print("\n" + "-" * 96)
    print(f"  {'matmul':<10} {'shape':<30}  {'DRAM ms':>9}  {'L1 ms':>9}  " f"{'L1 fits':>8}  {'speedup':>8}")
    print("-" * 96)
    for cfg, dram, l1 in rows:
        flags = ""
        if cfg.flops == max_flops:
            flags += "  <-max size"
        if dram is not None and max_dram_time is not None and dram[0] == max_dram_time:
            flags += "  <-max time"
        d = f"{dram[0]:.3f}" if dram else "n/a"
        if l1 is None:
            l, fits, sp = "—", "NO", "—"
        else:
            l = f"{l1[0]:.3f}"
            fits = "yes"
            sp = f"{dram[0] / l1[0]:.2f}x" if dram else "—"
        print(f"  {cfg.name:<10} {cfg.shape_str():<30}  {d:>9}  {l:>9}  {fits:>8}  {sp:>8}{flags}")
    print("=" * 96)
