# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert-MLP output-sharding sweep (single Gemma-300M action-expert block).

Sweeps the output ``memory_config`` of the GeGLU MLP gate/up/down matmuls
(``GemmaMLPTTNN``) across {L1, DRAM} x {interleaved, block, width, height} on
one ``AdaRMSGemmaBlockTTNN`` forward — the denoise action-expert block — times
each config, PCC-checks every config against the L1-interleaved baseline, and
plots the timings.

The knob is ``GemmaMLPTTNN.output_memcfg_mode`` (see ttnn_gemma.py); the default
``l1_interleaved`` reproduces production behavior byte-for-byte. Configs the
matmul cannot emit are recorded N/A:
  - DRAM-sharded: TT-NN has no predefined DRAM sharded MemoryConfig and
    create_sharded_memory_config is L1-only (docs), so these raise on resolve.
  - L1 block/height: the expert's small M (S=32 -> 1 tile) drives a 1D
    width-sharded program config; layouts it can't produce are rejected by the
    matmul and shown as N/A.

Activations and weights are held fixed (L1 interleaved acts, DRAM weights) so the
*only* variable is the MLP output placement.

Run:
    PI0_BENCH_EXPERT_SHARD_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_expert_mlp_sharding_sweep.py

Or under tracy:
    PI0_BENCH_EXPERT_SHARD_SWEEP=1 python -m tracy -p -r -v --op-support-count 100000 \\
      -m pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_expert_mlp_sharding_sweep.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import List, Optional

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tests.perf.test_expert_block_l1_vs_dram import (
    BATCH,
    MLP_DIM,
    S,
    WIDTH,
    _build_expert_block,
    _free_block,
    _real_or_random_expert_layer,
    _upload,
)

BENCH_ENABLED = os.environ.get("PI0_BENCH_EXPERT_SHARD_SWEEP") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_BENCH_EXPERT_SHARD_SWEEP=1 to run the expert-MLP output-sharding sweep",
)

NUM_WARMUP = int(os.environ.get("PI0_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_BENCH_ITER", "100"))
PCC_THRESHOLD = float(os.environ.get("PI0_BENCH_PCC", "0.99"))

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# Full matrix: {location} x {layout}.
CONFIGS = [
    "l1_interleaved",
    "l1_width",
    "l1_block",
    "l1_height",
    "dram_interleaved",
    "dram_width",
    "dram_block",
    "dram_height",
]

PLOT_PATH = Path(os.environ.get("PI0_BENCH_PLOT", "generated/profiler/expert_mlp_sharding_sweep.png"))


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Pearson Correlation Coefficient (matches the pi0.5 pcc test convention)."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


class Result:
    """One swept config's outcome."""

    def __init__(self, cfg: str):
        self.cfg = cfg
        self.status = "pending"
        self.mean: Optional[float] = None
        self.stdev: Optional[float] = None
        self.mn: Optional[float] = None
        self.mx: Optional[float] = None
        self.pcc: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.mean is not None


def _forward(block, h_host: torch.Tensor, adarms_tt, mask_tt) -> "ttnn.Tensor":
    """One expert-block forward with a freshly-uploaded (L1) activation."""
    h = _upload(h_host, block.device, ttnn.bfloat16, L1)
    out, _ = block.forward(
        h,
        block._bench_cos,
        block._bench_sin,
        adarms_tt,
        attention_mask=mask_tt,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    )
    ttnn.deallocate(h)
    return out


def _sweep_one(device, block, cfg: str, h_host, adarms_tt, mask_tt, baseline_torch: torch.Tensor) -> Result:
    r = Result(cfg)
    block.mlp.output_memcfg_mode = cfg

    # Warmup + probe: any unsupported config raises here (resolve-time
    # NotImplementedError for dram-sharded, or matmul validation RuntimeError
    # for layouts the program config can't emit). Catch -> mark N/A.
    probe_torch: Optional[torch.Tensor] = None
    try:
        for i in range(NUM_WARMUP):
            out = _forward(block, h_host, adarms_tt, mask_tt)
            if i == NUM_WARMUP - 1:
                probe_torch = ttnn.to_torch(out)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
    except Exception as e:  # noqa: BLE001 - we want to record any failure as N/A
        ttnn.synchronize_device(device)
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        r.status = f"N/A ({type(e).__name__})"
        print(f">> {cfg:18s}  N/A  {first_line[:90]}")
        return r

    r.pcc = compute_pcc(baseline_torch, probe_torch) if probe_torch is not None else 0.0

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = _upload(h_host, device, ttnn.bfloat16, L1)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out, _ = block.forward(
            h,
            block._bench_cos,
            block._bench_sin,
            adarms_tt,
            attention_mask=mask_tt,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
        )
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    r.mean = statistics.mean(samples)
    r.stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    r.mn = min(samples)
    r.mx = max(samples)
    r.status = "OK" if r.pcc >= PCC_THRESHOLD else f"LOW_PCC({r.pcc:.4f})"
    print(f">> {cfg:18s}  {r.status:14s}  mean={r.mean:7.3f} ms  stdev={r.stdev:6.3f}  pcc={r.pcc:.5f}")
    return r


def _print_summary(results: List[Result]) -> None:
    base = next((r for r in results if r.cfg == "l1_interleaved" and r.ok), None)
    print("\n" + "=" * 84)
    print(f"  EXPERT MLP OUTPUT-SHARDING SWEEP  (B={BATCH}, S={S}, W={WIDTH}, mlp_dim={MLP_DIM}, iter={NUM_ITER})")
    print("=" * 84)
    print(f"  {'config':<18}  {'status':<16}  {'mean ms':>9}  {'stdev':>7}  {'min':>7}  {'vs base':>8}  {'pcc':>8}")
    for r in results:
        if not r.ok:
            print(f"  {r.cfg:<18}  {r.status:<16}  {'-':>9}  {'-':>7}  {'-':>7}  {'-':>8}  {'-':>8}")
            continue
        rel = (base.mean / r.mean) if (base is not None and r.mean) else 0.0
        print(
            f"  {r.cfg:<18}  {r.status:<16}  {r.mean:>9.3f}  {r.stdev:>7.3f}  {r.mn:>7.3f}  {rel:>7.2f}x  {r.pcc:>8.5f}"
        )
    print("=" * 84)


def _plot(results: List[Result]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r.cfg for r in results]
    means = [r.mean if r.ok else 0.0 for r in results]
    lo_err = [(r.mean - r.mn) if r.ok else 0.0 for r in results]
    hi_err = [(r.mx - r.mean) if r.ok else 0.0 for r in results]
    colors = ["tab:green" if r.ok else "lightgray" for r in results]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = range(len(labels))
    ax.bar(xs, means, color=colors, yerr=[lo_err, hi_err], capsize=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("expert block forward (ms, lower is better)")
    ax.set_title(
        f"pi0.5 expert MLP output-sharding sweep\n(B={BATCH}, S={S}, W={WIDTH}, mlp_dim={MLP_DIM}, iters={NUM_ITER})"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, r in enumerate(results):
        if r.ok:
            ax.text(i, r.mean, f"{r.mean:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i, 0.0, r.status, rotation=90, ha="center", va="bottom", fontsize=7, color="dimgray")

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    plt.close(fig)
    print(f"\n  plot saved -> {PLOT_PATH}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_expert_mlp_sharding_sweep(device):
    print("\n" + "=" * 84)
    print(f"  EXPERT MLP OUTPUT-SHARDING SWEEP  warmup={NUM_WARMUP}  iter={NUM_ITER}  pcc>={PCC_THRESHOLD}")
    print("=" * 84)

    # Weights DRAM, activations L1 — fixed; only the MLP output placement varies.
    raw = _real_or_random_expert_layer(0)
    block = _build_expert_block(device, DRAM, DRAM, raw=raw, layer_idx=0)
    h_host = torch.randn(BATCH, S, WIDTH) * 0.5
    adarms_tt = _upload(torch.randn(BATCH, 1, WIDTH) * 0.1, device, ttnn.bfloat16, DRAM)
    mask_tt = _upload(torch.zeros(BATCH, 1, S, S), device, ttnn.bfloat16, DRAM)

    try:
        # Baseline = production mode. Capture its output as the PCC reference.
        block.mlp.output_memcfg_mode = "l1_interleaved"
        base_out = _forward(block, h_host, adarms_tt, mask_tt)
        baseline_torch = ttnn.to_torch(base_out)
        ttnn.deallocate(base_out)
        ttnn.synchronize_device(device)

        results = [_sweep_one(device, block, cfg, h_host, adarms_tt, mask_tt, baseline_torch) for cfg in CONFIGS]
    finally:
        _free_block(block)
        ttnn.deallocate(adarms_tt)
        ttnn.deallocate(mask_tt)
        ttnn.synchronize_device(device)

    _print_summary(results)
    _plot(results)

    # Correctness gates: baseline must run, and every config that produced a
    # number must be numerically equivalent to it (sharding changes layout, not
    # math). N/A cells are allowed (unsupported layouts), but wrong ones are not.
    base = next((r for r in results if r.cfg == "l1_interleaved"), None)
    assert base is not None and base.ok, "baseline l1_interleaved did not run"
    bad = [(r.cfg, r.pcc) for r in results if r.ok and r.pcc < PCC_THRESHOLD]
    assert not bad, f"configs below PCC {PCC_THRESHOLD}: {bad}"
