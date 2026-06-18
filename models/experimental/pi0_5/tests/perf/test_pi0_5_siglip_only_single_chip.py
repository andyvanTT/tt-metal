# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 SigLIP-only performance — single chip, no cross-chip D2D.

Isolates the vision stage (SigLIP tower + multimodal projector) on ONE
Blackhole chip. The pre-stacked image tensor is uploaded ONCE outside the
timed region; only `backbone.embed_image` is captured as a TTNN trace and
replayed. This is the first of the per-stage single-chip trace tests; summed
with the VLM-prefill and denoise-only tests it should approximate the full
`sample_actions` e2e trace (the isolated stages pay extra DRAM<->L1 staging
at the boundaries that the fused e2e trace avoids, so the sum is close, not
exact).

What it reports:
  - SigLIP-only steady-state replay latency (ms/loop) at bs=NUM_CAMERAS
  - PCC of the traced output vs the eager (non-traced) output — guards the
    trace path

Run (production config — same flags as the full e2e trace):
  source _bench_runs/pi05_production.env
  python_env/bin/python -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py

Skipped if the checkpoint isn't present locally.
"""

import os
import re
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = int(os.environ.get("PI05_TRACE_NUM_WARMUP", "2"))
NUM_ITERS = int(os.environ.get("PI05_TRACE_NUM_ITERS", "20"))
NUM_DENOISE_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
LANG_SEQ_LEN = int(os.environ.get("PI05_LANG_SEQ_LEN", "256"))
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB
PCC_THRESHOLD = float(os.environ.get("PI05_SIGLIP_PCC", "0.99"))

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _apply_production_env_defaults() -> None:
    root = os.environ.get("TT_METAL_HOME") or str(Path(__file__).resolve().parents[5])
    envf = Path(root) / "_bench_runs" / "pi05_production.env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
        if not m or m.group(1) == "PI05_CHECKPOINT_DIR":
            continue
        os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_stacked_images(device, num_cameras: int):
    """Pre-stack all cameras on host into one (N, H, W/patch, C*patch) ROW_MAJOR
    tensor and upload once — matches the full e2e trace's PI0_SIGLIP_USE_FOLD path.
    Returns a single ttnn tensor (the SigLIP bs=N fast-path input)."""
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
    if use_fold:
        _PATCH = 14
        stacked = torch.cat([im.permute(0, 2, 3, 1).contiguous() for im in images], dim=0)
        n, h, w, c = stacked.shape
        stacked = stacked.reshape(n, h, w // _PATCH, c * _PATCH).contiguous()
        return ttnn.from_torch(
            stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    # Non-fold fallback: stack BCHW in TILE layout (slower device concat path).
    stacked = torch.cat(images, dim=0)
    return ttnn.from_torch(
        stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_siglip_only_single_chip(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    n_chips = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    assert n_chips == 1, f"single-chip test requires a 1-chip device handle, got {n_chips}."
    print(f"\n🔒 single-chip guard OK: device spans {n_chips} chip")

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=NUM_DENOISE_STEPS)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    model = Pi0_5ModelTTNN(cfg, loader, device)

    stacked = _build_stacked_images(device, NUM_CAMERAS)
    siglip_bs = int(stacked.shape[0])
    print(f"\n📦 stacked image input built once; SigLIP runs bs={siglip_bs}")

    def _siglip():
        return model.backbone.embed_image(stacked)

    # ---- Eager reference (non-traced) ----
    eager_out = _siglip()
    ttnn.synchronize_device(device)
    eager_feats = ttnn.to_torch(eager_out).float()
    assert torch.isfinite(eager_feats).all(), "eager SigLIP produced NaN/Inf"
    if isinstance(eager_out, ttnn.Tensor):
        ttnn.deallocate(eager_out)

    # ---- Warmup the trace path (JIT) ----
    for _ in range(NUM_WARMUP):
        out = _siglip()
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)

    # ---- Capture SigLIP-only trace ----
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _siglip()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    traced_feats = ttnn.to_torch(out_trace).float()
    assert torch.isfinite(traced_feats).all(), "traced SigLIP produced NaN/Inf"
    pcc = _pcc(traced_feats, eager_feats)

    # ---- Time steady-state replay ----
    times_ms: List[float] = []
    for _ in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    ttnn.release_trace(device, tid)

    avg = statistics.mean(times_ms)
    mn, mx = min(times_ms), max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print("\n" + "=" * 72)
    print(f"  PI0.5 SIGLIP-ONLY (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
    print("=" * 72)
    print(f"   Config:            cameras={NUM_CAMERAS} (SigLIP bs={siglip_bs})")
    print(f"   Trace capture:     {capture_ms:7.2f} ms (one-time)")
    print(f"   SigLIP-only avg:   {avg:7.2f} ms/loop")
    print(f"   Per-call min/max:  {mn:7.2f} / {mx:7.2f} ms   stddev {sd:.2f}")
    print(f"   PCC traced vs eager: {pcc:.6f}  (threshold {PCC_THRESHOLD})")
    print("=" * 72)

    assert pcc >= PCC_THRESHOLD, f"SigLIP trace PCC {pcc:.6f} < {PCC_THRESHOLD}"
    assert avg > 0
