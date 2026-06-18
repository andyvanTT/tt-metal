# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 VLM-prefill-only performance — single chip, no cross-chip D2D.

Isolates the VLM prefill stage (18 Gemma-2B blocks over the prefix tokens +
final norm, producing the prefix KV cache) on ONE Blackhole chip. The prefix
embeddings (SigLIP output + language embeddings, concatenated) are built ONCE
outside the timed region; only `backbone.forward_vlm` is captured as a TTNN
trace and replayed.

Second of the per-stage single-chip trace tests. Summed with the SigLIP-only
and denoise-only tests it approximates the full `sample_actions` e2e trace —
close, not exact, since the isolated stages pay DRAM<->L1 staging at the
boundaries that the fused e2e trace avoids.

What it reports:
  - VLM-prefill-only steady-state replay latency (ms/loop) at prefix_len = the
    production length (NUM_CAMERAS * 256 image tokens + LANG_SEQ_LEN)
  - PCC of the traced final hidden state vs the eager (non-traced) output

Run (production config — same flags as the full e2e trace):
  source _bench_runs/pi05_production.env
  python_env/bin/python -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_vlm_prefill_only_single_chip.py

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
PCC_THRESHOLD = float(os.environ.get("PI05_VLM_PCC", "0.99"))

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


def _build_inputs(device, num_cameras: int):
    """Identical to the full e2e / denoise-only input builder so the prefix
    embeddings (and thus prefix_len) match the fused e2e trace exactly."""
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
    if use_fold:
        _PATCH = 14
        stacked = torch.cat([im.permute(0, 2, 3, 1).contiguous() for im in images], dim=0)
        n, h, w, c = stacked.shape
        stacked = stacked.reshape(n, h, w // _PATCH, c * _PATCH).contiguous()
        images_ttnn = [
            ttnn.from_torch(
                stacked,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]
    else:
        images_ttnn = [
            ttnn.from_torch(
                im,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for im in images
        ]
    img_masks_ttnn = [
        ttnn.from_torch(
            m.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for m in img_masks
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_vlm_prefill_only_single_chip(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN, use_upstream_masks

    n_chips = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    assert n_chips == 1, f"single-chip test requires a 1-chip device handle, got {n_chips}."
    print(f"\n🔒 single-chip guard OK: device spans {n_chips} chip")

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=NUM_DENOISE_STEPS)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    model = Pi0_5ModelTTNN(cfg, loader, device)

    imgs, im_masks, lt, lm = _build_inputs(device, NUM_CAMERAS)

    # ---- Build prefix embeddings ONCE (SigLIP + lang embed + concat), OUTSIDE
    #      the timed region. The VLM prefill is the only thing we trace.
    prefix_embs, _, _ = model.embed_prefix(imgs, im_masks, lt, lm)
    if prefix_embs.layout != ttnn.TILE_LAYOUT:
        prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)
    prefix_len = int(prefix_embs.shape[1])
    ttnn.synchronize_device(device)
    print(f"\n📦 prefix embeddings built once; prefix_len={prefix_len}; VLM prefill is the measured region")

    # Pre-stage upstream-compat artifacts (mask + RoPE) so forward_vlm does no
    # host->device transfer inside the captured trace.
    attn_mask = cos_o = sin_o = None
    if use_upstream_masks():
        model.prepare_upstream_artifacts(im_masks, lm, prefix_len=prefix_len)
        art = model._cached_upstream_artifacts
        attn_mask, cos_o, sin_o = art["prefix_attn_mask"], art["prefix_cos"], art["prefix_sin"]

    def _vlm():
        hidden, _ = model.backbone.forward_vlm(
            prefix_embs,
            attention_mask=attn_mask,
            cos_override=cos_o,
            sin_override=sin_o,
            use_cache=True,
        )
        return hidden

    # ---- Eager reference (non-traced) ----
    eager_hidden = _vlm()
    ttnn.synchronize_device(device)
    eager_t = ttnn.to_torch(eager_hidden).float()
    assert torch.isfinite(eager_t).all(), "eager VLM prefill produced NaN/Inf"
    ttnn.deallocate(eager_hidden)

    # ---- Warmup the trace path (JIT) ----
    for _ in range(NUM_WARMUP):
        h = _vlm()
        ttnn.synchronize_device(device)
        ttnn.deallocate(h)

    # ---- Capture VLM-prefill-only trace ----
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _vlm()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    traced_t = ttnn.to_torch(out_trace).float()
    assert torch.isfinite(traced_t).all(), "traced VLM prefill produced NaN/Inf"
    pcc = _pcc(traced_t, eager_t)

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
    print(f"  PI0.5 VLM-PREFILL-ONLY (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
    print("=" * 72)
    print(f"   Config:            cameras={NUM_CAMERAS}, prefix_len={prefix_len}")
    print(f"   Trace capture:     {capture_ms:7.2f} ms (one-time)")
    print(f"   VLM-prefill avg:   {avg:7.2f} ms/loop")
    print(f"   Per-call min/max:  {mn:7.2f} / {mx:7.2f} ms   stddev {sd:.2f}")
    print(f"   PCC traced vs eager: {pcc:.6f}  (threshold {PCC_THRESHOLD})")
    print("=" * 72)

    assert pcc >= PCC_THRESHOLD, f"VLM prefill trace PCC {pcc:.6f} < {PCC_THRESHOLD}"
    assert avg > 0
