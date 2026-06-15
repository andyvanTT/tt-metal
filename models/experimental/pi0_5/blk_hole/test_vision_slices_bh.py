# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-slice PCC tests for the SigLIP vision slices on a 4-chip Blackhole.

Each slice is built on one chip and compared to the matching torch reference at
the same granularity (reference/torch_siglip.py).

Run:
    pytest -xvs models/experimental/pi0_5/blk_hole/test_vision_slices_bh.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.reference.torch_siglip import (
    MultiModalProjector,
    PatchEmbedding,
    SigLIPVisionTower,
)
from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import (
    SigLIPEmbedSlice,
    SigLIPLayerSlice,
    SigLIPTailSlice,
)
from ._common import SEED, compute_pcc, requires_checkpoint

pytestmark = requires_checkpoint

PCC_THRESHOLD = 0.997


def _cfg() -> SigLIPConfig:
    return SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )


def _num_cameras() -> int:
    return int(os.environ.get("PI0_NUM_CAMERAS", "3"))


def test_embed_slice_pcc(bh_mesh, weights):
    """SigLIPEmbedSlice (chip 0): patch_embed + position_embedding."""
    cfg = _cfg()
    vw = weights["vlm_vision"]
    B = _num_cameras()
    torch.manual_seed(SEED)
    pixel_values = torch.randn(B, 3, cfg.image_size, cfg.image_size)

    # Reference: patch embed + position embedding (num_patches == num_positions, no interp).
    pe = PatchEmbedding(cfg, vw)
    pos = vw.get("position_embedding.weight") or vw.get("vision_model.embeddings.position_embedding.weight")
    with torch.no_grad():
        patches = pe.forward(pixel_values)  # (B, 256, 1152), fp32
        ref = patches + pos.to(patches.dtype)

    slice_ = SigLIPEmbedSlice(cfg, vw, bh_mesh.chips[0])
    # Mirror StageVision.run: upload pixel_values to the chip before the slice.
    px_ttnn = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=bh_mesh.chips[0],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(slice_.forward(px_ttnn))

    assert out.shape == ref.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"
    pcc = compute_pcc(ref, out)
    print(f"\n✅ SigLIPEmbedSlice PCC: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("layer_range,chip", [((0, 9), 1), ((9, 18), 2)], ids=["layers0-8", "layers9-17"])
def test_layer_slice_pcc(bh_mesh, weights, layer_range, chip):
    """SigLIPLayerSlice (chips 1/2): a contiguous range of SigLIP blocks."""
    cfg = _cfg()
    vw = weights["vlm_vision"]
    B = _num_cameras()
    lo, hi = layer_range
    torch.manual_seed(SEED)
    hidden_in = torch.randn(B, 256, cfg.hidden_size) * 0.5

    # Reference: chain the same torch blocks over the same range.
    tower = SigLIPVisionTower(cfg, vw)
    with torch.no_grad():
        ref = hidden_in
        for i in range(lo, hi):
            ref = tower.blocks[i].forward(ref)

    slice_ = SigLIPLayerSlice(cfg, vw, bh_mesh.chips[chip], layer_range=layer_range)
    hidden_ttnn = ttnn.from_torch(hidden_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[chip])
    out = ttnn.to_torch(slice_.forward(hidden_ttnn))

    assert out.shape == ref.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"
    pcc = compute_pcc(ref, out)
    print(f"\n✅ SigLIPLayerSlice{layer_range} PCC: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


def test_tail_slice_pcc(bh_mesh, weights):
    """SigLIPTailSlice (chip 3): last blocks + post_layernorm + mm_projector."""
    cfg = _cfg()
    vw = weights["vlm_vision"]
    pw = weights["vlm_projector"]
    B = _num_cameras()
    lo, hi = 18, 27
    torch.manual_seed(SEED)
    hidden_in = torch.randn(B, 256, cfg.hidden_size) * 0.5

    # Reference: blocks 18-26 -> post_layernorm -> mm_projector.
    tower = SigLIPVisionTower(cfg, vw)
    proj = MultiModalProjector(pw)
    with torch.no_grad():
        ref = hidden_in
        for i in range(lo, hi):
            ref = tower.blocks[i].forward(ref)
        post_ln_b = tower.post_layernorm_bias.to(ref.dtype) if tower.post_layernorm_bias is not None else None
        ref = F.layer_norm(
            ref,
            (cfg.hidden_size,),
            tower.post_layernorm_weight.to(ref.dtype),
            post_ln_b,
            cfg.layer_norm_eps,
        )
        ref = proj.forward(ref)  # (B, 256, 2048)

    slice_ = SigLIPTailSlice(cfg, vw, pw, bh_mesh.chips[3], layer_range=(lo, hi))
    hidden_ttnn = ttnn.from_torch(hidden_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[3])
    out = ttnn.to_torch(slice_.forward(hidden_ttnn))

    assert out.shape == ref.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"
    pcc = compute_pcc(ref, out)
    print(f"\n✅ SigLIPTailSlice PCC: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
