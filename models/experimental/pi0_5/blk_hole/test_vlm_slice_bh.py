# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-slice PCC test for one VLM (Gemma-2B) block on a 4-chip Blackhole.

VLMBlockSlice(layer_idx=0) vs the torch Gemma block at layer 0
(Pi0_5PaliGemmaBackbone.vlm_blocks[0]), sequential RoPE, no mask, no past-KV.

Run:
    pytest -xvs models/experimental/pi0_5/blk_hole/test_vlm_slice_bh.py
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone
from models.experimental.pi0_5.tt.tt_bh_glx.vlm_slice import VLMBlockSlice
from ._common import CHECKPOINT_DIR, SEED, compute_pcc, requires_checkpoint

pytestmark = requires_checkpoint

PCC_THRESHOLD = 0.99


def test_vlm_block_slice_pcc(bh_mesh, weights):
    """One Gemma-2B block (layer 0) on chip 0 vs torch reference."""
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=5,
    )
    seq_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "256"))
    vlm_width = cfg.vlm_config.width
    torch.manual_seed(SEED)
    hidden_in = torch.randn(1, seq_len, vlm_width) * 0.5

    # Reference: the exact layer-0 Gemma block + the backbone's RoPE tables.
    backbone = Pi0_5PaliGemmaBackbone(cfg, weights)
    blk = backbone.vlm_blocks[0]
    with torch.no_grad():
        ref, _ = blk.forward(
            hidden_in,
            backbone.cos,
            backbone.sin,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            use_cache=True,
        )

    slice_ = VLMBlockSlice(
        cfg.vlm_config, weights["vlm_language"], bh_mesh.chips[0], layer_idx=0, max_seq_len=cfg.max_seq_len
    )
    hidden_ttnn = ttnn.from_torch(hidden_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[0])
    out_ttnn, _kv = slice_.forward(hidden_ttnn)
    out = ttnn.to_torch(out_ttnn)

    assert out.shape == ref.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"
    pcc = compute_pcc(ref, out)
    print(f"\n✅ VLMBlockSlice (layer 0) PCC: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
