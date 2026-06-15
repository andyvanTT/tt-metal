# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-slice PCC test for one expert chunk (3 AdaRMS blocks) on a 4-chip Blackhole.

ExpertChunkSlice(layer_range=(0,3)) vs the torch AdaRMS expert blocks 0-2
(Pi0_5PaliGemmaBackbone.expert_blocks), with synthetic prefix KV. Stops before
the final ada-RMS norm (that lives downstream on the last chip, not in the chunk).

Synthetic prefix KV is uploaded bf8_b to match the expert's internal k_rope
dtype (see kv_migration.py); the torch reference keeps fp32 values, which costs
~3 PCC points -> threshold 0.95 (the project e2e bar).

Run:
    pytest -xvs models/experimental/pi0_5/blk_hole/test_expert_slice_bh.py
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.reference.torch_gemma import precompute_freqs_cis
from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import ExpertChunkSlice
from ._common import CHECKPOINT_DIR, SEED, compute_pcc, requires_checkpoint

pytestmark = requires_checkpoint

PCC_THRESHOLD = 0.95
N_LAYERS = 3


def test_expert_chunk_slice_pcc(bh_mesh, weights):
    """3 AdaRMS expert blocks (layers 0-2) on chip 0 vs torch reference."""
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=5,
    )
    ew = cfg.expert_config.width  # 1024
    head_dim = cfg.expert_config.head_dim  # 256
    num_kv = cfg.expert_config.num_kv_heads  # 1
    suffix_len = ((cfg.action_horizon + 31) // 32) * 32
    prefix_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "256"))

    torch.manual_seed(SEED)
    hidden_in = torch.randn(1, suffix_len, ew) * 0.5
    adarms_cond = torch.randn(1, ew) * 0.5
    prefix_kv_torch = [
        (torch.randn(1, num_kv, prefix_len, head_dim) * 0.5, torch.randn(1, num_kv, prefix_len, head_dim) * 0.5)
        for _ in range(N_LAYERS)
    ]

    # Reference: chain expert blocks 0-2 with the same fp32 prefix KV. No final norm.
    backbone = Pi0_5PaliGemmaBackbone(cfg, weights)
    cos, sin = precompute_freqs_cis(head_dim, cfg.max_seq_len, cfg.expert_config.rope_base)
    with torch.no_grad():
        ref = hidden_in
        for i in range(N_LAYERS):
            ref, _ = backbone.expert_blocks[i].forward(
                ref,
                cos,
                sin,
                adarms_cond,
                attention_mask=None,
                position_ids=None,
                past_key_value=prefix_kv_torch[i],
                use_cache=False,
            )

    chip = bh_mesh.chips[0]
    slice_ = ExpertChunkSlice(
        cfg.expert_config, weights["action_expert"], chip, layer_range=(0, N_LAYERS), max_seq_len=cfg.max_seq_len
    )

    hidden_ttnn = ttnn.from_torch(hidden_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=chip)
    adarms_ttnn = ttnn.from_torch(
        adarms_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=chip, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # bf8_b prefix KV matches the expert's internal k_rope dtype (kv_migration.py).
    prefix_kv_ttnn = [
        (
            ttnn.from_torch(
                kt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=chip, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            ttnn.from_torch(
                vt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=chip, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        for (kt, vt) in prefix_kv_torch
    ]

    out = ttnn.to_torch(slice_.forward(hidden_ttnn, adarms_ttnn, prefix_kv_ttnn))

    assert out.shape == ref.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"
    pcc = compute_pcc(ref, out)
    print(f"\n✅ ExpertChunkSlice (layers 0-2) PCC: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
