# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-slice PCC tests for the pi0.5 suffix projections on a 4-chip Blackhole.

SuffixSlice (embed_actions / embed_adarms_cond / project_output) vs
reference/torch_suffix.py Pi0_5SuffixEmbedding, on chip 0. Uses
loader.get_pi0_projections() (the exact dict the existing suffix PCC test uses).

Run:
    pytest -xvs models/experimental/pi0_5/blk_hole/test_suffix_slice_bh.py
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
from models.experimental.pi0_5.tt.tt_bh_glx.suffix_slice import SuffixSlice
from ._common import CHECKPOINT_DIR, SEED, compute_pcc, requires_checkpoint

pytestmark = requires_checkpoint

PCC_THRESHOLD = 0.93


def _suffix_cfg() -> SuffixConfig:
    return SuffixConfig(
        action_dim=32,
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        expert_width=1024,
        state_dim=32,
        time_emb_dim=1024,
        pi05=True,
    )


def test_suffix_embed_actions_pcc(bh_mesh, loader):
    cfg = _suffix_cfg()
    proj = loader.get_pi0_projections()
    torch.manual_seed(SEED)
    noisy_actions = torch.randn(1, cfg.action_horizon, cfg.action_dim)

    ref = Pi0_5SuffixEmbedding(cfg, proj).embed_actions(noisy_actions)

    slice_ = SuffixSlice(cfg, proj, bh_mesh.chips[0])
    actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[0])
    out = ttnn.to_torch(slice_.embed_actions(actions_ttnn))

    pcc = compute_pcc(ref, out)
    print(f"\n✅ SuffixSlice.embed_actions PCC: {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


def test_suffix_embed_adarms_cond_pcc(bh_mesh, loader):
    cfg = _suffix_cfg()
    proj = loader.get_pi0_projections()
    timestep = torch.tensor([0.5], dtype=torch.float32)

    ref = Pi0_5SuffixEmbedding(cfg, proj).embed_timestep_adarms(timestep)

    slice_ = SuffixSlice(cfg, proj, bh_mesh.chips[0])
    t_ttnn = ttnn.from_torch(timestep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[0])
    out = ttnn.to_torch(slice_.embed_adarms_cond(t_ttnn))

    pcc = compute_pcc(ref, out)
    print(f"\n✅ SuffixSlice.embed_adarms_cond PCC: {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


def test_suffix_project_output_pcc(bh_mesh, loader):
    cfg = _suffix_cfg()
    proj = loader.get_pi0_projections()
    torch.manual_seed(SEED)
    expert_output = torch.randn(1, cfg.action_horizon, cfg.expert_width)

    ref = Pi0_5SuffixEmbedding(cfg, proj).project_output(expert_output)

    slice_ = SuffixSlice(cfg, proj, bh_mesh.chips[0])
    expert_ttnn = ttnn.from_torch(expert_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bh_mesh.chips[0])
    out = ttnn.to_torch(slice_.project_output(expert_ttnn))

    pcc = compute_pcc(ref, out)
    print(f"\n✅ SuffixSlice.project_output PCC: {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
