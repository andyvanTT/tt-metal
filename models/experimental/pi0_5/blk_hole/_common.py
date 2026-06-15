# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants + helpers for the blk_hole per-slice tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

SEED = 42

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))

# Skip every PCC test (which needs real weights) when the checkpoint is absent.
# The D2D transport test does not import this mark — it uses a random probe.
requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, matching tests/pcc/test_pcc_tt_bh_glx_stages.py."""
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()
