# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the blk_hole per-slice tests.

- ``bh_mesh``  (module scope): one 4-chip mesh per test module (amortizes the
  expensive open + fabric setup across all slice tests in the module).
- ``loader`` / ``weights`` (session scope): the pi0.5 checkpoint, loaded once.
"""

from __future__ import annotations

import pytest

from .mesh_setup_bh import open_blackhole_mesh
from ._common import CHECKPOINT_DIR


@pytest.fixture(scope="module")
def bh_mesh():
    with open_blackhole_mesh(l1_small_size=24576) as h:
        yield h


@pytest.fixture(scope="session")
def loader():
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT_DIR}")
    return Pi0_5WeightLoader(str(CHECKPOINT_DIR))


@pytest.fixture(scope="session")
def weights(loader):
    return loader.categorized_weights
