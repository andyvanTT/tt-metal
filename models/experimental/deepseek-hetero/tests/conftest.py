# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures for deepseek-hetero.

Provides a single-chip mesh device and ensures the (hyphenated) package dir is on
``sys.path`` so tests import sibling modules by top-level name.
"""

import os
import sys

import pytest

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import ttnn  # noqa: E402


@pytest.fixture(scope="session")
def mesh():
    """A 1x1 Blackhole mesh device, opened once per test session."""
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    ttnn.close_mesh_device(device)
