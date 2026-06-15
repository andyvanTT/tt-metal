# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""4-chip Blackhole mesh harness for per-slice testing.

Auto-detects the local device count and opens a parent mesh, then carves four
1x1 submeshes that the slice tests place individual layers on:

    ttnn.get_num_devices() == 4  -> parent MeshShape(4,1), chips at (0,0)..(3,0)
    ttnn.get_num_devices() == 32 -> parent MeshShape(8,4), 4-chip corner (1,4)
                                    at offset (0,0): chips at (0,0)..(0,3)

The galaxy must open its full (8,4) parent (a partial open hangs fabric
handshakes), so on a 32-chip box we open the whole mesh and only use a corner.

FABRIC_2D is enabled before open so the D2D socket transport
(``SocketTransport`` in ``tt/tt_bh_glx/transport.py``) can route between the
carved 1x1 submeshes. Teardown mirrors ``open_galaxy_mesh`` exactly:
submeshes-before-parent, then disable fabric.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import ttnn

# Reuse the galaxy carver — do not reimplement.
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import _carve_per_chip

_DEFAULT_TRACE_REGION_SIZE = 134_217_728  # 128 MiB — matches the galaxy default.


@dataclass
class BHMeshHandle:
    """Live handles for a 4-chip Blackhole mesh."""

    parent: object
    chips: List[object] = field(default_factory=list)


@contextmanager
def open_blackhole_mesh(
    l1_small_size: Optional[int] = 24576,
    trace_region_size: Optional[int] = _DEFAULT_TRACE_REGION_SIZE,
    enable_fabric: bool = True,
):
    """Open a 4-chip mesh and carve 4 single-chip submeshes. Yields BHMeshHandle.

    enable_fabric=True (default) is required for the D2D socket transport test;
    the slice PCC tests don't need it but it is harmless for them.
    """
    n = ttnn.get_num_devices()
    if n == 4:
        parent_shape = ttnn.MeshShape(4, 1)
        corner_shape, corner_offset = (4, 1), (0, 0)
    elif n == 32:
        parent_shape = ttnn.MeshShape(8, 4)
        corner_shape, corner_offset = (1, 4), (0, 0)
    else:
        raise RuntimeError(f"open_blackhole_mesh requires 4 or 32 devices, got {n}")

    if enable_fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    open_kwargs = {"mesh_shape": parent_shape}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size

    parent = ttnn.open_mesh_device(**open_kwargs)
    chips: List[object] = []
    try:
        chips = _carve_per_chip(parent, corner_shape, corner_offset, 4)
        yield BHMeshHandle(parent=parent, chips=chips)
    finally:
        for sm in reversed(chips):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
