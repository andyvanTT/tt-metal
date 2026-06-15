# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D2D socket-transport tests on the 4-chip Blackhole mesh.

Mirrors the host-bounce probe in test_pcc_tt_bh_glx_stages.py::test_mesh_carve_smoke,
but exercises the fabric socket path (SocketTransport.send -> send_direct_async /
recv_direct_async). No checkpoint required — uses a random probe tensor.

Run:
    pytest -xvs models/experimental/pi0_5/blk_hole/test_d2d_transport_bh.py
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport
from ._common import SEED


def _probe_on(chip):
    torch.manual_seed(SEED)
    probe = torch.randn(1, 32, 32, dtype=torch.float32)  # tile-aligned
    x = ttnn.from_torch(
        probe,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=chip,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return probe, x


def test_d2d_single_hop(bh_mesh):
    """chip0 -> chip1 socket round-trip."""
    transport = SocketTransport()
    probe, x0 = _probe_on(bh_mesh.chips[0])

    x1 = transport.send(x0, bh_mesh.chips[1])
    out = ttnn.to_torch(x1)

    assert out.shape == probe.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(probe.shape)}"
    # bf16 round-trip — loose tolerance suffices (matches test_mesh_carve_smoke).
    assert torch.allclose(out.float(), probe, atol=1e-2), "socket round-trip altered the data"
    print("\n✅ D2D single hop chip0->chip1 OK")


def test_d2d_four_chip_chain(bh_mesh):
    """chip0 -> chip1 -> chip2 -> chip3 socket chain (3 distinct pairs, no tag needed)."""
    transport = SocketTransport()
    probe, x = _probe_on(bh_mesh.chips[0])

    for i in range(3):
        x = transport.send(x, bh_mesh.chips[i + 1])
    out = ttnn.to_torch(x)

    assert out.shape == probe.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(probe.shape)}"
    assert torch.allclose(out.float(), probe, atol=1e-2), "4-chip socket chain altered the data"
    print("\n✅ D2D 4-chip chain chip0->1->2->3 OK")
