# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused test for Option C inter-submesh KV migration over MeshSockets.

Validates `KVMigration.migrate_layer_paired_socket` — the inter-submesh d2d
hand-off (prefill chip -> denoise chip) — against the host-bounce baseline
(`migrate_layer_paired`), on synthetic per-layer K/V tensors of realistic size.
No model forward / checkpoint needed: it isolates the transport mechanism and
checks (a) correctness (bytes land intact) and (b) per-transport wall time.

Endpoints are carved DIRECTLY from the 8x4 parent at the real Option C physical
coords (prefill chip i = (2 + i//3, i%3); denoise chip d = (2 + d, 3)), so this
is the true non-adjacent hop (e.g. (2,0) -> (2,3), 3 columns apart). That hop
needs FABRIC_2D routing — FABRIC_1D raises "no forwarding direction" for a socket.

Socket op (FIFO vs direct-write) is chosen by PI0_OC_SOCKET_OP (fifo|direct); the
direct ops only exist on the sdawle_blaze_socket_direct branch, so this file runs
identically on both branches and reports which op it used.

Run (exclusive 32-chip cluster):
    PI0_OC_SOCKET_OP=fifo \
    python -m pytest models/experimental/pi0_5/tests/test_oc_kv_socket.py -s
"""

from __future__ import annotations

import gc
import time

import pytest
import torch

import ttnn

from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration
from models.experimental.pi0_5.tt.option_c.stages import DENOISE_SUBMESH_OFFSET
from models.experimental.pi0_5.tt.option_c.transport import resolve_socket_ops

# Realistic per-layer KV footprint (~1MB bf16/tensor), tile-aligned.
KV_SHAPE = [1, 1, 256, 2048]
VLM_DEPTH = 2  # layers 0,1 → prefill (2,0),(2,1) → both land on denoise chip 0 (2,3)
PARENT_SHAPE = (8, 4)


def _make_kv_on(micro_submesh, seed: int):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(KV_SHAPE, generator=gen, dtype=torch.float32)
    v = torch.randn(KV_SHAPE, generator=gen, dtype=torch.float32)

    def _up(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=micro_submesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(micro_submesh),
        )

    return _up(k), _up(v), k, v


@pytest.mark.timeout(600)
def test_kv_migration_socket_vs_host_bounce():
    """Migrate VLM_DEPTH layers of synthetic K/V prefill-chip -> denoise-chip via
    (1) host bounce and (2) socket; verify both land intact and report timings."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*PARENT_SHAPE))

    prefill_micro = []
    denoise_chips = sorted({KVMigration.denoise_chip_for_vlm_layer(i) for i in range(VLM_DEPTH)})
    denoise_micro = [None] * (max(denoise_chips) + 1)
    mig_sock = None
    try:
        # (1,1) endpoints at the real Option C parent coords.
        for i in range(VLM_DEPTH):
            r, c = KVMigration._prefill_parent_coord(i)
            prefill_micro.append(parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c)))
        for d in denoise_chips:
            r, c = DENOISE_SUBMESH_OFFSET[0] + d, DENOISE_SUBMESH_OFFSET[1]
            denoise_micro[d] = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))

        # Synthetic per-layer KV on each prefill chip + golden host copies.
        per_layer_kv, golden = [], []
        for i in range(VLM_DEPTH):
            k_dev, v_dev, k_host, v_host = _make_kv_on(prefill_micro[i], seed=100 + i)
            per_layer_kv.append((k_dev, v_dev))
            golden.append((k_host, v_host))

        _s, _r, op_name = resolve_socket_ops()
        src = KVMigration._prefill_parent_coord(0)
        dst = (DENOISE_SUBMESH_OFFSET[0], DENOISE_SUBMESH_OFFSET[1])
        print(f"\n[kv-socket] op={op_name} layers={VLM_DEPTH} hop={src}->{dst} kv_shape={KV_SHAPE}", flush=True)

        # ---- host-bounce baseline ----
        mig_host = KVMigration(denoise_submesh=denoise_micro[denoise_chips[0]])
        t0 = time.perf_counter()
        mig_host.migrate_layer_paired(per_layer_kv, denoise_micro_submeshes=denoise_micro)
        for d in denoise_chips:
            ttnn.synchronize_device(denoise_micro[d])
        host_ms = (time.perf_counter() - t0) * 1000

        # ---- socket path (build once, time steady-state second call) ----
        mig_sock = KVMigration(denoise_submesh=denoise_micro[denoise_chips[0]])
        mig_sock.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro)
        t0 = time.perf_counter()
        used = mig_sock.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro)
        sock_ms = (time.perf_counter() - t0) * 1000

        # ---- correctness: socket landings must match the golden inputs ----
        max_err = 0.0
        for i in range(VLM_DEPTH):
            k_dst, v_dst = mig_sock.get(i)
            k_back = ttnn.to_torch(ttnn.get_device_tensors(k_dst)[0]).to(torch.float32)
            v_back = ttnn.to_torch(ttnn.get_device_tensors(v_dst)[0]).to(torch.float32)
            k_ref = golden[i][0].to(torch.bfloat16).to(torch.float32)
            v_ref = golden[i][1].to(torch.bfloat16).to(torch.float32)
            max_err = max(max_err, (k_back - k_ref).abs().max().item(), (v_back - v_ref).abs().max().item())

        speedup = (host_ms / sock_ms) if sock_ms > 0 else float("nan")
        print(
            f"[kv-socket] op={used}  host_bounce={host_ms:.2f} ms  socket={sock_ms:.2f} ms  "
            f"speedup={speedup:.1f}x  max_abs_err={max_err:.4g}",
            flush=True,
        )
        assert max_err < 1e-2, f"socket KV transfer corrupted data (max_abs_err={max_err})"
    finally:
        # Drop socket refs + gc so the MeshSocket C++ objects release the submesh
        # cq, then close the (1,1) endpoints before the parent (2-level, clean).
        if mig_sock is not None:
            mig_sock._socket_state.clear()
            mig_sock.migrated_kv.clear()
        gc.collect()
        for sm in [*prefill_micro, *[m for m in denoise_micro if m is not None]]:
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        try:
            ttnn.close_mesh_device(parent)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
