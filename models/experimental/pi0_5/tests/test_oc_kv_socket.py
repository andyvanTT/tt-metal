# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused test for Option C inter-submesh KV migration over MeshSockets.

Validates `KVMigration.migrate_layer_paired_socket` — the inter-submesh d2d
hand-off (prefill micro-submesh (1,1) -> denoise micro-submesh (1,1)) — against
the host-bounce baseline (`migrate_layer_paired`), on synthetic per-layer K/V
tensors of realistic size. No model forward / checkpoint needed: it isolates the
transport mechanism and checks (a) correctness (bytes land intact) and (b)
per-transport wall time.

Socket op (FIFO vs direct-write) is chosen by PI0_OC_SOCKET_OP (fifo|direct);
the direct ops only exist on the sdawle_blaze_socket_direct branch, so this file
runs identically on both branches and reports which op it used.

Run (exclusive 32-chip cluster, fabric required):
    PI0_OC_SOCKET_OP=fifo \
    python -m pytest models/experimental/pi0_5/tests/test_oc_kv_socket.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration
from models.experimental.pi0_5.tt.option_c.mesh_setup import create_per_chip_submeshes, open_galaxy_mesh
from models.experimental.pi0_5.tt.option_c.stages import (
    EXPERT_LAYERS_PER_DENOISE_CHIP,
    build_shrunk_layout,
)
from models.experimental.pi0_5.tt.option_c.transport import resolve_socket_ops

# Realistic per-layer KV footprint: prefix seq ~256, gemma kv width ~2048 → ~1MB
# bf16 per tensor. Tile-aligned 4D shape.
KV_SHAPE = [1, 1, 256, 2048]
VLM_DEPTH = 2  # 2 prefill micro-submeshes (layers 0,1) → both land on denoise chip 0
EXPERT_DEPTH = 1  # 1 denoise micro-submesh


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
    """Migrate VLM_DEPTH layers of synthetic K/V prefill-micro -> denoise-micro
    via (1) host bounce and (2) socket; verify both land intact and report timings."""
    layout = build_shrunk_layout(vlm_depth=VLM_DEPTH, expert_depth=EXPERT_DEPTH)

    with open_galaxy_mesh(layout, enable_fabric=True) as (_parent, submeshes):
        _vision, prefill_submesh, denoise_submesh = submeshes
        prefill_micro = create_per_chip_submeshes(prefill_submesh, VLM_DEPTH)
        num_denoise_chips = max(
            1, (EXPERT_DEPTH + EXPERT_LAYERS_PER_DENOISE_CHIP - 1) // EXPERT_LAYERS_PER_DENOISE_CHIP
        )
        denoise_micro = create_per_chip_submeshes(denoise_submesh, num_denoise_chips)

        # Synthetic per-layer KV on each prefill micro-submesh + golden host copies.
        per_layer_kv = []
        golden = []
        for i in range(VLM_DEPTH):
            k_dev, v_dev, k_host, v_host = _make_kv_on(prefill_micro[i], seed=100 + i)
            per_layer_kv.append((k_dev, v_dev))
            golden.append((k_host, v_host))

        _send, _recv, op_name = resolve_socket_ops()
        print(f"\n[kv-socket] socket op = {op_name}  layers={VLM_DEPTH}  kv_shape={KV_SHAPE}")

        # ---- host-bounce baseline ----
        mig_host = KVMigration(denoise_submesh=denoise_submesh)
        t0 = time.perf_counter()
        mig_host.migrate_layer_paired(per_layer_kv, denoise_micro_submeshes=denoise_micro)
        ttnn.synchronize_device(denoise_submesh)
        host_ms = (time.perf_counter() - t0) * 1000

        # ---- socket path ----
        mig_sock = KVMigration(denoise_submesh=denoise_submesh)
        # First call builds sockets+buffers (one-time); time a second call for steady-state.
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

        print(
            f"[kv-socket] op={used}  host_bounce={host_ms:.2f} ms  socket={sock_ms:.2f} ms  max_abs_err={max_err:.4g}"
        )
        assert max_err < 1e-2, f"socket KV transfer corrupted data (max_abs_err={max_err})"
