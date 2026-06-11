# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: Option C KV-cache MOVEMENT, prefill chips -> denoise chips.

This measures only the inter-submesh *movement* of the VLM KV cache (the
prefill->denoise hand-off), NOT the prefill compute. The KV tensors are placed
directly on the real Option C chips at the real config shape
([1, num_kv_heads, S_prefix, head_dim] per layer, 18 layers); movement time
depends on shape/bytes + topology, not on the KV values, so synthetic-but-
real-shape KV gives the same number as genuine prefill output.

Four transports compared (averaged over NUM_ITERS, warmup excluded):
  1. host-bounce  — to_torch/from_torch through host DRAM (migrate_layer_paired)
  2. FIFO socket  — send_async/recv_async                  (migrate_layer_paired_socket)
  3. direct socket— send_direct_async/recv_direct_async    (migrate_layer_paired_socket)
  4. point_to_point — fabric P2P on parent mesh            (migrate_layer_paired_d2d)

Transports 1-3 consume KV on per-layer (1,1) micro-submeshes; (4) consumes the
SAME-shape KV replicated on the parent mesh (point_to_point reads the src-coord
shard). The KV hop prefill (2+i//3, i%3) -> denoise (2+i//3, 3) is non-adjacent,
so a socket needs FABRIC_2D. The 'direct' row only runs where the direct ops are
built (the sdawle_blaze_socket_direct branch); elsewhere it is skipped.

Exclusive 32-chip cluster + fabric. Gated by PI0_OC_KV_BENCH=1.

Run (ask before running — blocks other cluster users):
    PI0_OC_KV_BENCH=1 python -m pytest \
        models/experimental/pi0_5/tests/test_oc_kv_movement_bench.py -s
"""

from __future__ import annotations

import gc
import os
import statistics
import time

import pytest
import torch

import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration
from models.experimental.pi0_5.tt.option_c.stages import DENOISE_SUBMESH_OFFSET

BENCH_ENABLED = os.environ.get("PI0_OC_KV_BENCH") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_OC_KV_BENCH=1 to run the KV-movement benchmark")

PARENT_SHAPE = (8, 4)
VLM_DEPTH = int(os.environ.get("PI0_OC_VLM_DEPTH", "18"))  # full VLM depth
S_PREFIX = int(os.environ.get("PI0_OC_S_PREFIX", "768"))  # prefix seq len (tile-aligned: multiple of 32)
NUM_ITERS = int(os.environ.get("PI0_OC_BENCH_ITERS", "20"))
WARMUP = int(os.environ.get("PI0_OC_BENCH_WARMUP", "3"))

_CFG = PaliGemmaConfig()  # vlm: num_kv_heads=1, head_dim=256
NUM_KV_HEADS = _CFG.vlm_config.num_kv_heads
HEAD_DIM = _CFG.vlm_config.head_dim
KV_SHAPE = [1, NUM_KV_HEADS, S_PREFIX, HEAD_DIM]  # real per-layer K/V shape


def _kv_torch(seed: int):
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(KV_SHAPE, generator=gen, dtype=torch.float32),
        torch.randn(KV_SHAPE, generator=gen, dtype=torch.float32),
    )


def _up(t, mesh):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )


def _free_migrated(m: KVMigration):
    """Deallocate a migrator's output KV and clear (host-bounce / p2p paths
    allocate fresh tensors each call)."""
    for kv in m.migrated_kv.values():
        for t in kv:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
    m.migrated_kv.clear()


def _bench(migrate_once, cleanup):
    for _ in range(WARMUP):
        migrate_once()
        cleanup()
    times_ms = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        migrate_once()
        times_ms.append((time.perf_counter() - t0) * 1000)
        cleanup()
    return times_ms


def _stats(times_ms):
    return (
        statistics.mean(times_ms),
        statistics.median(times_ms),
        statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    )


def _max_err(migrated_list, src_torch):
    err = 0.0
    for i, kv in enumerate(migrated_list):
        if kv is None:
            continue
        k_back = ttnn.to_torch(ttnn.get_device_tensors(kv[0])[0]).to(torch.float32)
        v_back = ttnn.to_torch(ttnn.get_device_tensors(kv[1])[0]).to(torch.float32)
        k_ref = src_torch[i][0].to(torch.bfloat16).to(torch.float32)
        v_ref = src_torch[i][1].to(torch.bfloat16).to(torch.float32)
        err = max(err, (k_back - k_ref).abs().max().item(), (v_back - v_ref).abs().max().item())
    return err


@pytest.mark.timeout(1800)
def test_kv_movement_all_transports():
    """Move VLM_DEPTH layers of KV prefill->denoise via host-bounce, FIFO socket,
    direct socket, and point_to_point; report averaged movement time + speedups."""
    total_bytes = VLM_DEPTH * 2 * (NUM_KV_HEADS * S_PREFIX * HEAD_DIM) * 2  # K+V, bf16
    print(
        f"\n[kv-move] depth={VLM_DEPTH} kv_shape={KV_SHAPE} per-chip-pair "
        f"total={total_bytes/1e6:.1f} MB iters={NUM_ITERS} (warmup {WARMUP})",
        flush=True,
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*PARENT_SHAPE))

    prefill_micro = []
    denoise_chips = sorted({KVMigration.denoise_chip_for_vlm_layer(i) for i in range(VLM_DEPTH)})
    denoise_micro = [None] * (max(denoise_chips) + 1)
    src_torch = [_kv_torch(100 + i) for i in range(VLM_DEPTH)]
    rows = []
    migrators = []
    parent_kv = []
    try:
        # (1,1) endpoints at the real Option C parent coords.
        for i in range(VLM_DEPTH):
            r, c = KVMigration._prefill_parent_coord(i)
            prefill_micro.append(parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c)))
        for d in denoise_chips:
            r, c = DENOISE_SUBMESH_OFFSET[0] + d, DENOISE_SUBMESH_OFFSET[1]
            denoise_micro[d] = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))

        # Real-shape KV on the per-layer prefill micro-submeshes (host/fifo/direct).
        per_layer_kv = [(_up(k, prefill_micro[i]), _up(v, prefill_micro[i])) for i, (k, v) in enumerate(src_torch)]
        # Same KV replicated on the parent mesh (point_to_point reads the src-coord shard).
        parent_kv = [(_up(k, parent), _up(v, parent)) for (k, v) in src_torch]

        denoise0 = denoise_micro[denoise_chips[0]]

        # ---- 1. host-bounce ----
        m_host = KVMigration(denoise_submesh=denoise0)
        migrators.append(m_host)

        def _host():
            m_host.migrate_layer_paired(per_layer_kv, denoise_micro_submeshes=denoise_micro)
            for d in denoise_chips:
                ttnn.synchronize_device(denoise_micro[d])

        t = _bench(_host, lambda: _free_migrated(m_host))
        # one more for correctness snapshot
        m_host.migrate_layer_paired(per_layer_kv, denoise_micro_submeshes=denoise_micro)
        err = _max_err(m_host.as_list(VLM_DEPTH), src_torch)
        _free_migrated(m_host)
        rows.append(("host_bounce", *_stats(t), err))

        # ---- 2. FIFO socket ----
        fifo_ops = (ttnn.experimental.send_async, ttnn.experimental.recv_async, "fifo")
        m_fifo = KVMigration(denoise_submesh=denoise0)
        migrators.append(m_fifo)

        def _fifo():
            m_fifo.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro, socket_ops=fifo_ops)

        t = _bench(_fifo, lambda: m_fifo.migrated_kv.clear())  # sockets+buffers reused; don't free them
        m_fifo.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro, socket_ops=fifo_ops)
        err = _max_err(m_fifo.as_list(VLM_DEPTH), src_torch)
        rows.append(("socket_fifo", *_stats(t), err))

        # ---- 3. direct socket (only where the direct ops are built) ----
        if hasattr(ttnn.experimental, "send_direct_async"):
            direct_ops = (ttnn.experimental.send_direct_async, ttnn.experimental.recv_direct_async, "direct")
            m_dir = KVMigration(denoise_submesh=denoise0)
            migrators.append(m_dir)

            def _direct():
                m_dir.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro, socket_ops=direct_ops)

            t = _bench(_direct, lambda: m_dir.migrated_kv.clear())
            m_dir.migrate_layer_paired_socket(per_layer_kv, prefill_micro, denoise_micro, socket_ops=direct_ops)
            err = _max_err(m_dir.as_list(VLM_DEPTH), src_torch)
            rows.append(("socket_direct", *_stats(t), err))
        else:
            print("[kv-move] direct ops not built on this branch — skipping direct row", flush=True)

        # ---- 4. point_to_point (parent-mesh KV) ----
        m_p2p = KVMigration(denoise_submesh=denoise0)
        migrators.append(m_p2p)

        def _p2p():
            m_p2p.migrate_layer_paired_d2d(parent_kv)
            for d in denoise_chips:
                ttnn.synchronize_device(denoise_micro[d])

        t = _bench(_p2p, lambda: _free_migrated(m_p2p))
        # p2p lands on a parent-mesh tensor at the denoise coord (not a (1,1)
        # submesh), and our parent KV is replicated, so a bit-exact check here
        # would be vacuous. Time only; correctness of point_to_point is covered
        # by tests/test_p2p_smoke.py. err sentinel -1 => "not checked".
        _free_migrated(m_p2p)
        rows.append(("point_to_point", *_stats(t), -1.0))

        # ---- report ----
        host_mean = next(r[1] for r in rows if r[0] == "host_bounce")
        print(f"\n[kv-move] KV movement, {VLM_DEPTH} layers, {NUM_ITERS} iters (mean/median/stdev ms):", flush=True)
        print(f"  {'transport':<16}{'mean':>9}{'median':>9}{'stdev':>8}{'speedup':>9}{'max_err':>10}", flush=True)
        for name, mean, med, sd, err in rows:
            err_str = "n/a" if err < 0 else f"{err:.4g}"
            print(f"  {name:<16}{mean:>9.3f}{med:>9.3f}{sd:>8.3f}{host_mean/mean:>8.1f}x{err_str:>10}", flush=True)

        for name, mean, med, sd, err in rows:
            if err >= 0:  # -1 sentinel = correctness not checked here (p2p)
                assert err < 1e-2, f"{name} corrupted KV (max_abs_err={err})"
    finally:
        for m in migrators:
            try:
                m._socket_state.clear()
                m.migrated_kv.clear()
            except Exception:
                pass
        for kv in parent_kv:
            for tt in kv:
                try:
                    ttnn.deallocate(tt)
                except Exception:
                    pass
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
