# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end performance — pipeline-split variant with a socket d2d handoff.

This mirrors ``test_perf_ttnn_full_e2e.py`` but splits the model across two
single-chip submeshes and routes the prefix→suffix handoff through the
direct-write socket ops (``ttnn.experimental.send_direct_async`` /
``recv_direct_async``, added in commit 7261d72c81):

      submesh A (PREFIX)                         submesh B (SUFFIX)
  SigLIP + Gemma-2B VLM prefill   ──socket──►   10-step denoise loop
      → prefix K/V cache          (direct d2d)   over the received K/V cache

Both stages reuse the *same* ``Pi0_5ModelTTNN`` class (one instance per
submesh) via the ``run_prefix`` / ``run_denoise`` seam. The real prefix K/V
cache (one (K,V) pair per VLM layer) plus the upstream RoPE/mask tensors the
denoise loop consumes are transferred A→B each chunk; the loop then runs
entirely on submesh B.

Run under tracy to get a per-op CSV in which send/recv_direct_async appear
alongside the model ops:

  PI0_UPSTREAM_MASKS=1 QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \\
  PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \\
  python -m tracy -p -r -v --op-support-count 100000 \\
    -o generated/pi05_socket -n pi0.5_socket \\
    -m "pytest models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_socket.py"

Skipped if the checkpoint isn't present locally.
"""

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = 0
NUM_ITERS = 1
LANG_SEQ_LEN = 256
SEED = 0
TRACE_REGION_SIZE = 80_000_000

# Socket transfer knobs. NUM_CONNECTIONS == parallel sender/receiver core pairs
# per chip (more == more parallel ethernet channels). In direct mode the FIFO
# only carries the handshake + completion token, so the page size is small.
NUM_CONNECTIONS = 2
SOCKET_PAGE_SIZE = 8192

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device, batch_size: int = 1):
    torch.manual_seed(SEED)
    image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 256000, (batch_size, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(batch_size, LANG_SEQ_LEN, dtype=torch.bool)

    image_ttnn = ttnn.from_torch(
        image,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn


def _build_socket_connections(mesh_shape: "ttnn.MeshShape", num_connections: int):
    """Sender cores in row 0, receiver cores in row 1 (the two sets never overlap —
    the socket runtime forbids a core appearing in two connections of one socket)."""
    sender_cores = [ttnn.CoreCoord(i, 0) for i in range(num_connections)]
    recv_cores = [ttnn.CoreCoord(i, 1) for i in range(num_connections)]
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for sender, receiver in zip(sender_cores, recv_cores):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, sender),
                    ttnn.MeshCoreCoord(coord, receiver),
                )
            )
    return connections


def _flatten_kv(kv_cache):
    """[(k0,v0),(k1,v1),...] -> [k0,v0,k1,v1,...] (transfer order)."""
    flat = []
    for k, v in kv_cache:
        flat.append(k)
        flat.append(v)
    return flat


def _unflatten_kv(flat):
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _transfer(src_tensors, dst_tensors, send_socket, recv_socket, sender_mesh, recv_mesh):
    """Stream each source tensor directly into its pre-allocated landing buffer on
    the receiver submesh, then sync both submeshes so the data is present."""
    for src, dst in zip(src_tensors, dst_tensors):
        ttnn.experimental.send_direct_async(src, send_socket)
        ttnn.experimental.recv_direct_async(dst, recv_socket)
    ttnn.synchronize_device(sender_mesh)
    ttnn.synchronize_device(recv_mesh)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "l1_small_size": 24576,
            "trace_region_size": TRACE_REGION_SIZE,
        }
    ],
    indirect=True,
)
def test_pi0_5_ttnn_full_e2e_socket_fps(mesh_device):
    """End-to-end `sample_actions` split across two submeshes with a socket KV handoff."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    # Two single-chip submeshes: A = prefix producer, B = suffix consumer.
    submesh_a = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    submesh_b = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    num_denoising_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
    print(
        f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}  "
        f"(action_horizon={action_horizon}, num_denoising_steps={num_denoising_steps})"
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=num_denoising_steps)
    # One full model instance per submesh — A runs the prefix path, B the denoise loop.
    model_a = Pi0_5ModelTTNN(cfg, loader, submesh_a)
    model_b = Pi0_5ModelTTNN(cfg, loader, submesh_b)
    print("✅ Models loaded on both submeshes")

    image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(submesh_a)

    # --- one-time socket setup ---
    connections = _build_socket_connections(submesh_a.shape, NUM_CONNECTIONS)
    socket_mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCKET_PAGE_SIZE * 4)
    socket_config = ttnn.SocketConfig(connections, socket_mem)
    send_socket, recv_socket = ttnn.create_socket_pair(submesh_a, submesh_b, socket_config)

    def _run_prefix():
        return model_a.run_prefix([image_ttnn], [img_mask], lang_tokens_ttnn, lang_masks_ttnn)

    # First prefix pass establishes the tensor specs so we can pre-allocate the
    # receiver-side landing buffers once and reuse them every chunk.
    prefix_kv_a, artifacts_a, batch_size, _keepalive = _run_prefix()
    assert artifacts_a is not None, "expected PI0_UPSTREAM_MASKS=1 — denoise needs the upstream artifacts"

    # Tensors that cross A→B: the per-layer K/V cache + the three upstream
    # tensors the denoise loop reads (suffix RoPE tables + expert attn mask).
    artifact_keys = ["suffix_cos", "suffix_sin", "expert_attn_mask"]

    def _src_list(prefix_kv, artifacts):
        return _flatten_kv(prefix_kv) + [artifacts[k] for k in artifact_keys]

    src_tensors = _src_list(prefix_kv_a, artifacts_a)
    dst_tensors = [ttnn.allocate_tensor_on_device(t.spec, submesh_b) for t in src_tensors]
    num_kv = len(prefix_kv_a)
    print(f"   transferring {len(src_tensors)} tensors/chunk ({num_kv} K/V pairs + {len(artifact_keys)} artifacts)")

    def _b_artifacts(dst):
        flat_kv = dst[: 2 * num_kv]
        arts = dst[2 * num_kv :]
        kv_b = _unflatten_kv(flat_kv)
        artifacts_b = {k: arts[i] for i, k in enumerate(artifact_keys)}
        return kv_b, artifacts_b

    def _one_chunk(prefix_kv, artifacts):
        # Transfer this chunk's prefix outputs A→B, then denoise on B.
        _transfer(_src_list(prefix_kv, artifacts), dst_tensors, send_socket, recv_socket, submesh_a, submesh_b)
        kv_b, artifacts_b = _b_artifacts(dst_tensors)
        return model_b.run_denoise(kv_b, artifacts_b, batch_size, state=None)

    # Correctness: confirm the received K/V matches the source before timing.
    out = _one_chunk(prefix_kv_a, artifacts_a)
    src0 = ttnn.to_torch(src_tensors[0], mesh_composer=ttnn.ConcatMeshToTensor(submesh_a, dim=0))
    dst0 = ttnn.to_torch(dst_tensors[0], mesh_composer=ttnn.ConcatMeshToTensor(submesh_b, dim=0))
    assert torch.allclose(src0, dst0), "socket transfer corrupted the first K/V tensor"
    actions = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh_b, dim=0))
    actions = actions[:, : cfg.action_horizon, : cfg.action_dim]
    assert actions.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all(), "actions contain NaN/Inf"
    print(f"   ✅ socket transfer verified, output shape {tuple(actions.shape)}, all finite")

    print(f"\n⏱️  Measuring steady-state ({NUM_ITERS} split sample_actions calls)")
    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        with torch.no_grad():
            prefix_kv_a, artifacts_a, batch_size, _keepalive = _run_prefix()
            _ = _one_chunk(prefix_kv_a, artifacts_a)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   call {i + 1:2d}: {elapsed_ms:7.2f} ms")

    avg = statistics.mean(times_ms)
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon

    print("\n" + "=" * 72)
    print("  PI0.5 TTNN SOCKET-SPLIT END-TO-END PERFORMANCE")
    print("=" * 72)
    print(f"   Steady-state avg:    {avg:7.2f} ms")
    print(f"   Steady-state min:    {min(times_ms):7.2f} ms")
    print(f"   Steady-state max:    {max(times_ms):7.2f} ms")
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s  ({cfg.action_horizon}/chunk)")
    print("=" * 72)
    assert avg > 0
