# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inter-stage activation transport for Option C.

Two transport paths today:

1. `send_activation_via_host` — bounces through host DRAM. Works between any
   two submeshes (even ones not sharing a parent). Default path; ~100-200 ms
   per inference of total host-bounce overhead at full depth.

2. `send_shard_via_p2p` — direct device-to-device transfer via the ethernet
   fabric using `ttnn.point_to_point`. Requires:
     - tensor ALLOCATED on the parent mesh (with shards positioned per-coord)
     - sender + receiver coords on same row OR same column of the parent
     - fabric initialized at mesh open (`set_fabric_config(FABRIC_1D)`)
   Verified working on the BH Galaxy parent via
   `tests/test_p2p_smoke.py::test_p2p_basic_transfers`. Microsecond-class
   transfer time (no host involvement).

Inter-stage payload sizes (per deployment plan §3.2):

    HOST → vision chip 0    [2, 3, 224, 224] bf16 + lang [200] u32   ~600 KB / inference
    vision chip i → i+1     [2, 256, 1152] bf8                       ~590 KB / inference
    vision-embed → prefill  [2, 256, 2048] bf8 + lang_embed          ~1.4 MB / inference
    prefill chip i → i+1    [968, 2048] bf8                          ~2 MB   / inference
    prefill chip i → denoise (KV migration, see kv_migration.py)     ~500 KB / layer (one-shot)
    denoise chip i → i+1    [50, 1024] bf8                           ~50 KB  / Euler step (10× / inf)

Integration status:
- D2D primitive verified (test_p2p_smoke.py).
- Caller-side integration not yet wired — the layer-paired forward
  (`vlm_slice.py::Pi0_5OptionCVLMSlicePaired.forward`) allocates activations
  on 1-chip micro-submeshes, which doesn't fit the P2P "tensor on parent
  mesh, P2P between coords" model. Migrating to parent-mesh activations is
  the next step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import ttnn


def send_activation_via_host(src_tensor: "ttnn.Tensor", dst_submesh) -> "ttnn.Tensor":
    """Move a REPLICATED activation from one submesh to another via host DRAM.

    For Option C the activation is typically replicated within the source
    submesh's chip-group (or held only on the boundary chip). Sharded
    activations need a concat-via-mapper before being shipped — that's a
    separate helper we add when stage-internal sharding lands.

    Args:
        src_tensor: a replicated tensor allocated on the source submesh.
        dst_submesh: the destination submesh's MeshDevice.

    Returns:
        a tensor of the same shape and dtype, placed on dst_submesh
        (replicated, L1_MEMORY_CONFIG).
    """
    shards = ttnn.get_device_tensors(src_tensor)
    host_torch = ttnn.to_torch(shards[0])
    mapper = ttnn.replicate_tensor_to_mesh_mapper(dst_submesh)
    return ttnn.from_torch(
        host_torch,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        mesh_mapper=mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------- #
# D2D transport (ttnn.point_to_point via ethernet fabric)                  #
# ---------------------------------------------------------------------- #


def send_shard_via_p2p(
    tensor: "ttnn.Tensor",
    sender_coord: Tuple[int, int],
    receiver_coord: Tuple[int, int],
    output_tensor: Optional["ttnn.Tensor"] = None,
) -> "ttnn.Tensor":
    """Transfer one shard of a parent-mesh tensor from `sender_coord` to
    `receiver_coord` via the ethernet fabric (no host involvement).

    Constraints (verified via tests/test_p2p_smoke.py):
      - `tensor` must already live on the PARENT MeshDevice (full-mesh
        allocation, e.g. via ShardTensorToMesh or replicate_tensor_to_mesh_mapper).
      - `sender_coord` and `receiver_coord` must be on the same row OR same
        column of the parent mesh — 1D linear topology routing.
      - Parent must have been opened with
        `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` BEFORE
        `ttnn.open_mesh_device` (see `mesh_setup.open_galaxy_mesh(enable_fabric=True)`).
      - Interleaved memory only (sharded p2p not yet supported in ttnn —
        see deepseek_v3/utils/composite_ops.py:25 TODO).

    Semantics: returns a tensor whose shard at `receiver_coord` holds the
    bytes that were at `sender_coord` in `tensor`. The sender shard is NOT
    preserved in the returned tensor — this is a one-shot copy, not a
    broadcast.

    For the layer-paired prefill chain (activation flows layer i → i+1):
        out = send_shard_via_p2p(
            tensor=activation_on_parent,
            sender_coord=(row_i, col_i),
            receiver_coord=(row_{i+1}, col_{i+1}),
        )
    Then layer i+1's matmul reads the shard at (row_{i+1}, col_{i+1}).
    """
    src = ttnn.MeshCoordinate(*sender_coord)
    dst = ttnn.MeshCoordinate(*receiver_coord)
    kwargs = {
        "topology": ttnn.Topology.Linear,
    }
    if output_tensor is not None:
        kwargs["output_tensor"] = output_tensor
    return ttnn.point_to_point(tensor, src, dst, **kwargs)


def send_shard_via_p2p_multihop(
    tensor: "ttnn.Tensor",
    sender_coord: Tuple[int, int],
    receiver_coord: Tuple[int, int],
) -> "ttnn.Tensor":
    """Multi-hop variant of `send_shard_via_p2p` for arbitrary coord pairs.

    When sender and receiver share a row OR column, this is a single P2P
    call (same as `send_shard_via_p2p`). When they differ on both row and
    column (e.g. prefill row-boundary transitions (2, 2) → (3, 0)), routes
    via an intermediate chip on a shared row/column: A → (A_row, B_col) → B.
    That's two P2P calls = two fabric hops, still microseconds-class.

    Discovered constraint (per tests/test_parent_mesh_chain_smoke.py):
    `ttnn.Topology.Linear` only routes within a single mesh row or column.
    Diagonal transitions need this 2-hop pattern.
    """
    src_r, src_c = sender_coord
    dst_r, dst_c = receiver_coord
    if src_r == dst_r or src_c == dst_c:
        # Same row or same column → single hop.
        return send_shard_via_p2p(tensor, sender_coord, receiver_coord)
    # Route via (src_r, dst_c): same row as src, same column as dst.
    intermediate = (src_r, dst_c)
    hop1 = send_shard_via_p2p(tensor, sender_coord, intermediate)
    hop2 = send_shard_via_p2p(hop1, intermediate, receiver_coord)
    if hop1 is not hop2:
        ttnn.deallocate(hop1)
    return hop2


# ---------------------------------------------------------------------- #
# D2D transport via MeshSocket (inter-submesh)                            #
# ---------------------------------------------------------------------- #
#
# Sockets move a tensor between two SEPARATE submeshes of the same parent
# MeshDevice (cf. point_to_point, which moves a shard *within* one
# parent-mesh tensor). In Option C this is the inter-stage hand-off:
# prefill micro-submesh (1,1) -> denoise micro-submesh (1,1).
#
# Requires fabric enabled at parent-mesh open (open_galaxy_mesh(enable_fabric=True)).
# The send/recv ops are injected so the same plumbing serves both the FIFO
# path (ttnn.experimental.send_async/recv_async) and the direct-write path
# (ttnn.experimental.send_direct_async/recv_direct_async). Both share the same
# signature: send_op(src, send_socket); recv_op(pre_allocated_dst, recv_socket).


def build_socket_pair(
    sender_submesh, recv_submesh, num_connections: int = 1, page_size_bytes: int = 8192, core_base: int = 0
):
    """Create one MeshSocket pair between two equal-shape submeshes.

    Sender cores live in row 0, receiver cores in row 1 so the two sets never
    overlap (the socket runtime forbids a core appearing in two connections of
    one socket). One connection per mesh coord; for the (1,1) micro-submeshes
    used by Option C's KV migration that is a single sender/receiver core pair.

    The FIFO holds payload (FIFO path) or just the handshake/completion token
    (direct path); page_size_bytes*4 is a safe L1 FIFO size for both.
    """
    sender_cores = [ttnn.CoreCoord(core_base + i, 0) for i in range(num_connections)]
    recv_cores = [ttnn.CoreCoord(core_base + i, 1) for i in range(num_connections)]
    connections = []
    for coord in ttnn.MeshCoordinateRange(sender_submesh.shape):
        for s, r in zip(sender_cores, recv_cores):
            connections.append(ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, s), ttnn.MeshCoreCoord(coord, r)))
    socket_mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, page_size_bytes * 4)
    socket_config = ttnn.SocketConfig(connections, socket_mem)
    return ttnn.create_socket_pair(sender_submesh, recv_submesh, socket_config)


def transfer_via_socket(
    src_tensor: "ttnn.Tensor",
    dst_landing: "ttnn.Tensor",
    send_socket,
    recv_socket,
    sender_submesh,
    recv_submesh,
    send_op,
    recv_op,
) -> "ttnn.Tensor":
    """Stream one tensor src->dst over a pre-built socket pair, then sync both
    submeshes so the payload is guaranteed present. `dst_landing` must be
    pre-allocated on recv_submesh with a spec matching src_tensor.

    `send_op`/`recv_op` select FIFO (send_async/recv_async) vs direct-write
    (send_direct_async/recv_direct_async) — both async, same signature.
    """
    send_op(src_tensor, send_socket)
    recv_op(dst_landing, recv_socket)
    ttnn.synchronize_device(sender_submesh)
    ttnn.synchronize_device(recv_submesh)
    return dst_landing


def resolve_socket_ops(prefer: str = "fifo"):
    """Pick (send_op, recv_op). `prefer` ('fifo'|'direct') is overridable by the
    PI0_OC_SOCKET_OP env var. Falls back to FIFO if the direct ops are not built
    into this ttnn (the FIFO branch). Returns (send_op, recv_op, name)."""
    import os

    choice = os.environ.get("PI0_OC_SOCKET_OP", prefer).lower()
    if choice == "direct" and hasattr(ttnn.experimental, "send_direct_async"):
        return ttnn.experimental.send_direct_async, ttnn.experimental.recv_direct_async, "direct"
    return ttnn.experimental.send_async, ttnn.experimental.recv_async, "fifo"


def send_per_chip_activation_via_host(src_tensor: "ttnn.Tensor", src_chip_idx: int, dst_submesh) -> "ttnn.Tensor":
    """Move a single chip's shard from a source submesh to a (replicated)
    placement on a destination submesh, via host DRAM.

    Useful for the prefill-pipeline → denoise hand-off, where prefill chip 17
    (last VLM layer) is the only chip holding the final activation. We pluck
    its shard, send to host, then replicate onto every denoise chip.
    """
    shards = ttnn.get_device_tensors(src_tensor)
    if src_chip_idx >= len(shards):
        raise IndexError(f"src_chip_idx={src_chip_idx} out of range (source submesh has {len(shards)} chips)")
    host_torch = ttnn.to_torch(shards[src_chip_idx])
    mapper = ttnn.replicate_tensor_to_mesh_mapper(dst_submesh)
    return ttnn.from_torch(
        host_torch,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        mesh_mapper=mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
