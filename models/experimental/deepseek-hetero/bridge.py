# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bridge: host KV tensors → device ttnn tensors in the cache's tile layout.

Design choice (locked): the transport carries **plain bf16, row-major**; the TILE
re-layout happens **on-device** (``ttnn.tilize``), and the bfp8 block-float quant happens
on-device too (in ``inject.py`` via ``ttnn.typecast`` to the cache dtype). This keeps a
future BlueField DPU out of the block-float business — it only moves bf16 bytes.

Per layer this produces a bf16 TILE tensor ``[1, n_kv_heads, seq_padded, head_dim]``,
replicated across the mesh, ready for ``inject.py`` to typecast + ``fill_cache``.

Sequence length is padded up to a 32-tile multiple (TILE layout requires it). The padded
positions hold zeros; they sit beyond ``prompt_len`` in the cache and are overwritten by
decode before they are ever attended to (decode writes position ``current_pos`` each step,
starting at ``prompt_len``).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

import ttnn

TILE = 32


def _pad_seq_to_tile(t: torch.Tensor) -> torch.Tensor:
    """Pad ``[1, n_kv, seq, head_dim]`` along seq up to a multiple of 32 with zeros."""
    seq = t.shape[2]
    padded = ((seq + TILE - 1) // TILE) * TILE
    if padded == seq:
        return t
    pad = torch.zeros(t.shape[0], t.shape[1], padded - seq, t.shape[3], dtype=t.dtype)
    return torch.cat([t, pad], dim=2)


def to_device_kv(
    kv_host: List[Tuple[torch.Tensor, torch.Tensor]],
    mesh_device,
) -> List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]:
    """Move per-layer host (K, V) to device as bf16 TILE tensors (on-device tilize).

    kv_host: list over layers of (K, V), each ``[1, n_kv_heads, seq, head_dim]`` torch,
             with K already in Meta post-RoPE order (from ``prefill_hf``).
    Returns: list over layers of (k_dev, v_dev) — bf16, TILE layout, on ``mesh_device``.
    """
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    out = []
    for k, v in kv_host:
        # prefill emits [B, seq, n_kv_heads, head_dim] (meta-reference order); the TT KV
        # cache and ttnn.fill_cache expect [B, n_kv_heads, seq, head_dim]. Swap heads<->seq.
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        k = _pad_seq_to_tile(k.to(torch.bfloat16))
        v = _pad_seq_to_tile(v.to(torch.bfloat16))
        # bf16 row-major to device == the "plain bf16 on the wire" DMA step.
        k_rm = ttnn.from_torch(
            k, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        v_rm = ttnn.from_torch(
            v, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        # On-device tilize (row-major -> TILE). bfp8 quant is deferred to inject.py.
        k_dev = ttnn.tilize(k_rm)
        v_dev = ttnn.tilize(v_rm)
        ttnn.deallocate(k_rm)
        ttnn.deallocate(v_rm)
        out.append((k_dev, v_dev))
    return out
