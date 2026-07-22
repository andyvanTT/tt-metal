# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""T2 — bridge + inject round-trip.

Build a dummy non-paged KV cache tensor matching the real spec, run the on-device
bridge (tilize) + inject (typecast→bfp8, fill_cache), read it back, and check the
injected region matches the source and the untouched region stays zero.
"""

import torch

import ttnn

import bridge
import inject


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_bridge_inject_roundtrip(mesh):
    B, NKV, MAXSEQ, HD = 1, 2, 128, 64
    prompt_len = 4
    mapper = ttnn.ReplicateTensorToMesh(mesh)

    def _cache():
        return ttnn.as_tensor(
            torch.zeros(B, NKV, MAXSEQ, HD),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    cache_k, cache_v = _cache(), _cache()

    # prefill emits [B, seq, n_kv_heads, head_dim] (heads/seq swapped vs cache) — bridge fixes it.
    k_host = torch.randn(B, prompt_len, NKV, HD)
    v_host = torch.randn(B, prompt_len, NKV, HD)

    kv_dev = bridge.to_device_kv([(k_host, v_host)], mesh)
    assert list(kv_dev[0][0].shape) == [B, NKV, 32, HD]  # heads/seq swapped, seq padded to tile

    inject.inject_into_caches([(cache_k, cache_v)], kv_dev, batch_idx=0)

    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
    got_k = ttnn.to_torch(cache_k, mesh_composer=composer)[0:1]  # [B, NKV, MAXSEQ, HD]
    got_v = ttnn.to_torch(cache_v, mesh_composer=composer)[0:1]

    # Compare against the cache-order source (permute host to [B, NKV, seq, HD]).
    k_ref = k_host.permute(0, 2, 1, 3)
    v_ref = v_host.permute(0, 2, 1, 3)

    assert _pcc(got_k[:, :, :prompt_len, :], k_ref) > 0.99
    assert _pcc(got_v[:, :, :prompt_len, :], v_ref) > 0.99
    # Padding (prompt_len..32) and beyond the fill (32..) must stay zero.
    assert got_k[:, :, prompt_len:32, :].abs().max().item() == 0.0
    assert got_k[:, :, 32:, :].abs().max().item() == 0.0
