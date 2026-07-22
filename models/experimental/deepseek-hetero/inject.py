# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Inject: write the bridged per-layer KV into the TT decode KV cache.

Mirrors the native prefill fill path in ``tt_transformers/tt/attention.py``:
    k_fill = ttnn.typecast(k_heads, dtype=keys_BKSD.dtype)   # bf16 -> cache dtype (bfp8)
    ttnn.fill_cache(keys_BKSD, k_fill, user_id % batch_size_per_device_group)

The typecast to the cache dtype is the **on-device bfp8 block-float quant**. We inject
into each layer's persistent DRAM cache (``layer.attention.layer_past``) BEFORE the first
traced decode step, so the decode trace reads the injected values from the same addresses.
"""

from __future__ import annotations

from typing import List, Tuple

import ttnn


def caches_from_model(model) -> List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]:
    """Per-layer (cache_k, cache_v) tensors from a (non-paged) TT model."""
    return [(l.attention.layer_past[0], l.attention.layer_past[1]) for l in model.layers]


def inject_into_caches(
    cache_pairs: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]],
    kv_dev: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]],
    batch_idx: int = 0,
) -> None:
    """Typecast each bridged (k_dev, v_dev) to its cache dtype and ``fill_cache`` it.

    cache_pairs: per-layer (cache_k, cache_v) — the persistent DRAM KV cache tensors.
    kv_dev:      per-layer (k_dev, v_dev) — bf16 TILE tensors from ``bridge.to_device_kv``.
    batch_idx:   which user slot to fill (0 for the single-user demo).
    """
    if len(cache_pairs) != len(kv_dev):
        raise ValueError(f"layer count mismatch: {len(cache_pairs)} caches vs {len(kv_dev)} kv")

    for (cache_k, cache_v), (k_dev, v_dev) in zip(cache_pairs, kv_dev):
        k_fill = ttnn.typecast(k_dev, dtype=cache_k.dtype)  # on-device quant to cache dtype
        v_fill = ttnn.typecast(v_dev, dtype=cache_v.dtype)
        ttnn.fill_cache(cache_k, k_fill, batch_idx)
        ttnn.fill_cache(cache_v, v_fill, batch_idx)
        ttnn.deallocate(k_fill)
        ttnn.deallocate(v_fill)


def inject_kv(model, kv_dev, batch_idx: int = 0) -> None:
    """Convenience: inject bridged KV into a TT model's per-layer caches."""
    inject_into_caches(caches_from_model(model), kv_dev, batch_idx)
