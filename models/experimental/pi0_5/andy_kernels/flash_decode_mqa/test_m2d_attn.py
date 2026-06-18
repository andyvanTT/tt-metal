# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2d: validate single-block attention O = softmax(Q@K^T)@V (Sk=32, single core)
vs torch SDPA. Q is pre-scaled by 1/sqrt(d) on the host so the kernel softmax is plain."""

import math

import torch
import ttnn

from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        Sq, Sk, D = 32, 32, 256
        scale = 1.0 / math.sqrt(D)
        core = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
        grid = ttnn.CoreRangeSet({core})

        def shard(h, w):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
            )

        torch.manual_seed(0)
        q = torch.randn(Sq, D, dtype=torch.float32) * 0.3
        k = torch.randn(Sk, D, dtype=torch.float32) * 0.3
        v = torch.randn(Sk, D, dtype=torch.float32) * 0.3
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], scale=scale, is_causal=False
        )[
            0, 0
        ]  # (Sq, D)

        qs = q * scale  # pre-scale Q -> kernel softmax is plain
        tq = ttnn.from_torch(
            qs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D)
        )
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sk, D))
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sk, D))
        tsc = ttnn.from_torch(
            torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(32, 32)
        )
        tout = ttnn.from_torch(
            torch.zeros(Sq, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D)
        )

        FlashDecodeMQA.op_attn(tq, tk, tv, tsc, tout)
        ttnn.synchronize_device(device)

        out = ttnn.to_torch(tout).float()
        pcc = _pcc(out, ref)
        print(f"\nM2d attn: PCC={pcc:.6f}  out[0,:3]={out[0,:3].tolist()}  ref[0,:3]={ref[0,:3].tolist()}")
        assert pcc >= 0.99, f"attn PCC {pcc:.6f} < 0.99"
        print("M2d OK ✅ — full single-block attention (QK^T+softmax+PV) correct")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
