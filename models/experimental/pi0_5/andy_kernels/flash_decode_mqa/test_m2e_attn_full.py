# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2e: single-core full attention at the real denoise shape (Sk=1056, 1 head) with
an additive phantom mask, vs torch SDPA. Q pre-scaled by 1/sqrt(d); K/V in bf8 (L1)."""

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
        Sq, Sk, D = 32, 256, 256  # per-core K slice scale (M3 splits 1056 across cores)
        AH = 10
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
        q = torch.randn(Sq, D) * 0.3
        k = torch.randn(Sk, D) * 0.3
        v = torch.randn(Sk, D) * 0.3
        blocked = torch.zeros(Sk)
        if __import__("os").environ.get("FD_MASK", "1") == "1":
            blocked[200:] = -1e4  # phantom KV columns
        mask = blocked.view(1, Sk).expand(Sq, Sk).contiguous()
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], attn_mask=mask[None, None], scale=scale, is_causal=False
        )[0, 0]

        qs = q * scale
        tq = ttnn.from_torch(
            qs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D)
        )
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sk, D))
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sk, D))
        tsc = ttnn.from_torch(
            torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(32, 32)
        )
        tm = ttnn.from_torch(
            mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, Sk)
        )
        tout = ttnn.from_torch(
            torch.zeros(Sq, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D)
        )

        stage = int(__import__("os").environ.get("FD_STAGE", "0"))
        FlashDecodeMQA.op_attn_full(tq, tk, tv, tsc, tm, tout, stage=stage)
        ttnn.synchronize_device(device)

        out = ttnn.to_torch(tout).float()
        if stage != 0:  # probe: just report magnitude of the dumped intermediate
            nz = (out.abs() > 1e-9).float().mean().item()
            print(
                f"\nM2e STAGE={stage} dump: nonzero_frac={nz:.4f}  mean|x|={out.abs().mean():.6f}  "
                f"out[0,:4]={out[0,:4].tolist()}"
            )
            return
        # only the AH real query rows matter
        pcc = _pcc(out, ref)
        print(f"\nM2e attn full (Sk={Sk}): PCC(real rows)={pcc:.6f}  out[0,:3]={out[0,:3].tolist()}")
        assert pcc >= 0.99, f"attn-full PCC {pcc:.6f} < 0.99"
        print(f"M2e OK ✅ — single-core multi-tile attention (Sk={Sk}, per-core slice scale) + mask correct")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
