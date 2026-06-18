# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2a: validate in-kernel S = Q @ K^T (single 32x32 block, single core) vs torch."""

import torch
import ttnn

from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        Sq, D = 32, 256  # one query tile, head_dim 256 (8 contraction tiles)
        core = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
        grid = ttnn.CoreRangeSet({core})

        def shard(h, w):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
            )

        torch.manual_seed(0)
        q = torch.randn(Sq, D, dtype=torch.float32) * 0.1
        k = torch.randn(Sq, D, dtype=torch.float32) * 0.1  # Sk=32 (one tile)
        ref = q @ k.T  # (32, 32)

        tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D))
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(Sq, D))
        tout = ttnn.from_torch(
            torch.zeros(Sq, Sq),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=shard(Sq, Sq),
        )

        FlashDecodeMQA.op_qkt(tq, tk, tout)
        ttnn.synchronize_device(device)

        out = ttnn.to_torch(tout).float()
        pcc = _pcc(out, ref)
        print(f"\nM2a QK^T: PCC={pcc:.6f}  out[0,:4]={out[0,:4].tolist()}  ref[0,:4]={ref[0,:4].tolist()}")
        assert pcc >= 0.99, f"QK^T PCC {pcc:.6f} < 0.99"
        print("M2a OK ✅ — in-kernel Q@K^T (matmul+transpose) correct")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
