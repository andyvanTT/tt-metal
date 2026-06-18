# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2b: validate in-kernel row-softmax of one (32x32) tile vs torch.softmax(dim=-1)."""

import torch
import ttnn

from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        N = 32
        core = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
        grid = ttnn.CoreRangeSet({core})

        def shard():
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (N, N), ttnn.ShardOrientation.ROW_MAJOR),
            )

        torch.manual_seed(0)
        s = torch.randn(N, N, dtype=torch.float32)
        ref = torch.softmax(s, dim=-1)

        ts = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard())
        tsc = ttnn.from_torch(
            torch.ones(N, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard()
        )
        tout = ttnn.from_torch(
            torch.zeros(N, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard()
        )

        FlashDecodeMQA.op_softmax(ts, tsc, tout)
        ttnn.synchronize_device(device)

        out = ttnn.to_torch(tout).float()
        pcc = _pcc(out, ref)
        rowsum = out.sum(-1)
        print(f"\nM2b softmax: PCC={pcc:.6f}  rowsum[:4]={rowsum[:4].tolist()}  (want ~1.0)")
        assert pcc >= 0.99, f"softmax PCC {pcc:.6f} < 0.99"
        print("M2b OK ✅ — in-kernel row-softmax correct")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
