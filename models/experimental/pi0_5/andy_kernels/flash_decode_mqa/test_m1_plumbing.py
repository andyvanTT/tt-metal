# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: validate the flash_decode_mqa generic_op packaging end-to-end with a
single-core tile copy. Run: python_env/bin/python -m <this path> (standalone)."""

import torch
import ttnn

from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        H, W = 32, 256  # 1 x 8 tiles
        core = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
        grid = ttnn.CoreRangeSet({core})
        shard = ttnn.ShardSpec(grid, (H, W), ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

        torch.manual_seed(0)
        x = torch.randn(H, W, dtype=torch.float32)
        tin = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tout = ttnn.from_torch(
            torch.zeros(H, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )

        FlashDecodeMQA.op_passthrough(tin, tout)
        ttnn.synchronize_device(device)

        out = ttnn.to_torch(tout).float()
        ref = ttnn.to_torch(tin).float()
        match = torch.allclose(out, ref, atol=0, rtol=0)
        maxdiff = (out - ref).abs().max().item()
        print(f"\nM1 passthrough: exact_match={match}  max|diff|={maxdiff:.6f}  " f"out[0,:4]={out[0,:4].tolist()}")
        assert match, "M1 passthrough mismatch — plumbing bug"
        print("M1 OK ✅ — generic_op packaging + CB + kernel compile path works")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
