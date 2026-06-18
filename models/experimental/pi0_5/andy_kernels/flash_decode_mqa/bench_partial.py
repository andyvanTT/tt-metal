# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Interim perf signal: device time of the K-parallel partial op (1 head, Sk=1056,
C cores) via trace replay, vs the ~116 us/call prefill-SDPA baseline (8 heads). Not
yet apples-to-apples (no combine, 1 head), but tells us if the K-split compute is fast."""

import math
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA

INNER, WARM, ITERS = 25, 3, 30


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    try:
        Sq, D, Sk = 32, 256, 1056
        # C=11 fits one grid row (device is 13 wide). Higher C (33) needs a 2D core
        # layout (M3d will lay 8 heads x C cores into 2D). C=11 is the clean 1-row case.
        for C in (11,):
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(C - 1, 0))})

            def hshard(h, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (h // C, w), ttnn.ShardOrientation.ROW_MAJOR),
                )

            def wshard(h, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (h, w // C), ttnn.ShardOrientation.ROW_MAJOR),
                )

            def dev(t, mc):
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

            torch.manual_seed(0)
            qs = (torch.randn(Sq, D) * 0.3) * (1.0 / math.sqrt(D))
            tq = dev(qs.repeat(C, 1), hshard(C * Sq, D))
            tk = dev(torch.randn(Sk, D) * 0.3, hshard(Sk, D))
            tv = dev(torch.randn(Sk, D) * 0.3, hshard(Sk, D))
            tsc = dev(torch.ones(C * 32, 32), hshard(C * 32, 32))
            tm = dev(torch.zeros(Sq, Sk), wshard(Sq, Sk))
            tout = dev(torch.zeros(C * Sq, D), hshard(C * Sq, D))
            tmo = dev(torch.zeros(C * 32, 32), hshard(C * 32, 32))
            tlo = dev(torch.zeros(C * 32, 32), hshard(C * 32, 32))

            def run():
                FlashDecodeMQA.op_partial(tq, tk, tv, tsc, tm, tout, tmo, tlo)

            for _ in range(WARM):
                run()
                ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(INNER):
                run()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) * 1e6 / INNER)
            ttnn.release_trace(device, tid)
            print(
                f"   partial op  C={C:2d} cores (1 head, Sk={Sk}):  {statistics.mean(ts):7.2f} us/call  (min {min(ts):.2f})"
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
