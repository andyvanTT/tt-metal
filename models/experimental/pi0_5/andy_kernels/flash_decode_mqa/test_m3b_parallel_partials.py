# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3b: run the flash partials IN PARALLEL across C cores in one generic_op
(K/V height-sharded, Q replicated, mask width-sharded), combine on host, compare
to torch full SDPA. This is the K-parallel layout; M3c moves the combine on-device."""

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
        Sq, D, Sk, C = 32, 256, 1056, 11  # 11 cores, each Skt=3 (96 keys)
        assert (Sk // 32) % C == 0, "C must divide Skt"
        scale = 1.0 / math.sqrt(D)
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
        q = torch.randn(Sq, D) * 0.3
        k = torch.randn(Sk, D) * 0.3
        v = torch.randn(Sk, D) * 0.3
        mask_row = torch.zeros(Sk)
        mask_row[Sk - 40 :] = -1e4
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[None, None],
            k[None, None],
            v[None, None],
            attn_mask=mask_row.view(1, Sk).expand(Sq, Sk)[None, None],
            scale=scale,
            is_causal=False,
        )[0, 0]

        qs = q * scale
        tq = dev(qs.repeat(C, 1), hshard(C * Sq, D))  # replicate Q to each core
        tk = dev(k, hshard(Sk, D))  # K slice per core
        tv = dev(v, hshard(Sk, D))
        tsc = dev(torch.ones(C * 32, 32), hshard(C * 32, 32))
        tm = dev(mask_row.view(1, Sk).expand(Sq, Sk).contiguous(), wshard(Sq, Sk))  # mask cols per core
        tout = dev(torch.zeros(C * Sq, D), hshard(C * Sq, D))
        tmo = dev(torch.zeros(C * 32, 32), hshard(C * 32, 32))
        tlo = dev(torch.zeros(C * 32, 32), hshard(C * 32, 32))

        FlashDecodeMQA.op_partial(tq, tk, tv, tsc, tm, tout, tmo, tlo)
        ttnn.synchronize_device(device)

        O_all = ttnn.to_torch(tout).float().reshape(C, Sq, D)
        m_all = ttnn.to_torch(tmo).float().reshape(C, 32, 32)[:, :, 0]  # (C, Sq) row max
        l_all = ttnn.to_torch(tlo).float().reshape(C, 32, 32)[:, :, 0]  # (C, Sq) row sum

        # online-softmax combine across the C cores (host)
        M = torch.full((Sq,), -1e30)
        L = torch.zeros(Sq)
        Oacc = torch.zeros(Sq, D)
        for c in range(C):
            mc, lc, Oc = m_all[c], l_all[c], O_all[c]
            Mnew = torch.maximum(M, mc)
            sM = torch.exp(M - Mnew)[:, None]
            sc = torch.exp(mc - Mnew)[:, None]
            Oacc = Oacc * sM + Oc * sc
            L = L * sM[:, 0] + lc * sc[:, 0]
            M = Mnew
        out = Oacc / L[:, None]

        pcc = _pcc(out, ref)
        print(f"\nM3b parallel partials (Sk={Sk}, C={C} cores): PCC={pcc:.6f}  out[0,:3]={out[0,:3].tolist()}")
        assert pcc >= 0.99, f"M3b PCC {pcc:.6f} < 0.99"
        print("M3b OK ✅ — K-parallel partials across cores + combine reproduce full attention")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
