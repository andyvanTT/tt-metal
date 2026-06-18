# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3a: validate the flash partial + online-softmax combine. Split Sk into NSLICE
K-slices, run op_partial on each (un-normalized O_c, m_c, l_c), combine on host, and
compare to torch full SDPA. Proves the K-parallel math before cross-core sync (M3b/c)."""

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
        Sq, D, NSLICE, SLICE = 32, 256, 4, 256  # Sk = NSLICE*SLICE = 1024
        Sk = NSLICE * SLICE
        scale = 1.0 / math.sqrt(D)
        core = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
        grid = ttnn.CoreRangeSet({core})

        def shard(h, w):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
            )

        def to_dev(t, h, w, dt=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard(h, w))

        torch.manual_seed(0)
        q = torch.randn(Sq, D) * 0.3
        k = torch.randn(Sk, D) * 0.3
        v = torch.randn(Sk, D) * 0.3
        mask_row = torch.zeros(Sk)
        mask_row[Sk - 40 :] = -1e4  # some phantom columns
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[None, None],
            k[None, None],
            v[None, None],
            attn_mask=mask_row.view(1, Sk).expand(Sq, Sk)[None, None],
            scale=scale,
            is_causal=False,
        )[0, 0]

        qs = q * scale
        tq = to_dev(qs, Sq, D)
        tsc = to_dev(torch.ones(32, 32), 32, 32)

        # per-slice partials, combined on host (online softmax)
        O = None
        M = torch.full((Sq, 1), -1e30)
        L = torch.zeros(Sq, 1)
        Oacc = torch.zeros(Sq, D)
        for s in range(NSLICE):
            ks = k[s * SLICE : (s + 1) * SLICE]
            vs = v[s * SLICE : (s + 1) * SLICE]
            ms = mask_row[s * SLICE : (s + 1) * SLICE].view(1, SLICE).expand(Sq, SLICE).contiguous()
            tk, tv = to_dev(ks, SLICE, D), to_dev(vs, SLICE, D)
            tm = to_dev(ms, Sq, SLICE)
            tout = to_dev(torch.zeros(Sq, D), Sq, D)
            tmo = to_dev(torch.zeros(32, 32), 32, 32)
            tlo = to_dev(torch.zeros(32, 32), 32, 32)
            FlashDecodeMQA.op_partial(tq, tk, tv, tsc, tm, tout, tmo, tlo)
            ttnn.synchronize_device(device)
            Oc = ttnn.to_torch(tout).float()  # (Sq, D) un-normalized
            mc = ttnn.to_torch(tmo).float()[:, :1]  # (Sq,1) row max
            lc = ttnn.to_torch(tlo).float()[:, :1]  # (Sq,1) row sum
            # online-softmax combine
            Mnew = torch.maximum(M, mc)
            scaleM = torch.exp(M - Mnew)
            scalec = torch.exp(mc - Mnew)
            Oacc = Oacc * scaleM + Oc * scalec
            L = L * scaleM + lc * scalec
            M = Mnew
        out = Oacc / L

        pcc = _pcc(out, ref)
        print(f"\nM3a partial+combine (Sk={Sk}, {NSLICE} slices): PCC={pcc:.6f}  out[0,:3]={out[0,:3].tolist()}")
        assert pcc >= 0.99, f"M3a PCC {pcc:.6f} < 0.99"
        print("M3a OK ✅ — flash partials + online-softmax combine reproduce full attention")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
