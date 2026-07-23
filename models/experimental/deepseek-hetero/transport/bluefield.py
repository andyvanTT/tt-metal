# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BlueField/Ethernet transport — moves the KV handoff over a real NIC link into TT DRAM.

Datapath (proven end-to-end at the byte level on the bench, FW 88.1.x / core 13-1):

  producer (HF prefill, CPU/GPU) --serialize--> bf16 row-major KV blob
     --AF_PACKET raw L2 frames (EtherType 0x88B6, broadcast -> P150 RXQ0)-->
  P150 ERISC eth_data_rx firmware --noc_write_reg--> TT DRAM staging
     --DONE doorbell @ L1 0x7C200--> host reads blob back from DRAM (tt_umd)
     --deserialize--> per-layer (K,V) -> PrefillResult -> bridge/inject/decode (unchanged)

This is the real implementation of the seam the module previously stubbed. The RoPE-permute
convention still comes from prefill_hf (HfModelWrapper.cache_k), so the wire carries plain
bf16 and the on-device tilize + bfp8 quant stay in bridge.py / inject.py.

Requirements to run the wire path:
  * The P150 must be running the eth_data_rx firmware with the NIC link up (see the
    bh-erisc-fpga repo; core 13-1 <-> BF3 port0). The BF3 netdev must be a dedicated raw
    L2 pipe (NetworkManager off: `nmcli device set <netdev> managed no`).
  * This process needs CAP_NET_RAW (root) for the AF_PACKET send, and tt_umd for the
    doorbell/DRAM access.
  * `--prefill-device cuda` sources the KV from the 3090 (host-pinned bounce buffer; true
    PCIe P2P is unavailable on this box — see M0).

v1 reads the KV back from DRAM to host and hands torch tensors to bridge.py (which re-uploads
via PCIe). The host-bypass optimization — wrapping the DRAM bytes in place as a device tensor
and injecting without the readback — is a follow-up (see design doc / plan M3 note).
"""

from __future__ import annotations

import socket
import struct
import time

import torch
from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs

import prefill_hf
from prefill_hf import PrefillResult
from transport.base import KVTransport

# --- wire + firmware constants (must match src/common/api/eth_data_rx.h) ---
ETHERTYPE = 0x88B6
BROADCAST = b"\xff\xff\xff\xff\xff\xff"
MAGIC = 0x4E524358  # 'NRCX'
FLAG_EOT = 0x0001
HDR = struct.Struct("<IHHII")  # magic, stream_id, flags, byte_offset, payload_len

NIC_RX_CTRL_ADDR = 0x7C200
# nic_rx_ctrl_t field byte-offsets (host-written control + fw status).
OFF_DST_MODE, OFF_DRAM_X, OFF_DRAM_Y, OFF_DRAM_LO, OFF_DRAM_HI = 0, 4, 8, 12, 16
OFF_STATE, OFF_BYTES = 24, 28  # magic@20, state@24, bytes_received@28
STATE_DONE = 3


def _mac_of(netdev: str) -> bytes:
    with open(f"/sys/class/net/{netdev}/address") as f:
        return bytes(int(b, 16) for b in f.read().strip().split(":"))


class BluefieldTransport(KVTransport):
    name = "bluefield"

    def __init__(
        self,
        model_args: ModelArgs,
        device: str = "cpu",
        instruct: bool = False,
        state_dict=None,
        netdev: str = "enp33s0f0np0",
        eth_core: tuple = (13, 1),
        dram: tuple = (0, 0, 0x10000000),  # (noc_x, noc_y, addr)
        frame_bytes: int = 1024,
        verify: bool = True,
        **kwargs,
    ):
        self.model_args = model_args
        self.device = device
        self.instruct = instruct
        self.state_dict = state_dict
        self.netdev = netdev
        self.eth_core = eth_core
        self.dram = dram
        self.frame_bytes = frame_bytes
        self.verify = verify
        self._umd = None

    # --- tt_umd device handle (lazy; shared no-wait-for-eth-training init) ---
    def _dev(self):
        if self._umd is None:
            import tt_umd

            opts = tt_umd.TopologyDiscoveryOptions()
            for a in ("no_wait_for_eth_training", "no_eth_firmware_strictness"):
                if hasattr(opts, a):
                    setattr(opts, a, True)
            if hasattr(opts, "wait_on_ethernet_link_training"):
                opts.wait_on_ethernet_link_training = False
            for a in ("cmfw_mismatch_action", "eth_fw_mismatch_action", "eth_fw_heartbeat_failure"):
                if hasattr(opts, a):
                    setattr(opts, a, tt_umd.TopologyDiscoveryOptions.Action.IGNORE)
            try:
                cd, devs = tt_umd.TopologyDiscovery.discover(opts, tt_umd.IODeviceType.PCIe)
            except TypeError:
                cd, devs = tt_umd.TopologyDiscovery.discover(opts)
            self._umd = devs[list(cd.get_all_chips())[0]]
        return self._umd

    def _w32(self, x, y, addr, val):
        self._dev().noc_write32(x, y, addr, val & 0xFFFFFFFF)

    def _r32(self, x, y, addr):
        return self._dev().noc_read32(x, y, addr)

    def _read(self, x, y, addr, n):
        n = (n + 3) & ~3
        try:
            return self._dev().noc_read(x, y, addr, n)
        except AttributeError:
            return b"".join(struct.pack("<I", self._r32(x, y, addr + o)) for o in range(0, n, 4))

    # --- steps ---
    def _arm(self, nbytes: int):
        """Point the ERISC receiver at the DRAM target (dst_mode=DRAM)."""
        ex, ey = self.eth_core
        dx, dy, daddr = self.dram
        self._w32(ex, ey, NIC_RX_CTRL_ADDR + OFF_DRAM_X, dx)
        self._w32(ex, ey, NIC_RX_CTRL_ADDR + OFF_DRAM_Y, dy)
        self._w32(ex, ey, NIC_RX_CTRL_ADDR + OFF_DRAM_LO, daddr & 0xFFFFFFFF)
        self._w32(ex, ey, NIC_RX_CTRL_ADDR + OFF_DRAM_HI, (daddr >> 32) & 0xFFFFFFFF)
        self._w32(ex, ey, NIC_RX_CTRL_ADDR + OFF_DST_MODE, 1)  # arm DRAM mode last
        logger.info(f"armed eth{self.eth_core} -> DRAM {self.dram[:2]}@0x{daddr:X} for {nbytes} B")

    def _send(self, blob: bytes):
        """Frame the blob and TX it as broadcast L2 frames (needs CAP_NET_RAW)."""
        src = _mac_of(self.netdev)
        eth = BROADCAST + src + struct.pack("!H", ETHERTYPE)
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETHERTYPE))
        s.bind((self.netdev, 0))
        off = 0
        n = len(blob)
        while off < n:
            m = min(self.frame_bytes, n - off)
            eot = FLAG_EOT if (off + m >= n) else 0
            s.send(eth + HDR.pack(MAGIC, 1, eot, off, m) + blob[off : off + m])
            off += m
        s.close()

    def _wait_done(self, timeout_s: float = 10.0):
        ex, ey = self.eth_core
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self._r32(ex, ey, NIC_RX_CTRL_ADDR + OFF_STATE) == STATE_DONE:
                return self._r32(ex, ey, NIC_RX_CTRL_ADDR + OFF_BYTES)
            time.sleep(0.005)
        raise TimeoutError("eth_data_rx did not reach DONE — check link/firmware/netdev")

    def deliver(self, prompt: str) -> PrefillResult:
        # 1. Producer: HF prefill -> per-layer (K,V), first token, prompt length.
        res = prefill_hf.run_prefill(
            self.model_args, prompt, device=self.device, instruct=self.instruct, state_dict=self.state_dict
        )

        # 2. Serialize to a bf16 row-major blob + per-layer layout (shapes preserved).
        blob, shapes = _serialize_kv(res.kv)

        # 3. Move over the wire into TT DRAM, and wait for the completion doorbell.
        self._arm(len(blob))
        t0 = time.time()
        self._send(blob)
        got = self._wait_done()
        dt = time.time() - t0
        logger.info(
            f"KV over wire: {len(blob)} B in {dt*1e3:.1f} ms host-observed "
            f"({len(blob)*8/dt/1e9:.2f} Gb/s); fw counted {got} B"
        )
        if got != len(blob):
            raise RuntimeError(f"DRAM received {got} B, expected {len(blob)}")

        # 4. Read the KV back from DRAM (v1 host readback) and reconstruct tensors.
        dx, dy, daddr = self.dram
        back = self._read(dx, dy, daddr, len(blob))[: len(blob)]
        if self.verify and back != blob:
            raise RuntimeError("KV blob mismatch after DRAM round-trip (wire corruption)")
        kv = _deserialize_kv(back, shapes)

        return PrefillResult(
            kv=kv,
            first_token=res.first_token,
            prompt_len=res.prompt_len,
            prompt_tokens=res.prompt_tokens,
            logits_last=res.logits_last,
        )


def _serialize_kv(kv):
    """list[(K,V)] (each [B,seq,nkv,hd] float) -> (bf16 bytes, [(shape,shape),...])."""
    parts, shapes = [], []
    for k, v in kv:
        kb = k.to(torch.bfloat16).contiguous()
        vb = v.to(torch.bfloat16).contiguous()
        parts.append(kb.view(torch.uint16).numpy().tobytes())
        parts.append(vb.view(torch.uint16).numpy().tobytes())
        shapes.append((tuple(k.shape), tuple(v.shape)))
    return b"".join(parts), shapes


def _deserialize_kv(blob, shapes):
    """Inverse of _serialize_kv."""
    import numpy as np

    kv, off = [], 0
    for ks, vs in shapes:
        out = []
        for shp in (ks, vs):
            n = 1
            for d in shp:
                n *= d
            nbytes = n * 2
            arr = np.frombuffer(blob[off : off + nbytes], dtype=np.uint16).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shp).float()
            out.append(t)
            off += nbytes
        kv.append((out[0], out[1]))
    return kv
