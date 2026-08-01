# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TT-link transport: KV handoff over the TT-link packet-mode datapath.

Thin adapter over the ``ttlink`` package (packet_decode repo,
DESIGN_TTLINK_API.md -- installed into this venv with ``pip install -e``).
All link logic, firmware knowledge, and guardrails live in the package; this
file only maps the demo's prefill output onto ``ttlink.Pipeline.push()``.

Datapath (GPU 3090 -> BlueField-3 -> P150, v1 host-drain sink):

  prefill_hf on the 3090 -> per-layer bf16 (K,V) device tensors
    -> leg A: async cuMemcpyDtoH into a K-slot pinned ring (no host assembly)
    -> leg B: libttlink_dp.so frames each ring slot (unicast -> RXQ2, 0x88b5)
    -> P150 deposits into the L1 staging window 0x40000-0x60000
    -> host drains L1 via tt_umd per window (zero firmware; the v1 sink)
    -> deserialize -> PrefillResult -> bridge/inject/decode (unchanged)

The KV layout on the wire is the same bf16 row-major convention as
transport.bluefield (per-layer K then V, shapes preserved), so
``_deserialize_kv`` is shared with the BlueField transport.

Ordering constraints (same as BluefieldTransport): deliver() completes --
including the L1 drain into host tensors -- before run_hetero opens the mesh
device, and close() releases this transport's UMD handle so tt-metal and the
transport never hold two UMD contexts on the same chip at once.

Requires: CAP_NET_RAW on the venv interpreter, rshim up, bringup.sh state
applied (the session verifies all of it and fails with the fix instructions).
"""

from __future__ import annotations

import time

import torch
from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs

import prefill_hf
from prefill_hf import PrefillResult
from transport.base import KVTransport
from transport.bluefield import _deserialize_kv

BOARD_LO_DEFAULT = 0x3191A01C   # the BF3-facing P150 on this bench


class TTLinkTransport(KVTransport):
    name = "ttlink"

    def __init__(
        self,
        model_args: ModelArgs,
        device: str = "cpu",
        instruct: bool = False,
        state_dict=None,
        iface: str = None,
        chip: int = BOARD_LO_DEFAULT,
        core: str = "13-1",
        rxq: int = 2,
        frame_payload: int = 4064,
        rate_pps: int = 0,
        verify: bool = True,
        stage_log_path: str = None,
        explicit_log: bool = False,
        **kwargs,
    ):
        self.model_args = model_args
        self.device = device
        self.instruct = instruct
        self.state_dict = state_dict
        self.iface = iface
        self.chip = chip
        self.core = core
        self.rxq = rxq
        self.frame_payload = frame_payload
        self.rate_pps = rate_pps
        self.verify = verify
        self.stage_log_path = stage_log_path
        self.explicit_log = explicit_log
        self._pipe = None

    # --- lazy pipeline: importing this module must not require the package ---
    def _pipeline(self):
        if self._pipe is None:
            from ttlink import Pipeline, StageLog

            log = StageLog(explicit=self.explicit_log, path=self.stage_log_path) \
                if (self.stage_log_path or self.explicit_log) else None
            self._stage_log = log
            self._pipe = Pipeline(
                iface=self.iface,
                chip=self.chip,
                core=self.core,
                rxq=self.rxq,
                frame_payload=self.frame_payload,
                on_stage=self._on_stage if log else None,
            )
            # Verify non-persistent link state (never mutates; the failure
            # message names the fix: rshim restart, bringup.sh, CAP_NET_RAW).
            self._pipe.session.bringup(check_only=True)
            self._pipe.session.open()
            seed = self._pipe.session.arm_rxq(
                rxq=self.rxq, dest_base=0x40000, window_bytes=0x20000,
                handshake=True,
            )
            logger.info(f"ttlink: link up, RXQ{self.rxq} armed, tx_seq seed={seed}")
        return self._pipe

    def _on_stage(self, name, detail=False, **kv):
        log = self._stage_log
        if log is None:
            return
        (log.detail if detail else log.stage)(name, **kv)

    def close(self):
        """Dereg the ring, close the data plane, drop the UMD handle.

        Called by run_hetero after deliver() and before open_mesh_device():
        the KV is already in host tensors, so this transport no longer needs
        the chip. Two UMD contexts on one P150 can conflict.
        """
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None
        if getattr(self, "_stage_log", None) is not None:
            self._stage_log.close()
            self._stage_log = None

    def deliver(self, prompt: str) -> PrefillResult:
        # 1. Producer: HF prefill -> per-layer (K,V) on the prefill device.
        res = prefill_hf.run_prefill(
            self.model_args, prompt, device=self.device,
            instruct=self.instruct, state_dict=self.state_dict,
        )

        # 2. Segments straight out of the tensors (device VAs for CUDA -- no
        #    host blob assembly; leg A scatters them into the pinned ring).
        segs, shapes, keep, total = _kv_segments(res.kv, self.device)
        logger.info(f"ttlink: KV {total / 1e6:.1f} MB in {len(segs)} segments "
                    f"({self.device})")

        # 3. One call: 3090/host -> BF3 -> P150 L1 -> host drain.
        pipe = self._pipeline()
        if self.device == "cuda":
            torch.cuda.synchronize()  # prefill kernels precede leg-A copies
        pipe.snapshot()               # arm the counter oracle before pushing
        t0 = time.time()
        rep = pipe.push(segs, total, rate_pps=self.rate_pps,
                        source="gpu" if self.device == "cuda" else "host")
        dt = time.time() - t0
        logger.info(
            f"ttlink: push done: {rep.frames} frames, {rep.windows} window(s), "
            f"{dt:.2f}s = {rep.gbps_payload:.2f} GB/s payload "
            f"(tx_seq {rep.tx_seq_start}->{rep.tx_seq_end})"
        )
        if rep.verify is not None and not rep.verify.ok:
            raise RuntimeError(f"ttlink oracle failed: {rep.verify.summary()}")
        if rep.cqe_err:
            raise RuntimeError(f"ttlink: {rep.cqe_err} CQE errors, "
                               f"first bad tx_seq {rep.first_bad_tx_seq}")

        # 4. Byte-level round-trip verify against a host copy of the source.
        if self.verify:
            want = _kv_bytes_host(res.kv)
            if rep.drained != want:
                raise RuntimeError("KV blob mismatch after TT-link round-trip")
            logger.info("ttlink: drained bytes byte-identical to source KV")

        kv = _deserialize_kv(rep.drained, shapes)
        return PrefillResult(
            kv=kv,
            first_token=res.first_token,
            prompt_len=res.prompt_len,
            prompt_tokens=res.prompt_tokens,
            logits_last=res.logits_last,
        )


def _kv_segments(kv, device):
    """Per-layer bf16 (K,V) -> [(ptr, nbytes), ...] + shapes + keepalive.

    CUDA tensors contribute device VAs (consumed by leg A's cuMemcpyDtoHAsync);
    CPU tensors contribute host buffer addresses. The returned `keep` list pins
    every tensor/array for the duration of the push.
    """
    segs, shapes, keep = [], [], []
    total = 0
    for k, v in kv:
        shapes.append((tuple(k.shape), tuple(v.shape)))
        for t in (k, v):
            b = t.to(torch.bfloat16).contiguous()
            n = b.numel() * 2
            if device == "cuda":
                keep.append(b)
                ptr = b.data_ptr()
            else:
                u = b.view(torch.uint16).numpy()
                keep.append(u)
                ptr = u.ctypes.data
            segs.append((ptr, n))
            total += n
    return segs, shapes, keep, total


def _kv_bytes_host(kv):
    """The same bf16 layout as a host byte string (the round-trip reference)."""
    parts = []
    for k, v in kv:
        for t in (k, v):
            b = t.to(torch.bfloat16).contiguous().cpu()
            parts.append(b.view(torch.uint16).numpy().tobytes())
    return b"".join(parts)
