# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BlueField/Ethernet transport (FUTURE — stub).

Intended design (host-bypassing where possible):

  Producer (NVIDIA GPU box)                    Consumer (this TT host)
  ─────────────────────────                    ───────────────────────
  HF/TRT-LLM prefill → per-layer K,V   ──RDMA──▶  receive raw bytes into a
  as **bf16, RoPE-permuted, row-major**  (RoCE/   registered host buffer, then
  (block-float quant is NOT done here —   Ether-  wrap with zero-copy:
  it happens on-device on TT)             net)      ttnn.experimental.disaggregation
                                                    .tensor_from_bf16_bytes(buf, shape)

Why bf16 on the wire: the delta between NVIDIA and TT KV is (1) RoPE interleave, (2) tiling,
(3) numeric encoding. (1) is a cheap permute done on the producer/DPU; (2)+(3) (tilize +
bfp8 block-float) are done **on the TT device** by the bridge, so the DPU never has to run
block-float quantization (there is no turnkey DOCA primitive for it). A BlueField DPU can
carry the transport (RDMA) and, optionally, the cheap permute in its Arm/DPA datapath.

Open integration question (see design doc §caveat): TT's Ethernet is its own fabric
(TT-link/compliance mode), not a standard RoCE endpoint — so the realistic v1 path is
BlueField → TT **host** RAM (RDMA) → TT DRAM (PCIe DMA), unless TT DRAM can be exposed as a
direct RDMA target.

This class only fixes the interface/seam; the datapath is not implemented yet.
"""

from __future__ import annotations

from prefill_hf import PrefillResult
from transport.base import KVTransport

_NOT_IMPLEMENTED = (
    "BluefieldTransport is a future stub. v1 uses --transport pcie. "
    "Implementing this means: register an RDMA-able host buffer, receive per-layer bf16 "
    "K/V bytes from the GPU producer, and wrap them with "
    "ttnn.experimental.disaggregation.tensor_from_bf16_bytes; on-device tilize+bfp8 quant "
    "is then done by bridge.py (unchanged). See this module's docstring."
)


class BluefieldTransport(KVTransport):
    name = "bluefield"

    def __init__(self, **kwargs):
        # Keep kwargs so the CLI/factory signature is stable once implemented
        # (e.g. endpoint, nic, qp params). No datapath yet.
        self.kwargs = kwargs

    def deliver(self, prompt: str) -> PrefillResult:
        raise NotImplementedError(_NOT_IMPLEMENTED)
