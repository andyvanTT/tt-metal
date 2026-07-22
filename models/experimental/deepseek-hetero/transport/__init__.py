# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cache transport layer for disaggregated prefill→decode.

The transport moves the per-layer KV produced by the (NVIDIA/CPU) prefill side to
the Tenstorrent host, behind a single swappable interface. Selected by one pipeline
parameter (``--transport {pcie,bluefield}``); see ``transport.base.make_transport``.

- ``pcie``      : v1 — host tensors handed over in-process / via PCIe host→device DMA.
- ``bluefield`` : future — KV bytes received over RDMA/Ethernet into a TT host buffer.

Everything downstream of the transport (bridge → inject → decode) is transport-agnostic.
"""
