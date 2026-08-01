# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV transport interface + factory — the single swap point for the handoff mechanism.

A ``KVTransport`` moves the per-layer KV (plus first token + prompt length) from the
prefill producer to the Tenstorrent host. Downstream (bridge → inject → decode) is
transport-agnostic: it always receives a ``PrefillResult``.

Swap the whole handoff with one parameter via ``make_transport(name, ...)``:
  - ``pcie``      : v1 — prefill runs in-process; host tensors handed over directly
                    (PCIe host→device DMA happens later, on ttnn.from_torch/fill_cache).
  - ``bluefield`` : KV bytes over AF_PACKET/broadcast frames into the eth_data_rx
                    ERISC firmware's DRAM staging.
  - ``ttlink``    : KV over TT-link packet mode (unicast → RXQ2, EtherType 0x88b5)
                    via the ``ttlink`` package's Pipeline (GPU → BF3 → P150,
                    v1 L1 host-drain sink).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing prefill_hf (→ ttnn) just to reference the type
    from prefill_hf import PrefillResult


class KVTransport(abc.ABC):
    """Moves a prompt's KV handoff from the prefill side to the TT host.

    Implementations differ ONLY in how bytes travel; the returned object is identical,
    so nothing downstream needs to know which transport is in use.
    """

    name: str = "base"

    @abc.abstractmethod
    def deliver(self, prompt: str) -> "PrefillResult":
        """Produce/receive the KV handoff for ``prompt`` and return it on the TT host."""
        raise NotImplementedError


def make_transport(name: str, **kwargs) -> KVTransport:
    """Factory: return the transport selected by ``--transport {pcie,bluefield}``.

    This call in ``run_hetero.py`` is the single, explicit swap point between the
    current PCIe path and the future BlueField/Ethernet path.
    """
    name = name.lower()
    if name == "pcie":
        from transport.pcie import PcieTransport

        return PcieTransport(**kwargs)
    if name == "bluefield":
        from transport.bluefield import BluefieldTransport

        return BluefieldTransport(**kwargs)
    if name == "ttlink":
        from transport.ttlink import TTLinkTransport

        return TTLinkTransport(**kwargs)
    raise ValueError(f"unknown transport {name!r}; expected 'pcie', 'bluefield' or 'ttlink'")
