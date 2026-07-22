# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCIe transport (v1): prefill runs in-process; KV handed over as host tensors.

There is no network hop here — the prefill producer runs in the same process (on CPU or
a local CUDA device) and returns host tensors directly. The actual PCIe host→device DMA
happens later, downstream, when the bridge calls ``ttnn.from_torch`` / ``ttnn.fill_cache``
to place the KV into device DRAM. This transport therefore just runs prefill and returns
its result, so the pipeline shape is identical to the future BlueField path.
"""

from __future__ import annotations

from models.tt_transformers.tt.model_config import ModelArgs

import prefill_hf
from prefill_hf import PrefillResult
from transport.base import KVTransport


class PcieTransport(KVTransport):
    name = "pcie"

    def __init__(self, model_args: ModelArgs, device: str = "cpu", instruct: bool = False, state_dict=None):
        """
        model_args: shared config/weights object (same one the decode side uses).
        device:     where HF prefill runs — "cpu" (v1 default) or "cuda".
        instruct:   apply the chat template when encoding the prompt.
        state_dict: optional shared meta state dict (avoids re-reading the checkpoint).
        """
        self.model_args = model_args
        self.device = device
        self.instruct = instruct
        self.state_dict = state_dict

    def deliver(self, prompt: str) -> PrefillResult:
        return prefill_hf.run_prefill(
            self.model_args, prompt, device=self.device, instruct=self.instruct, state_dict=self.state_dict
        )
