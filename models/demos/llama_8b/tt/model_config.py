# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs


class Llama8bModelArgs(ModelArgs):
    """ModelArgs fork for the Llama-3.1-8B demo tuning.

    Keeps the full tt_transformers baseline configuration and only overrides
    the two sites that were changed in the original branch:
      1. Force-argmax sampling on single-chip.
      2. P150 max prefill chunk size for Llama-3.1-8B.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Branch change: enable force-argmax on single-chip where sampling is on
        # the decode critical path. Galaxy already has a model-specific override
        # that keeps it enabled; leave that untouched.
        if not (self.is_galaxy_cluster and self.base_model_name == "Llama-3.1-8B"):
            self.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = self.num_devices == 1

    def get_max_prefill_chunk_size(self):
        # Branch change: add P150 entry for Llama-3.1-8B.
        max_prefill_chunk_size_div1024 = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if max_prefill_chunk_size_div1024 is None:
            MAX_PREFILL_CHUNK_SIZES_DIV1024 = {
                "Llama-3.2-1B": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.2-3B": {"N150": 8, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150": 16, "P150x4": 128},
                "Llama-3.2-11B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "Llama-3.1-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "Llama-3.2-90B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "DeepSeek-R1-Distill-Llama-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128, "P150x4": 128},
                "Qwen2.5-7B": {"N150": 4, "N300": 32, "T3K": 128, "TG": 128, "P150x4": 128},
                "Qwen2.5-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128, "P150x4": 128, "P150x8": 128},
                "Qwen2.5-72B": {"N150": None, "N300": None, "T3K": 16, "TG": 128, "P150x4": 128, "P150x8": 128},
                "Qwen2.5-VL-3B": {"N150": 128, "N300": 128, "T3K": None, "TG": None, "P150x4": None},
                "Qwen2.5-VL-7B": {"N150": 64, "N300": 128, "T3K": None, "TG": None, "P150x4": None},
                "Qwen2.5-VL-32B": {"N150": None, "N300": None, "T3K": 64, "TG": None, "P150x4": None},
                "Qwen2.5-VL-72B": {"N150": None, "N300": None, "T3K": 32, "TG": None, "P150x4": None},
                "Qwen3-VL-32B": {"N150": None, "N300": None, "T3K": 64, "TG": None, "P150x4": None, "P150x8": 64},
                "DeepSeek-R1-Distill-Qwen-14B": {"N150": 4, "N300": 64, "T3K": 128, "TG": None, "P150x4": None},
                "Phi-3.5-mini-instruct": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "Phi-3-mini-128k-instruct": {"N150": 32, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "QwQ-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128, "P150x4": 128},
                "Qwen3-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128, "P150x4": 128},
                "Qwen3-Embedding-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "Phi-4": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128, "P150x4": 128},
                "Mistral-Small-3.1-24B": {"N150": 8, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "gemma-3-1b": {"N150": 32, "N300": 32, "T3K": 32, "TG": 32, "P150x4": 32},
                "gemma-3-4b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "medgemma-4b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "gemma-3-27b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "medgemma-27b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
            }
            try:
                max_prefill_chunk_size_div1024 = MAX_PREFILL_CHUNK_SIZES_DIV1024[self.base_model_name][self.device_name]
            except KeyError:
                logger.warning(
                    f"Unknown model {self.model_name} on device {self.device_name}, setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"
                )
                logger.warning(
                    "Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128 for faster performance (if you run out of L1 memory it was too high)"
                )
                max_prefill_chunk_size_div1024 = 4
            assert (
                max_prefill_chunk_size_div1024 is not None
            ), f"Unsupported model {self.model_name} on device {self.device_name}"
        else:
            max_prefill_chunk_size_div1024 = int(max_prefill_chunk_size_div1024)
        return max_prefill_chunk_size_div1024 * 1024
