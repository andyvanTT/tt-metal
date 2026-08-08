# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.demos.llama_8b.tt import attention as llama8b_attention
from models.demos.llama_8b.tt import lm_head as llama8b_lm_head
from models.demos.llama_8b.tt import model_config as llama8b_model_config
from models.demos.llama_8b.tt.generator import Generator as Llama8bGenerator
from models.tt_transformers.tt.generator import create_submeshes
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import parse_decoder_json


def _load_demo_optimizations():
    """Load the demo's performance decoder JSON as a DecodersPrecision instance."""
    config_path = Path(__file__).parent.parent / "configs" / "performance_decoder_config.json"
    optimizations = parse_decoder_json(str(config_path))
    if optimizations is not None:
        optimizations.__name__ = "performance"
    return optimizations


class Llama8bForCausalLM(Llama8bGenerator):
    """vLLM adapter for the Llama-3.1-8B tuning fork.

    Inherits from the forked ``Generator`` so that all prefill/decode behavior
    uses the packed-row continuous-batching fixes, while providing the vLLM
    contract expected by the TT vLLM plugin.
    """

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # Llama8B on N150
        if "Llama-3.1-8B" in model_name and devices_per_dp_cache == 1 and is_wormhole:
            return 32_768
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,  # kept for API compatibility; demo JSON is always used
    ):
        hf_model_name = hf_config._name_or_path
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 32768
            if max_seq_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-Llama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
                    f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in vLLM."
                )

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
        demo_optimizations = _load_demo_optimizations()

        model_args = []
        for submesh in submesh_devices:
            model_args_i = llama8b_model_config.Llama8bModelArgs(
                submesh,
                instruct="Instruct" in hf_model_name,
                max_batch_size=max_batch_size // tt_data_parallel,
                optimizations=lambda model_args: demo_optimizations,
                max_seq_len=max_seq_len,
            )
            assert model_args_i.model_name.replace("-", "") in hf_model_name.replace(
                "-", ""
            ), f"The model specified in vLLM ({hf_model_name}) does not match the model name ({model_args_i.model_name}) with weights ({model_args_i.CKPT_DIR})."
            if n_layers is not None:
                model_args_i.n_layers = n_layers
            model_args.append(model_args_i)

        state_dict = model_args[0].load_state_dict()

        tt_model = []
        for i, submesh in enumerate(submesh_devices):
            tt_model_i = Transformer(
                args=model_args[i],
                mesh_device=submesh,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                weight_cache_path=model_args[i].weight_cache_path(ttnn.bfloat8_b),
                use_paged_kv_cache=True,
                attention_class=llama8b_attention.Attention,
            )
            llama8b_lm_head.apply_lofi_lm_head(tt_model_i)
            tt_model.append(tt_model_i)

        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)
