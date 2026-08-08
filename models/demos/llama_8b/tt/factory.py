# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
from models.demos.llama_8b.tt import attention as llama8b_attention
from models.demos.llama_8b.tt import lm_head as llama8b_lm_head
from models.demos.llama_8b.tt import model_config as llama8b_model_config
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import parse_decoder_json
from models.tt_transformers.tt.prefetcher import Prefetcher


def create_llama8b_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config=None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
    use_prefetcher=False,
    use_hf_rope=False,
):
    """Create a Llama-3.1-8B model with the demo tuning fork.

    Mirrors the baseline ``create_tt_model`` API so that the demo and vLLM
    adapter can drop it in as a replacement.
    """
    num_tensors = 5 if use_prefetcher else 0
    prefetcher = Prefetcher(mesh_device, num_tensors, num_layers) if use_prefetcher else None

    tt_model_args = llama8b_model_config.Llama8bModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        prefetcher=prefetcher,
        use_hf_rope=use_hf_rope,
    )

    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    if prefetcher is not None:
        prefetcher.num_layers = tt_model_args.n_layers

    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    # Inject the forked attention class via the existing Transformer hook.
    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        prefetcher=prefetcher,
        attention_class=llama8b_attention.Attention,
    )

    # Apply the LoFi LM head tuning without forking the LMHead class. The
    # compute kernel config is consumed lazily on the first forward pass.
    for model_instance in model if isinstance(model, (list, tuple)) else [model]:
        llama8b_lm_head.apply_lofi_lm_head(model_instance)

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict


def load_llama8b_optimizations(config_path=None):
    """Load the demo's performance decoder JSON.

    If ``config_path`` is None, use the JSON bundled with this demo.
    ``parse_decoder_json`` returns a DecodersPrecision instance that can be
    passed directly to ModelArgs as the ``optimizations`` argument.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "performance_decoder_config.json"
    optimizations = parse_decoder_json(str(config_path))
    if optimizations is not None:
        optimizations.__name__ = "performance"
    return optimizations
