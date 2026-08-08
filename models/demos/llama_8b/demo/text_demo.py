# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin fork of simple_text_demo.py wired to the Llama-8B demo factory.

This demo reuses the baseline ``prepare_generator_args`` and input
preprocessing from ``models/tt_transformers.demo.simple_text_demo`` and only
replaces the model constructor with ``create_llama8b_model`` and the generator
with the forked ``Generator``.  Run it the same way you would run the
corresponding baseline demo, e.g.:

    pytest models/demos/llama_8b/demo/text_demo.py -k "Llama-3.1-8B-Instruct" \
        --max_seq_len 32768 --batch_size 1 --max_generated_tokens 32
"""

import json

import torch
from loguru import logger

import models.tt_transformers.tt.common as tt_common
import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama_8b.tt.factory import create_llama8b_model, load_llama8b_optimizations
from models.demos.llama_8b.tt.generator import Generator as Llama8bGenerator
from models.tt_transformers.demo.simple_text_demo import prepare_generator_args, preprocess_inputs_prefill

# Replace the baseline model constructor with the demo fork for the duration of
# the demo.  This lets us reuse the baseline ``prepare_generator_args`` without
# forking it.
tt_common.create_tt_model = create_llama8b_model


def test_demo_text(
    input_prompts,
    instruct,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    mesh_device,
    data_parallel,
    num_layers,
    use_prefetcher,
    use_hf_rope,
    request,
):
    """Minimal Llama-3.1-8B demo using the demo tuning fork."""
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    global_batch_size = batch_size * data_parallel

    # Allow command-line overrides where the baseline conftest registers them.
    input_prompts = request.config.getoption("--input_prompts") or input_prompts
    if request.config.getoption("--instruct") in [0, 1]:
        instruct = request.config.getoption("--instruct")
    max_seq_len = request.config.getoption("--max_seq_len") or max_seq_len
    batch_size = request.config.getoption("--batch_size") or batch_size
    max_generated_tokens = request.config.getoption("--max_generated_tokens") or max_generated_tokens
    data_parallel = request.config.getoption("--data_parallel") or data_parallel
    paged_attention = request.config.getoption("--paged_attention") or paged_attention
    page_params = request.config.getoption("--page_params") or page_params
    if isinstance(page_params, str):
        page_params = json.loads(page_params)
    sampling_params = request.config.getoption("--sampling_params") or sampling_params
    num_layers = request.config.getoption("--num_layers") or num_layers
    use_prefetcher = request.config.getoption("--use_prefetcher") or use_prefetcher
    use_hf_rope = request.config.getoption("--use_hf_rope")

    json_config_file = request.config.getoption("--decoder_config_file")
    if json_config_file:
        from models.tt_transformers.tt.model_config import parse_decoder_json

        optimizations = parse_decoder_json(json_config_file)
    else:
        optimizations = request.config.getoption("--optimizations") or load_llama8b_optimizations()

    # Reuse the baseline helper that builds submeshes, page tables, and caches.
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        local_data_parallel,
        local_submesh_indices,
    ) = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        num_layers=num_layers,
        use_prefetcher=use_prefetcher,
        use_hf_rope=use_hf_rope,
    )

    generator = Llama8bGenerator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Minimal single-batch run: tokenize the first prompt and run one prefill
    # plus ``max_generated_tokens`` decode steps.  This is intentionally
    # simplified compared to the full baseline demo.
    if len(input_prompts) == 1:
        input_prompts = input_prompts * global_batch_size

    (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(input_prompts, tokenizer, model_args, instruct, max_generated_tokens, max_seq_len)

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill).view(global_batch_size, -1)

    device_sampling_params = SamplingParams(
        temperature=sampling_params["temperature"],
        top_k=sampling_params["top_k"],
        top_p=sampling_params["top_p"],
        seed=sampling_params.get("seed"),
    )

    logger.info("Running forked Llama-8B prefill...")
    generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=prefill_lens,
        empty_slots=list(range(global_batch_size)),
        sampling_params=device_sampling_params,
        start_pos=decoding_pos,
    )

    logger.info(f"Running forked Llama-8B decode for {max_generated_tokens} tokens...")
    for _ in range(max_generated_tokens):
        generator.decode_forward(
            tokens=input_tokens_prefill_pt,
            start_pos=decoding_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
        )

    logger.info("Demo completed.")
