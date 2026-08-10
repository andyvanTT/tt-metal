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
import os

import pytest
import torch
from loguru import logger

import models.tt_transformers.tt.common as tt_common
import ttnn
from models.common.sampling import SamplingParams
from models.demos.llama_8b.tt.factory import create_llama8b_model, load_llama8b_optimizations
from models.demos.llama_8b.tt.generator import Generator as Llama8bGenerator
from models.tt_transformers.demo.simple_text_demo import load_inputs, prepare_generator_args, preprocess_inputs_prefill

# Replace the baseline model constructor with the demo fork for the duration of
# the demo.  This lets us reuse the baseline ``prepare_generator_args`` without
# forking it.
tt_common.create_tt_model = create_llama8b_model


@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "input_prompts, instruct, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, data_parallel, num_layers, use_prefetcher, use_hf_rope",
    [
        (
            "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json",
            True,
            32768,
            1,
            32,
            True,
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
            {"temperature": 1.0, "top_k": 32, "top_p": 0.9},
            1,
            None,
            False,
            False,
        ),
    ],
    ids=["Llama-3.1-8B-Instruct"],
)
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

    # Load the prompt file and slice to the batch size, matching the baseline
    # demo (``load_inputs``). Passing the whole file's prompt list with a small
    # batch_size makes ``preprocess_inputs_prefill`` see every prompt as a
    # separate user, which breaks the batch-1 decode shard shape.
    input_prompts, _all_prompts = load_inputs(input_prompts, global_batch_size, instruct)

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
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        sampling_params=device_sampling_params,
    )
    # With device sampling enabled, prefill returns (output_tokens, log_probs);
    # output_tokens is [batch, 1] — the first generated token to feed decode.
    if isinstance(prefill_out, tuple):
        prefilled_token = prefill_out[0]
    else:
        prefilled_token = torch.argmax(prefill_out, dim=-1)
    prefilled_token = prefilled_token.view(global_batch_size, 1)

    logger.info(f"Running forked Llama-8B decode for {max_generated_tokens} tokens...")
    current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
    out_tok = prefilled_token
    for iteration in range(max_generated_tokens):
        decode_out = generator.decode_forward(
            out_tok,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
        )
        # Device sampling returns (tokens, log_probs); tokens is [batch, 1].
        next_tok = decode_out[0] if isinstance(decode_out, tuple) else decode_out
        out_tok = next_tok.view(global_batch_size, 1)
        current_pos += 1

    logger.info("Demo completed.")
