# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.tt_transformers.tt.model_config import parse_optimizations


def pytest_addoption(parser):
    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption("--input_prompts", action="store", default=None, help="input prompts json file")
    parser.addoption("--instruct", action="store", default=None, type=int, help="Use instruct weights")
    parser.addoption("--max_seq_len", action="store", default=None, type=int, help="Maximum context length")
    parser.addoption("--batch_size", action="store", default=None, type=int, help="Number of users in a batch")
    parser.addoption(
        "--max_generated_tokens", action="store", default=None, type=int, help="Maximum number of tokens to generate"
    )
    parser.addoption("--data_parallel", action="store", default=None, type=int, help="Number of data parallel workers")
    parser.addoption(
        "--paged_attention", action="store", default=None, type=bool, help="Whether to use paged attention"
    )
    parser.addoption(
        "--page_params", action="store", default=None, type=str, help="Page parameters for paged attention"
    )
    parser.addoption(
        "--sampling_params", action="store", default=None, type=str, help="Sampling parameters for decoding"
    )
    parser.addoption("--num_layers", action="store", default=None, type=int, help="Number of layers to use")
    parser.addoption("--use_prefetcher", action="store", default=None, type=bool, help="Whether to use DRAM prefetcher")
    parser.addoption("--use_hf_rope", action="store_true", default=False, help="Whether to use HF-style rope")


@pytest.fixture
def device_params(request):
    """Single-chip Blackhole bring-up overrides for the demo.

    The forked ``text_demo.py`` hardcodes ``fabric_config: True`` in its
    parametrization, which triggers an ethernet fabric router handshake that
    times out on a lone P150/N150 with no healthy remote partner. Flip it off
    for single-chip meshes; this only affects device bring-up, not the model's
    compute path. Also bump the trace region to 100MB: the stock 50MB default
    is too small for the batch-32 decode trace (~51.6MB required).
    """
    params = dict(getattr(request, "param", {}) or {})
    mesh_device_env = os.environ.get("MESH_DEVICE", "")
    if mesh_device_env in ("P150", "N150"):
        params["fabric_config"] = False
    params["trace_region_size"] = 100_000_000
    return params
