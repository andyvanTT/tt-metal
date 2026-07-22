# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tenstorrent decode side: build the TT model + cache, then run a traced, no-prefill
decode loop seeded by the injected KV.

We reuse ``tt_transformers`` wholesale:
  - ``create_tt_model`` builds the ``Transformer`` and (non-paged) allocates each layer's
    persistent DRAM KV cache at ``layer.attention.layer_past``.
  - ``Generator.decode_forward`` runs one **traced** decode step (``enable_trace=True``);
    rope indices derive from ``current_pos`` internally, so decode needs no prefill call.

v1 uses **non-paged** attention (``paged_attention_config=None`` → ``page_table=None``,
``kv_cache=None``); the decode path branches to non-paged SDPA and reads each layer's
``layer_past`` — exactly the cache ``inject.py`` fills. Injection must happen BEFORE the
first ``decode_forward`` (which captures the trace over the persistent cache addresses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from loguru import logger

from models.tt_transformers.tt.common import create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


def _default_optimizations():
    return lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)


@dataclass
class TTDecode:
    model_args: object
    model: object
    generator: Generator
    state_dict: dict


def build_tt_model(
    mesh_device,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    instruct: bool = False,
    optimizations=None,
) -> TTDecode:
    """Build the non-paged TT decode model + KV cache and wrap it in a ``Generator``."""
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations or _default_optimizations(),
        max_seq_len=max_seq_len,
        paged_attention_config=None,  # non-paged: cache lives in layer.attention.layer_past
        state_dict=None,
    )
    assert tt_kv_cache is None, "expected non-paged model (tt_kv_cache is None)"
    generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
    logger.info(f"Built TT decode model: {model_args.n_layers} layers, max_seq_len={max_seq_len}")
    return TTDecode(model_args=model_args, model=model, generator=generator, state_dict=state_dict)


@torch.no_grad()
def run_decode(
    tt: TTDecode,
    first_token: torch.Tensor,
    prompt_len: int,
    max_tokens: int = 32,
    enable_trace: bool = True,
) -> List[int]:
    """Greedy, traced decode starting from the injected KV + ``first_token``.

    Returns the generated token ids (including ``first_token``). Stops at a stop token
    or after ``max_tokens``. Uses non-paged decode (``page_table=None, kv_cache=None``).
    """
    generator = tt.generator
    stop_tokens = set(getattr(tt.model_args.tokenizer, "stop_tokens", []) or [])

    out_tok = first_token.view(1, 1).to(torch.int64)  # [B=1, 1]
    current_pos = torch.tensor([prompt_len], dtype=torch.int64)
    generated = [int(out_tok.item())]

    for step in range(max_tokens):
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=None,
            kv_cache=None,
            reset_batch=(step == 0),
            sampling_params=None,  # host greedy sampling below
        )
        # logits: [B, (1,) vocab] -> [B, vocab]; greedy argmax.
        next_tok = torch.argmax(logits.reshape(logits.shape[0], -1), dim=-1, keepdim=True)  # [B,1]
        tok_id = int(next_tok[0].item())
        current_pos = current_pos + 1
        out_tok = next_tok.to(torch.int64)
        if tok_id in stop_tokens:
            logger.info(f"Hit stop token at step {step}")
            break
        generated.append(tok_id)

    logger.info(f"Decoded {len(generated)} tokens (max_tokens={max_tokens})")
    return generated
