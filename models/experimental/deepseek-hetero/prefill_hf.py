# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace (torch) prefill side of the heterogeneous pipeline.

Runs a standard HF Qwen forward over the prompt on ``cpu`` or ``cuda`` and returns the
per-layer KV cache in the exact layout the Tenstorrent decode expects, plus the first
sampled token and the prompt length.

Why reuse ``tt_transformers``' ``HfModelWrapper`` instead of raw ``transformers``:
its ``.cache_k`` property already applies ``reverse_permute`` along ``head_dim``, i.e.
it emits **post-RoPE K in Meta-interleaved order** — the same convention the TT KV cache
stores. ``.cache_v`` returns V (no head_dim reshuffle). So the RoPE-convention transform,
the single real correctness trap in this project, is handled by existing, tested code.

Note: importing ``ModelArgs`` pulls in ``ttnn``, so v1 assumes prefill runs on the same
box as decode (CPU or local CUDA). A pure-NVIDIA-host variant (transformers + a standalone
``reverse_permute``, no ttnn) is a future refactor for the BlueField transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs


@dataclass
class PrefillResult:
    """Everything the decode side needs from prefill.

    kv:            per-layer (K, V); each tensor is ``[B, seq, n_kv_heads, head_dim]``.
                   K is post-RoPE in **Meta** order (ready for the TT cache); V is raw.
    first_token:   ``[B, 1]`` int64 — argmax of the last prompt position's logits.
    prompt_len:    number of prompt tokens (== seq, == decode start position).
    prompt_tokens: the encoded prompt ids (for reference decoding / seeding).
    logits_last:   ``[B, vocab]`` last-position logits (kept for PCC checks; optional use).
    """

    kv: List[Tuple[torch.Tensor, torch.Tensor]]
    first_token: torch.Tensor
    prompt_len: int
    prompt_tokens: List[int]
    logits_last: torch.Tensor


@torch.no_grad()
def run_prefill(
    model_args: ModelArgs,
    prompt: str,
    device: str = "cpu",
    instruct: bool = False,
    state_dict: Optional[dict] = None,
) -> PrefillResult:
    """Prefill ``prompt`` with the HF reference model and return its KV + first token.

    ``model_args`` is shared with the decode side so both use the identical config and
    the identical (meta-roundtripped) weights — this guarantees the KV we hand off is
    what the TT model would itself have cached.
    """
    # Meta-format state dict (same conversion the TT model uses). Load once; caller may
    # pass it in to avoid re-reading the checkpoint for both prefill and decode.
    if state_dict is None:
        state_dict = model_args.load_state_dict()

    # HF reference model, wrapped so .cache_k emits Meta-order post-RoPE K.
    reference_model = model_args.reference_transformer()  # wrap=True by default
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Host embedding (Meta weight), matching test_torch.py.
    embd = model_args.reference_embedding()
    prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{prefix}tok_embeddings.weight"]})

    if device != "cpu":
        logger.info(f"Moving HF reference model + embedding to {device}")
        reference_model.model.to(device)
        embd.to(device)

    # Encode the prompt. instruct=True applies the chat template.
    prompt_tokens = model_args.encode_prompt(prompt, instruct=instruct)
    prompt_len = len(prompt_tokens)
    logger.info(f"Prefill prompt: {prompt_len} tokens (instruct={instruct}, device={device})")

    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)  # [1, seq]
    inputs_embeds = embd(tokens)  # [1, seq, hidden]

    # Single causal forward over the whole prompt; populates the wrapper's KV cache.
    logits = reference_model.forward(inputs_embeds, start_pos=0, mode="decode")  # [1, seq, vocab]

    logits_last = logits[:, -1, :].float().cpu()  # [1, vocab]
    first_token = torch.argmax(logits_last, dim=-1, keepdim=True)  # [1, 1]

    # Per-layer KV, moved to host for the transport. K is already Meta-order.
    cache_k = reference_model.cache_k  # list[layer] of [1, seq, n_kv_heads, head_dim]
    cache_v = reference_model.cache_v
    kv = [(k.detach().float().cpu(), v.detach().float().cpu()) for k, v in zip(cache_k, cache_v)]

    logger.info(
        f"Prefill done: {len(kv)} layers, K/V shape {tuple(kv[0][0].shape)}, " f"first_token={first_token.item()}"
    )
    return PrefillResult(
        kv=kv,
        first_token=first_token.cpu(),
        prompt_len=prompt_len,
        prompt_tokens=prompt_tokens,
        logits_last=logits_last,
    )
