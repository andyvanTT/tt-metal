# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""T4 — full heterogeneous pipeline vs a full-HF greedy reference.

Runs prefill (CPU HF) → bridge → inject → traced TT decode.

Correctness is measured two ways:
  1. Free-running greedy output must be coherent and its first token must match HF (a direct
     test that injection + the RoPE convention are correct).
  2. **Teacher-forced per-step top-1** agreement with HF — at each position the reference token
     is fed back, so a single bfp8-induced difference does not cascade. This is the standard
     accuracy metric (same idea as tt_transformers' ci-token-matching); free-running greedy
     comparison is NOT a valid metric because greedy divergence is chaotic.
"""

import torch
from loguru import logger

import bridge
import decode_driver
import inject
import prefill_hf

PROMPT = "Once upon a time"
MAX_TOKENS = 24


def _hf_greedy(model_args, state_dict, prompt_tokens, n_new):
    """Greedy continuation from the same (meta-roundtripped) HF weights, on CPU."""
    wrapper = model_args.reference_transformer()  # wrap=True
    wrapper.load_state_dict(state_dict)
    wrapper.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    out = wrapper.model.generate(input_ids, max_new_tokens=n_new, do_sample=False, num_beams=1, use_cache=True)
    return out[0, len(prompt_tokens) :].tolist()


def _fresh_inject(tt, res, mesh):
    kv_dev = bridge.to_device_kv(res.kv, mesh)
    inject.inject_kv(tt.model, kv_dev, batch_idx=0)


def _teacher_forced_top1(tt, first_token, prompt_len, hf_new):
    """Feed the HF reference token at each step; count how often TT's argmax == HF next token."""
    out_tok = first_token.view(1, 1).to(torch.int64)
    current_pos = torch.tensor([prompt_len], dtype=torch.int64)
    matches = 0
    for i in range(len(hf_new) - 1):
        logits, _ = tt.generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=None,
            kv_cache=None,
            reset_batch=(i == 0),
            sampling_params=None,
        )
        pred = int(torch.argmax(logits.reshape(logits.shape[0], -1), dim=-1)[0].item())
        matches += int(pred == hf_new[i + 1])
        out_tok = torch.tensor([[hf_new[i + 1]]], dtype=torch.int64)  # teacher force
        current_pos = current_pos + 1
        if i == 0:
            first_pred = pred
    return matches, len(hf_new) - 1, first_pred


def test_hetero_vs_hf_greedy(mesh):
    tt = decode_driver.build_tt_model(mesh, max_seq_len=1024, max_batch_size=1, instruct=False)
    res = prefill_hf.run_prefill(tt.model_args, PROMPT, device="cpu", instruct=False, state_dict=tt.state_dict)

    hf_new = _hf_greedy(tt.model_args, tt.state_dict, res.prompt_tokens, MAX_TOKENS)

    # (1) free-running output for coherence + first-token check
    _fresh_inject(tt, res, mesh)
    gen = decode_driver.run_decode(tt, res.first_token, res.prompt_len, max_tokens=MAX_TOKENS)
    text = tt.model_args.tokenizer.decode(res.prompt_tokens + gen)
    logger.info("hetero text : " + repr(text))
    logger.info(f"hetero gen  : {gen}")
    logger.info(f"hf greedy   : {[res.first_token.item()] + hf_new[:len(gen)]}")

    # (2) teacher-forced per-step top-1 (re-inject to reset the cache first)
    _fresh_inject(tt, res, mesh)
    matches, total, first_pred = _teacher_forced_top1(tt, res.first_token, res.prompt_len, hf_new)
    top1 = matches / total
    logger.info(f"teacher-forced top-1: {matches}/{total} = {top1:.2%}  (first_pred={first_pred}, hf={hf_new[1]})")

    assert gen[0] == hf_new[0], f"first token mismatch: hetero {gen[0]} vs hf {hf_new[0]}"
    assert top1 >= 0.6, f"teacher-forced top-1 too low: {top1:.2%}"
