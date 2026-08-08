# Llama-3.1-8B demo tuning fork

This directory contains a performance-tuning fork of the
`models/tt_transformers` Llama-3.1-8B implementation.  The goal is to keep the
baseline `tt_transformers` files unchanged while hosting the tuned variants in
a demo subtree.

## What lives here

| Path | Purpose |
|---|---|
| `configs/performance_decoder_config.json` | Per-decoder BFP4/LoFi tuning for all 32 layers. |
| `tt/model_config.py` | `Llama8bModelArgs(ModelArgs)` — force-argmax on single-chip and P150 prefill chunk size. |
| `tt/attention.py` | Forked `Attention` with packed-row `paged_fill_cache` for batched prefill. |
| `tt/lm_head.py` | Helper that applies LoFi math fidelity to a constructed LM head. |
| `tt/generator.py` | Forked `Generator` with packed-row continuous-batching fixes. |
| `tt/factory.py` | `create_llama8b_model()` drop-in replacement for `create_tt_model()`. |
| `tt/generator_vllm.py` | `Llama8bForCausalLM` vLLM adapter. |
| `demo/text_demo.py` | Thin fork of `simple_text_demo.py` wired to the demo factory. |
| `tests/test_fork_equivalence.py` | Unit tests for the forked configuration (no device required). |
| `tests/test_continuous_batching.py` | Placeholder for continuous-batching regression tests. |

## How the baseline is kept clean

The factory wires the forked pieces together without editing any
`tt_transformers` source:

* `Llama8bModelArgs` is a subclass of the baseline `ModelArgs` (only two
  small overrides).
* `Transformer` receives the forked `Attention` via its existing
  `attention_class` constructor argument.
* The LM head is reconfigured after construction by the helper in
  `tt/lm_head.py` (its compute kernel config is consumed lazily on the first
  forward pass).
* The generator is used directly from the forked file.

`tt/attention.py` and `tt/generator.py` are full file forks rather than thin
subclasses because the changes are deep inside large methods (`forward_prefill`,
`prefill_forward_text`, `decode_forward`, `_get_prefill_user_page_table`).
Keeping the whole file avoids fragile partial-method overrides and makes
rebases explicit.

## Running the demo

```bash
pytest models/demos/llama_8b/demo/text_demo.py -k "Llama-3.1-8B-Instruct" \
    --max_seq_len 32768 --batch_size 1 --max_generated_tokens 32
```

The demo defaults to the bundled `configs/performance_decoder_config.json`.  You
may override it with `--decoder_config_file <path>`.

## Running the unit tests

```bash
pytest models/demos/llama_8b/tests/test_fork_equivalence.py -v
```

## CI registration

To run this demo in the TT-Metal CI pipeline, add an entry to the relevant
`tests/pipeline_reorg/models_*.yaml` file pointing at
`models/demos/llama_8b/demo/text_demo.py` and the desired parameter set, or
register the vLLM adapter class name in the TT vLLM plugin.

## Rebase / drift policy

When the baseline `tt_transformers` changes, rebase this fork by:

1. Reverting `tt_transformers` to the new baseline.
2. Re-applying the functional changes (see the original diff stat below) to
   the files in this directory.
3. Updating `factory.py` wiring if the baseline constructor API changes.

Original diff that motivated this fork:

```
models/tt_transformers/model_params/Llama-3.1-8B-Instruct/performance_decoder_config.json | 40 ++++++++++++++++++++-----
models/tt_transformers/tt/attention.py                                                    | 20 +++++++------
models/tt_transformers/tt/generator.py                                                    | 55 +++++++++++++++++++++++-----------
models/tt_transformers/tt/lm_head.py                                                      |  4 +--
models/tt_transformers/tt/model_config.py                                                 |  8 +++--
```
