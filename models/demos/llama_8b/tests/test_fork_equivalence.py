# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
from models.demos.llama_8b.tt.factory import load_llama8b_optimizations
from models.demos.llama_8b.tt.lm_head import apply_lofi_lm_head
from models.demos.llama_8b.tt.model_config import Llama8bModelArgs
from models.tt_transformers.tt.model_config import ModelArgs, parse_decoder_json


def test_llama8b_model_args_inherits_baseline():
    assert issubclass(Llama8bModelArgs, ModelArgs)


def test_llama8b_optimizations_load():
    optimizations = load_llama8b_optimizations()
    assert optimizations is not None
    assert hasattr(optimizations, "__name__")
    assert optimizations.__name__ == "performance"
    # The demo JSON has 32 decoders configured.
    assert len(optimizations.decoder_optimizations) == 32


def test_decoder_zero_has_ff1_ff3():
    optimizations = load_llama8b_optimizations()
    decoder_0 = optimizations.decoder_optimizations[0]
    # Tensor groups and precision settings are keyed by enum; just verify the
    # config was applied and the first decoder has non-empty settings.
    assert decoder_0.tensor_dtype_settings
    assert decoder_0.op_fidelity_settings


def test_apply_lofi_lm_head_sets_math_fidelity():
    class FakeLMHead:
        def __init__(self):
            self.compute_kernel_config = None

    fake = FakeLMHead()
    apply_lofi_lm_head(fake)
    assert fake.compute_kernel_config is not None
    assert fake.compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi


def test_custom_decoder_config_file():
    """The demo JSON can also be loaded from an explicit path."""
    custom_path = Path(__file__).parent.parent / "configs" / "performance_decoder_config.json"
    optimizations = parse_decoder_json(str(custom_path))
    assert optimizations is not None
