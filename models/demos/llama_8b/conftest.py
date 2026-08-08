# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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
