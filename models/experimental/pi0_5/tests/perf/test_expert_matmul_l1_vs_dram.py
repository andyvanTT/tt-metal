# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Action-expert denoise (Gemma-300M, S=32, W=1024) matmul L1-vs-DRAM bench.

Replicates the 5 distinct large matmuls each adaRMS expert layer runs in the
denoise loop (shapes from a real tracy ops_perf_results profile) and times each
with EVERY operand+output on DRAM vs EVERY operand+output on L1.

These are the smallest of the three block types (M=32, narrow width) and the
most likely to fit entirely in L1 — the interesting question here is how much
L1 residency buys over DRAM at this size.

Run:
    PI0_MATMUL_BENCH_EXPERT=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_expert_matmul_l1_vs_dram.py

Under tracy (device-kernel CSVs in generated/profiler/.logs/):
    PI0_MATMUL_BENCH_EXPERT=1 python -m tracy -p -r -v --op-support-count 100000 \\
      -m pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_expert_matmul_l1_vs_dram.py
"""

from __future__ import annotations

import os

import pytest

from models.experimental.pi0_5.tests.perf.matmul_bench_common import MatmulCfg, run_stage


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_MATMUL_BENCH_EXPERT") != "1",
    reason="set PI0_MATMUL_BENCH_EXPERT=1 to run the expert matmul bench",
)

STAGE = "EXPERT DENOISE  Gemma-300M  S=32 W=1024"

# Shapes from ops_perf_results_pi05_matmuls_l1_vs_dram.xlsx (stage 3). bf8
# weights, bf16/bf8 activations, output bf16, LoFi fidelity — matching the profile.
CONFIGS = [
    MatmulCfg("qkv", 32, 1024, 2560, "bf16", "bf8"),
    MatmulCfg("o_proj", 32, 2048, 1024, "bf8", "bf8"),
    MatmulCfg("gate", 32, 1024, 4096, "bf16", "bf8", "gelu"),
    MatmulCfg("up", 32, 1024, 4096, "bf16", "bf8"),
    MatmulCfg("down", 32, 4096, 1024, "bf16", "bf8"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_expert_matmul_l1_vs_dram(device):
    run_stage(device, STAGE, CONFIGS)
