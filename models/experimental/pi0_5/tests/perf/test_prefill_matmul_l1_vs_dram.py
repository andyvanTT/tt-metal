# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VLM-prefill (Gemma-2B, S=512, W=2048) large-matmul L1-vs-DRAM micro-bench.

Replicates the 5 distinct large matmuls the Gemma-2B prefix-prefill path runs
per layer (shapes from a real tracy ops_perf_results profile) and times each
with EVERY operand+output on DRAM vs EVERY operand+output on L1.

The gate/up/down matmuls (N or K = 16384) are the largest in the whole pi0.5
model — `down` is both the max-size (17.2 GFLOP) and max-time matmul, and is the
prime candidate for "too big to fit in L1".

Run:
    PI0_MATMUL_BENCH_PREFILL=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_prefill_matmul_l1_vs_dram.py

Under tracy (device-kernel CSVs in generated/profiler/.logs/):
    PI0_MATMUL_BENCH_PREFILL=1 python -m tracy -p -r -v --op-support-count 100000 \\
      -m pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_prefill_matmul_l1_vs_dram.py
"""

from __future__ import annotations

import os

import pytest

from models.experimental.pi0_5.tests.perf.matmul_bench_common import MatmulCfg, run_stage


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_MATMUL_BENCH_PREFILL") != "1",
    reason="set PI0_MATMUL_BENCH_PREFILL=1 to run the prefill matmul bench",
)

STAGE = "VLM PREFILL  Gemma-2B  S=512 W=2048"

# Shapes from ops_perf_results_pi05_matmuls_l1_vs_dram.xlsx (stage 2). bf8
# weights, bf16/bf8 activations, output bf16, LoFi fidelity — matching the profile.
CONFIGS = [
    MatmulCfg("qkv", 512, 2048, 2560, "bf16", "bf8"),
    MatmulCfg("o_proj", 512, 2048, 2048, "bf8", "bf8"),
    MatmulCfg("gate", 512, 2048, 16384, "bf16", "bf8", "gelu"),
    MatmulCfg("up", 512, 2048, 16384, "bf16", "bf8"),
    MatmulCfg("down", 512, 16384, 2048, "bf16", "bf8"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_prefill_matmul_l1_vs_dram(device):
    run_stage(device, STAGE, CONFIGS)
