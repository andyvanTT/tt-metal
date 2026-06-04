# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP vision block (S=256, H=1152) large-matmul L1-vs-DRAM micro-bench.

Replicates the 4 distinct large matmuls each SigLIP encoder layer runs (shapes
from a real tracy ops_perf_results profile) and times each with EVERY
operand+output on DRAM vs EVERY operand+output on L1.

Note QKV and MLP fc1 share the same (256x1152)@(1152x4608) shape after head
padding — fc1 carries a fused GELU, QKV does not. fc2 (256x4608 @ 4608x1152) is
the slowest SigLIP matmul in the source profile.

Run:
    PI0_MATMUL_BENCH_SIGLIP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_siglip_matmul_l1_vs_dram.py

Under tracy (device-kernel CSVs in generated/profiler/.logs/):
    PI0_MATMUL_BENCH_SIGLIP=1 python -m tracy -p -r -v --op-support-count 100000 \\
      -m pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_siglip_matmul_l1_vs_dram.py
"""

from __future__ import annotations

import os

import pytest

from models.experimental.pi0_5.tests.perf.matmul_bench_common import MatmulCfg, run_stage


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_MATMUL_BENCH_SIGLIP") != "1",
    reason="set PI0_MATMUL_BENCH_SIGLIP=1 to run the siglip matmul bench",
)

STAGE = "SIGLIP vision  S=256 H=1152"

# Shapes from ops_perf_results_pi05_matmuls_l1_vs_dram.xlsx (stage 1). bf8
# weights, bf16/bf8 activations, output bf16, LoFi fidelity — matching the profile.
CONFIGS = [
    MatmulCfg("qkv", 256, 1152, 4608, "bf16", "bf8"),
    MatmulCfg("mlp_fc1", 256, 1152, 4608, "bf16", "bf8", "gelu"),
    MatmulCfg("o_proj", 256, 1536, 1152, "bf8", "bf8"),
    MatmulCfg("mlp_fc2", 256, 4608, 1152, "bf8", "bf8"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_siglip_matmul_l1_vs_dram(device):
    run_stage(device, STAGE, CONFIGS)
