#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot runner for the Llama-3.1-8B demo tuning fork.
#
# Defaults to a single Blackhole P150 chip and the local Meta-Llama-3.1-8B
# weights (no HF token needed). Override knobs via the flags/env vars below.
#
# Usage:
#   ./run_demo.sh                              # single P150 + local weights
#   MESH_DEVICE=P150x4 ./run_demo.sh           # 1x4 BH mesh instead
#   ./run_demo.sh --max_generated_tokens 64    # extra flags forwarded to pytest
#   ./run_demo.sh --unit-only                  # skip the device demo, run unit tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/python_env}"

# --- Model / weights -------------------------------------------------------
# Hardcoded to the Meta-Llama-3.1-8B weights bundled next to this script. An
# absolute path means the HF hub is never contacted, so no HF token is needed.
# See "Model weights" in README.md for how to download them.
HF_MODEL="${SCRIPT_DIR}/NousResearch/Meta-Llama-3.1-8B"
export HF_MODEL

# --- Device mesh -----------------------------------------------------------
# Default to a single P150 chip. The fork's conftest.py flips fabric_config
# off and bumps the trace region when MESH_DEVICE is P150/N150.
export MESH_DEVICE="${MESH_DEVICE:-P150}"

# Weight cache location. Default matches ModelArgs: model_cache/<HF_MODEL>/<device>.
export TT_CACHE_PATH="${TT_CACHE_PATH:-${REPO_ROOT}/model_cache}"

# --- Activate the project venv --------------------------------------------
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}" >&2
    echo "Create it with: ${REPO_ROOT}/create_venv.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_ROOT}"

# --- Parse our own flags vs pytest flags -----------------------------------
PYTEST_ARGS=()
RUN_UNIT=0
RUN_DEMO=1
for arg in "$@"; do
    case "${arg}" in
        --unit-only)   RUN_UNIT=1; RUN_DEMO=0 ;;
        --unit)        RUN_UNIT=1 ;;
        --no-demo)    RUN_DEMO=0 ;;
        --help|-h)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)            PYTEST_ARGS+=("${arg}") ;;
    esac
done

DEMO_TEST="models/demos/llama_8b/demo/text_demo.py"
DEMO_SELECTOR="Llama-3.1-8B-Instruct"
UNIT_TEST="models/demos/llama_8b/tests/test_fork_equivalence.py"

# Defaults passed to pytest (overridable via PYTEST_DEFAULTS env or extra args).
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-32}"

run_unit() {
    echo "==> Running unit tests (no device required): ${UNIT_TEST}"
    pytest "${UNIT_TEST}" -v "${PYTEST_ARGS[@]}"
}

run_demo() {
    echo "==> Running Llama-8B demo (HF_MODEL=${HF_MODEL})"
    pytest "${DEMO_TEST}" -k "${DEMO_SELECTOR}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --max_generated_tokens "${MAX_GEN_TOKENS}" \
        "${PYTEST_ARGS[@]}"
}

rc=0
if [[ "${RUN_UNIT}" -eq 1 ]]; then run_unit || rc=$?; fi
if [[ "${RUN_DEMO}" -eq 1 ]]; then run_demo || rc=$?; fi
exit "${rc}"
