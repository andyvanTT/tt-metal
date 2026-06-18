// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2b reader: scores + scaler CBs are sharded-tensor-backed; make both available.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    cb_reserve_back(cb_s, 1);
    cb_push_back(cb_s, 1);
    cb_reserve_back(cb_scaler, 1);
    cb_push_back(cb_scaler, 1);
}
