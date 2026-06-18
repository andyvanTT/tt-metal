// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2a reader: Q and K CBs are backed by sharded tensors (resident in L1); make
// both available to compute. dt = contraction tiles (head_dim / 32) per row.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t dt = get_compile_time_arg_val(2);
    cb_reserve_back(cb_q, dt);
    cb_push_back(cb_q, dt);
    cb_reserve_back(cb_k, dt);
    cb_push_back(cb_k, dt);
}
