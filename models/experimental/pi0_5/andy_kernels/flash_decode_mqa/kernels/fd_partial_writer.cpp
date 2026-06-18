// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M3a writer: O_c / m_c / l_c output CBs are sharded-tensor-backed; wait for all.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t vt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mout = get_compile_time_arg_val(2);
    constexpr uint32_t cb_lout = get_compile_time_arg_val(3);
    cb_wait_front(cb_out, vt);
    cb_wait_front(cb_mout, 1);
    cb_wait_front(cb_lout, 1);
}
