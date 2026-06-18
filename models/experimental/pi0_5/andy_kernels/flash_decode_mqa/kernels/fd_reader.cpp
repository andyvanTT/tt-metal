// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_decode_mqa reader (NCRISC). M1: input CB is backed by a sharded tensor
// (data already resident in L1), so we just make the tiles available to compute.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    cb_reserve_back(cb_in, num_tiles);
    cb_push_back(cb_in, num_tiles);
}
