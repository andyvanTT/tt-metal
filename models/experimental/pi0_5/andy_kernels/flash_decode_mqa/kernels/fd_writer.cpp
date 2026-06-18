// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_decode_mqa writer (BRISC). M1: output CB is backed by the output sharded
// tensor; compute packs directly into it, so the writer only waits for completion.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    cb_wait_front(cb_out, num_tiles);
}
