// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2a compute: S = Q @ K^T for a single (Sq=1 tile, Sk=1 tile) block, contracting
// over dt head-dim tiles. transpose=true on B (=K) gives Q·K^T. Accumulates the
// dt partial products into DST[0]. Validates matmul+transpose in our kernel before
// softmax/PV (M2b/M2c) and Sk-split (M3) are layered on.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t dt = get_compile_time_arg_val(3);

    mm_init(cb_q, cb_k, cb_out, 1 /*transpose B = K*/);
    cb_wait_front(cb_q, dt);
    cb_wait_front(cb_k, dt);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    for (uint32_t d = 0; d < dt; ++d) {
        matmul_tiles(cb_q, cb_k, d, d, 0);  // DST[0] += Q_d @ K_d^T
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_q, dt);
    cb_pop_front(cb_k, dt);
}
