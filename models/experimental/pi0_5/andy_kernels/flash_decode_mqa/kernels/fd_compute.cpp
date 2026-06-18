// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// flash_decode_mqa compute (TRISC). M1: pure tile copy cb_in(c_0) -> cb_out(c_16),
// to validate the andy_kernels generic_op packaging + CB/compile path before the
// attention math (QK^T -> mask -> softmax -> AV, then Sk-split) is added.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    experimental::CircularBuffer buff_in(tt::CBIndex::c_0);
    experimental::CircularBuffer buff_out(tt::CBIndex::c_16);
    const uint32_t in_id = tt::CBIndex::c_0;
    const uint32_t out_id = tt::CBIndex::c_16;

    init_sfpu(in_id, out_id);
    buff_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        buff_in.wait_front(1);
        copy_tile(in_id, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id);
        buff_in.pop_front(1);
        tile_regs_release();
    }
    buff_out.push_back(num_tiles);
}
