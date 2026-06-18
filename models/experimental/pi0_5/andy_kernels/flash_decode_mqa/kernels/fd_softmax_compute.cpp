// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2b compute: row-softmax of a single (32x32) score tile.
//   max = reduce_row_max(S); e = exp(S - max); s = reduce_row_sum(e); P = e * (1/s)
// Isolates the reduce/bcast/exp/recip mode transitions before fusing with QK^T/PV.
// scaler CB holds a 1.0 tile (reduce multiplier).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"  // binary_op_init_common
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"                      // exp_tile
#include "api/compute/eltwise_unary/eltwise_unary.h"  // recip_tile
#include "api/compute/tile_move_copy.h"               // copy_tile
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_s = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_max = 24;
    constexpr uint32_t cb_exp = 25;
    constexpr uint32_t cb_sum = 26;
    constexpr uint32_t cb_recip = 27;

    // Base HW datapath startup (unpack/math/pack), else pack writes zeros.
    binary_op_init_common(cb_s, cb_scaler, cb_out);

    cb_wait_front(cb_s, 1);
    cb_wait_front(cb_scaler, 1);

    // 1) row max -> cb_max
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, cb_max);
    cb_reserve_back(cb_max, 1);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    cb_push_back(cb_max, 1);
    reduce_uninit();

    // 2) e = exp(S - max) -> cb_exp
    cb_wait_front(cb_max, 1);
    cb_reserve_back(cb_exp, 1);
    sub_bcast_cols_init_short(cb_s, cb_max);
    exp_tile_init<true>();
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_s, cb_max, 0, 0, 0);
    exp_tile<true>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_exp);
    tile_regs_release();
    cb_push_back(cb_exp, 1);

    // 3) row sum -> cb_sum
    cb_wait_front(cb_exp, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, cb_sum);
    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    tile_regs_release();
    cb_push_back(cb_sum, 1);
    reduce_uninit();

    // 4) recip(sum) -> cb_recip
    cb_wait_front(cb_sum, 1);
    cb_reserve_back(cb_recip, 1);
    copy_tile_init(cb_sum);
    tile_regs_acquire();
    copy_tile(cb_sum, 0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_recip);
    tile_regs_release();
    cb_push_back(cb_recip, 1);

    // 5) P = e * (1/sum), broadcast recip across columns -> cb_out
    cb_wait_front(cb_recip, 1);
    cb_reserve_back(cb_out, 1);
    mul_bcast_cols_init_short(cb_exp, cb_recip);
    tile_regs_acquire();
    mul_tiles_bcast<BroadcastType::COL>(cb_exp, cb_recip, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
