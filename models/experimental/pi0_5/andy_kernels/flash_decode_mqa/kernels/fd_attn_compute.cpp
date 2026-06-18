// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2d compute: full single-block attention on one core, no flash (Sk = 1 tile):
//   S = Q @ K^T   (matmul, transpose B)
//   P = softmax_row(S)              (reduce-max, exp, reduce-sum, recip, bcast-mul)
//   O = P @ V     (matmul, no transpose)  -> vt output tiles
// Scale is folded into Q on the host (caller pre-scales Q), so softmax is plain.
// dt = head_dim/32 (QK contraction), vt = head_dim_v/32 (PV output tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_scaler = 3;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_qk = 24;
    constexpr uint32_t cb_max = 25;
    constexpr uint32_t cb_exp = 26;
    constexpr uint32_t cb_sum = 27;
    constexpr uint32_t cb_recip = 28;
    constexpr uint32_t cb_p = 29;

    constexpr uint32_t dt = get_compile_time_arg_val(0);  // QK contraction tiles (head_dim/32)
    constexpr uint32_t vt = get_compile_time_arg_val(1);  // PV output tiles (head_dim_v/32)

    binary_op_init_common(cb_q, cb_k, cb_out);
    cb_wait_front(cb_q, dt);
    cb_wait_front(cb_k, dt);
    cb_wait_front(cb_v, vt);
    cb_wait_front(cb_scaler, 1);

    // ---- Phase A: S = Q @ K^T -> cb_qk (1 tile) ----
    mm_init(cb_q, cb_k, cb_qk, 1 /*transpose K*/);
    cb_reserve_back(cb_qk, 1);
    tile_regs_acquire();
    for (uint32_t d = 0; d < dt; ++d) {
        matmul_tiles(cb_q, cb_k, d, d, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_qk);
    tile_regs_release();
    cb_push_back(cb_qk, 1);

    // ---- Phase B: P = softmax_row(cb_qk) -> cb_p (1 tile) ----
    cb_wait_front(cb_qk, 1);
    // row max
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qk, cb_scaler, cb_max);
    cb_reserve_back(cb_max, 1);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qk, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    cb_push_back(cb_max, 1);
    reduce_uninit();
    // e = exp(S - max)
    cb_wait_front(cb_max, 1);
    cb_reserve_back(cb_exp, 1);
    sub_bcast_cols_init_short(cb_qk, cb_max);
    exp_tile_init<true>();
    tile_regs_acquire();
    sub_tiles_bcast_cols(cb_qk, cb_max, 0, 0, 0);
    exp_tile<true>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_exp);
    tile_regs_release();
    cb_push_back(cb_exp, 1);
    // sum
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
    // recip
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
    // P = e * recip(sum)
    cb_wait_front(cb_recip, 1);
    cb_reserve_back(cb_p, 1);
    mul_bcast_cols_init_short(cb_exp, cb_recip);
    tile_regs_acquire();
    mul_tiles_bcast<BroadcastType::COL>(cb_exp, cb_recip, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_p);
    tile_regs_release();
    cb_push_back(cb_p, 1);

    // ---- Phase C: O = P @ V -> cb_out (vt tiles) ----
    cb_wait_front(cb_p, 1);
    mm_init(cb_p, cb_v, cb_out, 0 /*no transpose*/);
    cb_reserve_back(cb_out, vt);
    tile_regs_acquire();
    for (uint32_t d = 0; d < vt; ++d) {
        matmul_tiles(cb_p, cb_v, 0, d, d);  // O_d = P @ V[:, d]
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < vt; ++d) {
        pack_tile(d, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, vt);
}
