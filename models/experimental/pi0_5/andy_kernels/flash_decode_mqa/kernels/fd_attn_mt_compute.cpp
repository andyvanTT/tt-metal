// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2e compute: single-core full-row attention over Skt K-tiles (real Sk=1056=33),
// with additive mask. Non-flash (whole score row resident): keeps all Skt score
// tiles, reduces max/sum across them.
//   S[n] = Q @ K[n]^T  (n in 0..Skt-1, contract dt)     -> cb_qk[Skt]
//   S += mask                                            (cb_mask[Skt])
//   m  = max_n reduce_row(S[n]); e[n] = exp(S[n]-m); l = sum_n reduce_row(e[n])
//   P[n] = e[n] * (1/l)
//   O[:,dv] = sum_n P[n] @ V[n,dv]  (contract Skt)       -> cb_out[vt]
// Q is pre-scaled by 1/sqrt(d) on the host. This is the per-core kernel M3 K-splits.

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
    constexpr uint32_t cb_mask = 4;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_qk = 24;
    constexpr uint32_t cb_max = 25;
    constexpr uint32_t cb_exp = 26;
    constexpr uint32_t cb_sum = 27;
    constexpr uint32_t cb_recip = 28;
    constexpr uint32_t cb_p = 29;
    constexpr uint32_t cb_qkm = 30;  // masked scores (S + mask)

    constexpr uint32_t dt = get_compile_time_arg_val(0);   // QK contraction tiles (head_dim/32)
    constexpr uint32_t vt = get_compile_time_arg_val(1);   // PV output tiles (head_dim_v/32)
    constexpr uint32_t Skt = get_compile_time_arg_val(2);  // K sequence tiles

    binary_op_init_common(cb_q, cb_k, cb_out);
    cb_wait_front(cb_q, dt);
    cb_wait_front(cb_k, Skt * dt);
    cb_wait_front(cb_v, Skt * vt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_mask, Skt);

    // ---- Phase A: S[n] = Q @ K[n]^T -> cb_qk[Skt] ----
    mm_init(cb_q, cb_k, cb_qk, 1 /*transpose K*/);
    cb_reserve_back(cb_qk, Skt);
    for (uint32_t n = 0; n < Skt; ++n) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < dt; ++d) {
            matmul_tiles(cb_q, cb_k, d, n * dt + d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_qk);
        tile_regs_release();
    }
    cb_push_back(cb_qk, Skt);

    // ---- Phase A2: cb_qkm = cb_qk + mask ----
    cb_wait_front(cb_qk, Skt);
    cb_reserve_back(cb_qkm, Skt);
    add_tiles_init(cb_qk, cb_mask);
    for (uint32_t n = 0; n < Skt; ++n) {
        tile_regs_acquire();
        add_tiles(cb_qk, cb_mask, n, n, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_qkm);
        tile_regs_release();
    }
    cb_push_back(cb_qkm, Skt);

    // ---- Phase B: softmax across Skt tiles -> cb_p[Skt] ----
    // row max across all Skt tiles (reduce accumulates into dst[0])
    cb_wait_front(cb_qkm, Skt);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qkm, cb_scaler, cb_max);
    cb_reserve_back(cb_max, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Skt; ++n) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qkm, cb_scaler, n, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    cb_push_back(cb_max, 1);
    reduce_uninit();

    // e[n] = exp(S[n] - max) -> cb_exp[Skt]
    cb_wait_front(cb_max, 1);
    cb_reserve_back(cb_exp, Skt);
    sub_bcast_cols_init_short(cb_qkm, cb_max);
    exp_tile_init<true>();
    for (uint32_t n = 0; n < Skt; ++n) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_qkm, cb_max, n, 0, 0);
        exp_tile<true>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_exp);
        tile_regs_release();
    }
    cb_push_back(cb_exp, Skt);

    // l = sum across Skt tiles
    cb_wait_front(cb_exp, Skt);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, cb_sum);
    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Skt; ++n) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, n, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    tile_regs_release();
    cb_push_back(cb_sum, 1);
    reduce_uninit();

    // recip(l)
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

    // P[n] = e[n] * recip(l)
    cb_wait_front(cb_recip, 1);
    cb_reserve_back(cb_p, Skt);
    mul_bcast_cols_init_short(cb_exp, cb_recip);
    for (uint32_t n = 0; n < Skt; ++n) {
        tile_regs_acquire();
        mul_tiles_bcast<BroadcastType::COL>(cb_exp, cb_recip, n, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_p);
        tile_regs_release();
    }
    cb_push_back(cb_p, Skt);

    // ---- Phase C: O[:,dv] = sum_n P[n] @ V[n,dv] -> cb_out[vt] ----
    cb_wait_front(cb_p, Skt);
    mm_init(cb_p, cb_v, cb_out, 0 /*no transpose*/);
    cb_reserve_back(cb_out, vt);
    tile_regs_acquire();
    for (uint32_t dv = 0; dv < vt; ++dv) {
        for (uint32_t n = 0; n < Skt; ++n) {
            matmul_tiles(cb_p, cb_v, n, n * vt + dv, dv);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t dv = 0; dv < vt; ++dv) {
        pack_tile(dv, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, vt);
}
