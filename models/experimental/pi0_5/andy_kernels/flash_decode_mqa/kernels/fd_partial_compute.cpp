// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M3a partial-attention (one K-slice, one core). Emits the flash partials needed
// for a cross-core online-softmax combine:
//   m_c = max_n reduce_row(S[n] + mask)       -> cb_mout (1 tile)
//   e[n] = exp(S[n]+mask - m_c)
//   l_c = sum_n reduce_row(e[n])              -> cb_lout (1 tile)
//   O_c = sum_n e[n] @ V[n]   (UN-normalized)  -> cb_out  (vt tiles)
// Combine across slices: M=max(m_c); O=sum O_c*exp(m_c-M); L=sum l_c*exp(m_c-M); out=O/L.
// Q pre-scaled by 1/sqrt(d). bf16 K/V (per-core slice fits L1).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_scaler = 3;
    constexpr uint32_t cb_mask = 4;
    constexpr uint32_t cb_out = 16;   // O_c (un-normalized), vt tiles
    constexpr uint32_t cb_mout = 17;  // m_c, 1 tile
    constexpr uint32_t cb_lout = 18;  // l_c, 1 tile
    constexpr uint32_t cb_qk = 24;
    constexpr uint32_t cb_max = 25;
    constexpr uint32_t cb_exp = 26;
    constexpr uint32_t cb_sum = 27;
    constexpr uint32_t cb_qkm = 30;

    constexpr uint32_t dt = get_compile_time_arg_val(0);
    constexpr uint32_t vt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);

    binary_op_init_common(cb_q, cb_k, cb_out);
    cb_wait_front(cb_q, dt);
    cb_wait_front(cb_k, Skt * dt);
    cb_wait_front(cb_v, Skt * vt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_mask, Skt);

    // S[n] = Q @ K[n]^T
    mm_init(cb_q, cb_k, cb_qk, 1);
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

    // S += mask -> cb_qkm
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

    // m_c = row max across Skt -> cb_max and cb_mout
    cb_wait_front(cb_qkm, Skt);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qkm, cb_scaler, cb_max);
    cb_reserve_back(cb_max, 1);
    cb_reserve_back(cb_mout, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Skt; ++n) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qkm, cb_scaler, n, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    pack_tile(0, cb_mout);
    tile_regs_release();
    cb_push_back(cb_max, 1);
    cb_push_back(cb_mout, 1);
    reduce_uninit();

    // e[n] = exp(S[n] - m_c) -> cb_exp
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

    // l_c = sum_n row-sum(e) -> cb_sum and cb_lout
    cb_wait_front(cb_exp, Skt);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, cb_sum);
    cb_reserve_back(cb_sum, 1);
    cb_reserve_back(cb_lout, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Skt; ++n) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exp, cb_scaler, n, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    pack_tile(0, cb_lout);
    tile_regs_release();
    cb_push_back(cb_sum, 1);
    cb_push_back(cb_lout, 1);
    reduce_uninit();

    // O_c = sum_n e[n] @ V[n]  (un-normalized) -> cb_out
    mm_init(cb_exp, cb_v, cb_out, 0);
    cb_reserve_back(cb_out, vt);
    tile_regs_acquire();
    for (uint32_t dv = 0; dv < vt; ++dv) {
        for (uint32_t n = 0; n < Skt; ++n) {
            matmul_tiles(cb_exp, cb_v, n, n * vt + dv, dv);
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
