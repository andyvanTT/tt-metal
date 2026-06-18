// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M2e reader: q, k, v, scaler, mask CBs are sharded-tensor-backed; make available.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(4);
    constexpr uint32_t dt = get_compile_time_arg_val(5);
    constexpr uint32_t vt = get_compile_time_arg_val(6);
    constexpr uint32_t Skt = get_compile_time_arg_val(7);
    cb_reserve_back(cb_q, dt);
    cb_push_back(cb_q, dt);
    cb_reserve_back(cb_k, Skt * dt);
    cb_push_back(cb_k, Skt * dt);
    cb_reserve_back(cb_v, Skt * vt);
    cb_push_back(cb_v, Skt * vt);
    cb_reserve_back(cb_scaler, 1);
    cb_push_back(cb_scaler, 1);
    cb_reserve_back(cb_mask, Skt);
    cb_push_back(cb_mask, Skt);
}
