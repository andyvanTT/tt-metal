# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point: heterogeneous prefill (NVIDIA/CPU) → decode (Tenstorrent).

Wires the pipeline:  transport.deliver → bridge → inject → traced decode.

Run (from the tt-metal root, TT venv active, TT_METAL_HOME set):

    python models/experimental/deepseek-hetero/demo/run_hetero.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Once upon a time" \
        --prefill-device cpu --transport pcie --max-tokens 32

The ``--transport`` value selects the KV handoff mechanism at ONE place
(``make_transport`` below): ``pcie`` (v1, in-process host tensors), ``bluefield``
(AF_PACKET frames -> eth_data_rx ERISC -> DRAM staging), or ``ttlink`` (TT-link
packet mode via the ttlink package: GPU -> BF3 -> P150 L1, host-drain sink).
(future, RDMA/Ethernet). Nothing else in the pipeline changes when it is swapped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Hyphenated package dir → put ourselves on sys.path so sibling modules import by name.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import ttnn
from loguru import logger

import bridge
import decode_driver
import inject
from transport.base import make_transport


def parse_args():
    p = argparse.ArgumentParser(description="Heterogeneous prefill (GPU/CPU) -> decode (Tenstorrent)")
    p.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--prefill-device", default="cpu", choices=["cpu", "cuda"], help="where HF prefill runs")
    p.add_argument("--transport", default="pcie", choices=["pcie", "bluefield", "ttlink"], help="KV handoff mechanism")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--instruct", action="store_true", help="apply the chat template to the prompt")
    p.add_argument("--no-trace", action="store_true", help="disable Metal Trace on decode (slower)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_MODEL", args.model)
    logger.info(f"HETERO run: model={args.model} prefill={args.prefill_device} transport={args.transport}")

    # 1. --- TRANSPORT SWAP POINT ------------------------------------------------------
    # Prefill + KV handoff run FIRST, before open_mesh_device(). tt-metal's device init
    # resets the ethernet cores (tearing down the eth_data_rx ERISC RX ring), so a wire
    # transport that receives into the P150 must complete its transfer before the mesh
    # device is opened. deliver() reads the KV back into HOST tensors (res.kv), so the
    # subsequent device open can freely reinitialize the chip. Uses a host-only ModelArgs
    # (mesh_device=None) — prefill only needs config/tokenizer/reference weights, no device.
    # Downstream (bridge -> inject -> decode) is identical regardless of transport.
    from models.tt_transformers.tt.model_config import ModelArgs

    host_model_args = ModelArgs(
        None,  # no mesh device: host/CPU reference path for prefill
        instruct=args.instruct,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
    )
    transport = make_transport(
        args.transport,
        model_args=host_model_args,
        device=args.prefill_device,
        instruct=args.instruct,
        state_dict=None,
    )
    # ----------------------------------------------------------------------------------

    t0 = time.time()
    res = transport.deliver(args.prompt)
    t_prefill = time.time() - t0
    logger.info(f"Prefill+transport: {t_prefill*1000:.0f} ms (prompt_len={res.prompt_len})")

    # Release the transport's own device handle (if any) BEFORE opening the mesh device,
    # so tt-metal and the transport don't hold two UMD contexts on the same chip at once.
    close = getattr(transport, "close", None)
    if callable(close):
        close()

    # 2. Now open the device and build the Tenstorrent decode model + (non-paged) KV cache.
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        tt = decode_driver.build_tt_model(
            mesh_device, max_seq_len=args.max_seq_len, max_batch_size=1, instruct=args.instruct
        )

        # 4. Bridge host KV -> device bf16 TILE (on-device tilize).
        t0 = time.time()
        kv_dev = bridge.to_device_kv(res.kv, mesh_device)
        # 5. Inject into the decode KV cache (on-device typecast to bfp8 + fill_cache).
        inject.inject_kv(tt.model, kv_dev, batch_idx=0)
        t_bridge = time.time() - t0
        logger.info(f"Bridge+inject: {t_bridge*1000:.0f} ms")

        # 6. Traced decode on Tenstorrent.
        t0 = time.time()
        gen = decode_driver.run_decode(
            tt, res.first_token, res.prompt_len, max_tokens=args.max_tokens, enable_trace=not args.no_trace
        )
        t_decode = time.time() - t0
        n = max(len(gen), 1)

        text = tt.model_args.tokenizer.decode(res.prompt_tokens + gen)
        print("\n================ HETERO OUTPUT ================")
        print(text)
        print("==============================================")
        print(
            f"prefill+transport={t_prefill*1000:.0f}ms | bridge+inject={t_bridge*1000:.0f}ms | "
            f"decode={t_decode*1000:.0f}ms ({n} tok, {n/t_decode:.1f} tok/s incl. compile)"
        )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
