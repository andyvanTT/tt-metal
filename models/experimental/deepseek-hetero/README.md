# deepseek-hetero — heterogeneous prefill/decode (NVIDIA prefill → Tenstorrent decode)

Disaggregated inference demo for Qwen: **prefill on an NVIDIA GPU** (or CPU for validation),
**decode on Tenstorrent (Blackhole p150a)**. The only thing that crosses the boundary is the
per-layer **KV cache** (post-RoPE K, raw V) + `current_pos` + the first token; decode then runs
fully on-device with no per-token cross-device traffic.

> Status: **work in progress.** Being built incrementally per the design doc.

## Why this split
Prefill is compute-bound (fits GPU FLOPS → low TTFT). Decode is memory-bandwidth + latency bound
and is the long pole per output token (fits Tenstorrent, frees the GPU). Steady-state decode reuses
the optimized, **traced** `tt_transformers` path (~95 tok/s/user on a single p150) unchanged.

## Layout
```
deepseek-hetero/
├── conftest.py            # pytest sys.path bootstrap (hyphen dir → by-path imports)
├── prefill_hf.py          # HF Qwen prefill (cpu/cuda) → per-layer (K,V), first token, prompt_len
├── transport/             # swappable KV transport (one param: --transport pcie|bluefield)
│   ├── base.py            #   KVTransport interface + make_transport() factory
│   ├── pcie.py            #   v1: in-process host tensors / PCIe DMA
│   └── bluefield.py       #   future: RDMA/Ethernet → TT host buffer (stub)
├── bridge.py              # on-device dtype cast + ttnn.tilize into cache layout
├── inject.py              # ttnn.fill_cache per layer; set current_pos
├── decode_driver.py       # build model+cache (create_tt_model), traced no-prefill decode loop
├── demo/run_hetero.py     # entry point; wires prefill→transport→bridge→inject→decode
└── tests/                 # T1–T10 (see design doc)
```

## Design decisions (locked)
- Directory name is the **literal hyphen** form → not importable as a dotted package. Internal
  modules import each other as **top-level names** with this dir on `sys.path` (via `conftest.py`
  for tests; a self-insert at the top of `demo/run_hetero.py` for CLI runs). `tt_transformers` is
  imported normally.
- **Transport is swappable behind one parameter.** PCIe/host-DMA is v1; BlueField/Ethernet (RDMA)
  is the future impl. The `--transport` factory call in `run_hetero.py` is the single swap point.
- **Bridge runs on-device** (`ttnn.tilize`/`typecast`) so the transport carries plain bf16 and a
  future BlueField DPU never has to do block-float quantization.

## Running (once implemented)
```bash
cd /home/andy/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=/home/andy/tt-metal
python models/experimental/deepseek-hetero/demo/run_hetero.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Once upon a time" \
    --prefill-device cpu --transport pcie
```
Tests: `pytest models/experimental/deepseek-hetero/tests/`.

See the full design doc for the data flow, format delta, and T1–T10 test matrix.
