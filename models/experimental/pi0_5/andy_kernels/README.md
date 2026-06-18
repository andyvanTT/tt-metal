# andy_kernels

Custom TTNN device kernels for pi0.5, packaged in the deepseek_v3_b1 "blitz"
micro-op style: each op is a self-contained directory with an `op.py`
(`@staticmethod golden()` PyTorch reference + `@staticmethod op()` that builds a
`ttnn.ProgramDescriptor` of hand-written `.cpp` kernels and dispatches via
`ttnn.generic_op`). Kernels live in a per-op `kernels/` subdir, referenced by
project-relative path.

## Ops

- `flash_decode_mqa/` — K-parallel flash attention for the pi0.5 denoise step:
  small query (Q=32 padded, 8 heads), MQA (1 KV head), long shared KV (~1056),
  non-causal with an additive phantom mask. Splits the KV sequence across cores
  (flash-decoding) with online-softmax cross-core reduction, keeping K/V resident
  in L1 (no batch replication). Replaces the prefill-SDPA op that pins this shape
  to ~8 cores. See `models/experimental/pi0_5/tests/perf/_sdpa_decode_qfold_repro.py`
  for the validation harness and the prefill-SDPA / torch references it must match.

## Convention

```
<op_name>/
├── op.py            # class <Op>: @staticmethod golden(...) ; @staticmethod op(...)
└── kernels/
    ├── <op>_reader.cpp     # NCRISC dataflow (in)
    ├── <op>_compute.cpp    # TRISC compute
    └── <op>_writer.cpp     # BRISC dataflow (out)
```

Imported directly: `from models.experimental.pi0_5.andy_kernels.flash_decode_mqa.op import FlashDecodeMQA`.
