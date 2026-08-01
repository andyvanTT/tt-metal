# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Offline test for the ttlink transport adapter (no hardware, no package).

Stubs the two heavy dependencies in sys.modules BEFORE importing
transport.ttlink:
  - ``prefill_hf``  : fake run_prefill returning known KV tensors
  - ``ttlink``      : fake Pipeline whose push() returns drained bytes read
                      back from the submitted segments (mirroring the real
                      byte-identity guarantee) plus a passing oracle

Asserts the adapter contract: correct segment layout (per-layer K then V,
bf16), one push with the full byte count, PrefillResult passthrough, KV
byte-identity through the round trip, verify-mismatch raises, close() drops
and rebuilds the pipeline.
"""

import ctypes
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------- fake ttlink
class _FakeRep:
    def __init__(self, frames, windows, drained):
        self.frames = frames
        self.windows = windows
        self.elapsed_s = 0.01
        self.gbps_payload = 1.0
        self.tx_seq_start = 0
        self.tx_seq_end = frames & 0xFF
        self.cqe_err = 0
        self.first_bad_tx_seq = (1 << 64) - 1
        self.drained = drained
        self.verify = types.SimpleNamespace(ok=True, summary=lambda: "ok")


class _FakeSession:
        def bringup(self, check_only=True):
            return types.SimpleNamespace(ok=True, summary=lambda: "ok")

        def open(self):
            return self

        def arm_rxq(self, **kw):
            return 0


class _FakePipeline:
    instances = []

    def __init__(self, **kw):
        self.session = _FakeSession()
        self.pushes = []
        self.closed = False
        self.corrupt = False
        _FakePipeline.instances.append(self)

    def snapshot(self):
        return {}

    def push(self, segs, nbytes, rate_pps=0, source="gpu", drain=True):
        blob = b"".join(ctypes.string_at(p, n) for p, n in segs)
        assert len(blob) == nbytes
        self.pushes.append({"segs": list(segs), "nbytes": nbytes,
                            "source": source})
        if self.corrupt and blob:
            blob = bytes([blob[0] ^ 0xFF]) + blob[1:]
        return _FakeRep(frames=(nbytes + 4063) // 4064, windows=1,
                        drained=blob)

    def close(self):
        self.closed = True


def _install_stubs(monkeypatch, kv):
    @dataclass
    class PrefillResult:
        kv: object
        first_token: torch.Tensor
        prompt_len: int
        prompt_tokens: list = None
        logits_last: torch.Tensor = None

    prefill_hf = types.ModuleType("prefill_hf")
    prefill_hf.PrefillResult = PrefillResult
    prefill_hf.run_prefill = lambda model_args, prompt, device="cpu", instruct=False, state_dict=None: PrefillResult(
        kv=kv,
        first_token=torch.tensor([[42]]),
        prompt_len=7,
    )
    monkeypatch.setitem(sys.modules, "prefill_hf", prefill_hf)

    ttlink = types.ModuleType("ttlink")
    ttlink.Pipeline = _FakePipeline
    ttlink.StageLog = object
    monkeypatch.setitem(sys.modules, "ttlink", ttlink)

    model_config = types.ModuleType("models.tt_transformers.tt.model_config")
    model_config.ModelArgs = object
    monkeypatch.setitem(sys.modules,
                        "models.tt_transformers.tt.model_config", model_config)

    return PrefillResult


def _kv(n_layers=2, seq=5, heads=2, hd=4):
    torch.manual_seed(0)
    return [(torch.randn(1, seq, heads, hd), torch.randn(1, seq, heads, hd))
            for _ in range(n_layers)]


def _expected_blob(kv):
    parts = []
    for k, v in kv:
        for t in (k, v):
            parts.append(t.to(torch.bfloat16).contiguous()
                         .view(torch.uint16).numpy().tobytes())
    return b"".join(parts)


# -------------------------------------------------------------------- tests
def test_deliver_roundtrip(monkeypatch):
    kv = _kv()
    PrefillResult = _install_stubs(monkeypatch, kv)
    _FakePipeline.instances.clear()

    from transport.ttlink import TTLinkTransport

    tr = TTLinkTransport(model_args=None, device="cpu")
    res = tr.deliver("hi")

    assert isinstance(res, PrefillResult)
    assert res.first_token.item() == 42 and res.prompt_len == 7

    pipe = _FakePipeline.instances[0]
    assert len(pipe.pushes) == 1
    segs = pipe.pushes[0]["segs"]
    want = _expected_blob(kv)
    # per-layer K then V, bf16 -> 2 segments per layer
    assert len(segs) == 2 * len(kv)
    per = kv[0][0].numel() * 2
    assert all(n == per for _, n in segs)
    assert pipe.pushes[0]["nbytes"] == len(want)

    # KV byte-identity through the round trip
    got = b"".join(
        t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
        for k, v in res.kv for t in (k, v)
    )
    assert got == want
    tr.close()
    assert pipe.closed


def test_deliver_verify_mismatch_raises(monkeypatch):
    kv = _kv()
    _install_stubs(monkeypatch, kv)
    _FakePipeline.instances.clear()

    from transport.ttlink import TTLinkTransport

    tr = TTLinkTransport(model_args=None, device="cpu", verify=True)
    tr._pipeline().corrupt = True
    with pytest.raises(RuntimeError, match="mismatch"):
        tr.deliver("hi")
    tr.close()


def test_close_rebuilds_pipeline(monkeypatch):
    kv = _kv()
    _install_stubs(monkeypatch, kv)
    _FakePipeline.instances.clear()

    from transport.ttlink import TTLinkTransport

    tr = TTLinkTransport(model_args=None, device="cpu")
    tr.deliver("hi")
    first = _FakePipeline.instances[0]
    tr.close()
    tr.deliver("hi again")
    assert len(_FakePipeline.instances) == 2
    assert _FakePipeline.instances[1] is not first
    tr.close()
