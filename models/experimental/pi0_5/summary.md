# PI0.5 Socket-Split Benchmark — Carryover Summary

> Handoff doc for continuing this work on another agent / server. Captures the goal,
> environment, what was built, how to run it, open risks, and the hardware blocker we hit.

---

## 1. Goal

Benchmark the **direct-write socket device-to-device (d2d) ops**
(`ttnn.experimental.send_direct_async` / `recv_direct_async`) in the context of the real
**PI0.5** model, and produce a **per-op tracy CSV** comparing a socket-split run against the
single-mesh baseline.

PI0.5 has a natural pipeline boundary:

```
   submesh A (PREFIX)                          submesh B (SUFFIX / action expert)
 SigLIP vision + Gemma-2B VLM prefill  ──socket d2d──►  10-step flow-matching denoise loop
   → prefix K/V cache (once/chunk)      (direct write)   reads the prefix K/V every step
```

The prefix K/V cache is computed **once per chunk** and read **read-only** across all ~10
denoise steps → it's the ideal payload to ship A→B once via the socket, after which submesh B
runs the entire denoise loop locally.

**Reference doc (read this):** `models/experimental/pi0_5/sockets.md` — explains the socket API,
the direct-write mechanism (FIFO carries only handshake + completion; payload streams straight to
the receiver's output tensor), and integration notes.

---

## 2. Architecture decision (resolved — do NOT change)

We considered reverting the in-tree socket ops and reimplementing them as out-of-tree custom
kernels inside `pi0_5/`. **Decision: keep the ops as-is.** Rationale:

- The "d2d speedup" commit (`d425ecc5db`) only **bumped the `umd` submodule** to `b4a59998`,
  which is a plain **upstream commit on `tenstorrent/tt-umd` main** (PRs incl. #2771 noc/dma read,
  #2710 SIMD memcpy) — **not a forked driver**. A normal submodule update carries it forward;
  nothing to port. (It also bumped `tracy` and added `sockets.md`.)
- The ops live in a **self-contained, additive** directory
  `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/` (impl commit `7261d72c81`:
  **+1571 / -0** to existing files; only 2 one-line registrations in CMakeLists + nanobind).
  They are **already compiled** into this branch's `_ttnncpp.so` (`ttnn.experimental.send_direct_async`
  imports and resolves). No rebuild needed to use them.
- An out-of-tree ttnn op would have to link `libttnn` and would break on every ttnn rebuild
  (ABI coupling) — **more** fragile, not less. Device kernels are JIT-loaded by path and can stay
  where they are.
- If rebase-conflict avoidance is the real concern, **upstream the op** rather than vendoring it.

So: the only new code is the **benchmark driver** in `pi0_5/` (below).

---

## 3. Environment

- **Repo:** `/home/tt-admin/avan/tt-metal`  ·  **Branch:** `my_flash`
- **Machine:** `g11blx01`, Ubuntu 22.04, **32× `tt-galaxy-bh` (Blackhole)**, KMD 2.8.0, FW bundle 19.9.0.
- **Checkpoint:** `/home/tt-admin/pi05_cache/pi05_libero_upstream/model.safetensors` (~7.2 GB, pi05_libero).
- **Activate the env (every shell):**
  ```bash
  cd /home/tt-admin/avan/tt-metal
  source python_env/bin/activate
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
  ```
- **Required env vars for the pi0.5 e2e tests** (match the established run config):
  ```
  PI0_UPSTREAM_MASKS=1
  QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1
  QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
  PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
  ```
  `PI0_UPSTREAM_MASKS=1` is important: it makes the model build the upstream RoPE/mask artifacts
  the denoise loop consumes — the socket test depends on those being present.

> ⚠️ **Shared cluster.** Opening a device touches the whole cluster; per-chip `CHIP_IN_USE_<n>`
> locks at `/dev/shm/TT_UMD_LOCK.*` serialize use. Default pytest fixtures grab **chip 0**, and
> most pi0.5 jobs start there, so concurrent runs contend on chip 0 (→ segfault/hang). **Always
> confirm idle first:** `ps -eo pid,etime,cmd | grep -iE "pytest|tracy" | grep -v grep`.

---

## 4. What was built / changed (both UNCOMMITTED on branch `my_flash`)

### 4a. Model seam refactor — `tt/ttnn_pi0_5_model.py`
Split the monolithic `Pi0_5ModelTTNN.sample_actions` into two reusable methods so the
prefix/suffix stages can run on different submeshes. **Behavior-preserving** — `sample_actions`
now just calls both in sequence.

- `run_prefix(images, img_masks, lang_tokens, lang_masks)` → returns
  `(prefix_kv_cache, upstream_artifacts, batch_size, _keepalive)`.
  (`_keepalive` holds the pre-`fill_implicit_tile_padding` cache alive for tensor lifetime.)
- `run_denoise(prefix_kv_cache, upstream_artifacts, batch_size, state=None)` → the 10-step
  Euler/flow-matching loop; returns the `[1, action_horizon, action_dim]` actions tensor.
- `sample_actions(...)` = `run_prefix(...)` then `run_denoise(...)`.

**Handoff payload (what crosses A→B):**
- `prefix_kv_cache`: a list of **one `(K, V)` pair per VLM layer** (~18 layers → ~36 tensors),
  each `bf16`, **DRAM**, shape `(1, 1, prefix_padded, 256)`, **read-only across all denoise steps**.
- With `PI0_UPSTREAM_MASKS=1`, the denoise loop also reads 3 tensors from `upstream_artifacts`:
  `suffix_cos`, `suffix_sin`, `expert_attn_mask` (these must also be transferred to B).
  (The `prefix_*` artifact entries are NOT read by `run_denoise`.)

### 4b. Socket benchmark test — `tests/perf/test_perf_ttnn_full_e2e_socket.py` (NEW)
Mirrors `test_perf_ttnn_full_e2e.py` but pipeline-splits across two single-chip submeshes:
1. `@pytest.mark.parametrize("mesh_device", [(1, 2)])` + `device_params` with
   `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING`, `l1_small_size=24576`, `trace_region_size=80M`.
2. `submesh_a = create_submesh((1,1)@(0,0))`, `submesh_b = create_submesh((1,1)@(0,1))`.
3. Builds a **full `Pi0_5ModelTTNN` on each submesh** (`model_a` for prefix, `model_b` for denoise) —
   reuses the class unchanged via the 4a seam (≈2× weight memory, fine on BH).
4. One-time socket setup: `_build_socket_connections` (row-0 senders / row-1 receivers,
   `NUM_CONNECTIONS=2`), `SocketMemoryConfig(L1, SOCKET_PAGE_SIZE*4)`, `create_socket_pair(A, B, cfg)`.
   Pre-allocates landing buffers on B via `allocate_tensor_on_device(src.spec, submesh_b)` for the
   flattened K/V list + the 3 artifact tensors.
5. Per chunk: `model_a.run_prefix(...)` → `send_direct_async`/`recv_direct_async` each tensor A→B →
   `synchronize_device` both → `model_b.run_denoise(received_kv, received_artifacts, ...)`.
6. Verifies a `comp`/`allclose` on the first received K/V and finite/shape on the actions, then
   times `NUM_ITERS` steady-state chunks (`NUM_WARMUP=0, NUM_ITERS=1, LANG_SEQ_LEN=256`).

Helper provenance: the socket helpers are modeled on
`tests/ttnn/distributed/test_socket_perf.py` (the existing async-vs-direct GB/s micro-benchmark).

---

## 5. Status

| Phase | What | Status |
|-------|------|--------|
| 0 | Baseline tracy CSV (single-mesh) | ✅ **DONE** — test passed (136 s), 5276 op rows |
| 1 | `run_prefix`/`run_denoise` refactor | ✅ code written + parses; ⏳ **runtime validation NOT done** (blocked by hardware) |
| 2 | `test_perf_ttnn_full_e2e_socket.py` | ✅ written; ⏳ **never executed** (blocked) |
| 3 | Socket tracy CSV + baseline diff | ⏳ not started (blocked) |

**Baseline CSV (already produced):**
```
/home/tt-admin/avan/tt-metal/generated/pi05_baseline/reports/pi0.5_baseline/2026_06_09_23_27_39/ops_perf_results_pi0.5_baseline_2026_06_09_23_27_39.csv
```
CSV columns include `OP CODE`, `OP TYPE`, `CORE COUNT`, `DEVICE KERNEL DURATION [ns]`,
`OP TO OP LATENCY [ns]`, etc. (defined in `tools/tracy/process_ops_logs.py`).

---

## 6. How to run (absolute, copy-paste)

Run only when the cluster is idle. Order: validate refactor → socket functional → socket tracy.

**(1) Validate the Phase-1 refactor (single-mesh, no profiler):**
```bash
cd /home/tt-admin/avan/tt-metal && source python_env/bin/activate && \
TT_METAL_HOME=/home/tt-admin/avan/tt-metal PYTHONPATH=/home/tt-admin/avan/tt-metal \
PI0_UPSTREAM_MASKS=1 QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \
pytest /home/tt-admin/avan/tt-metal/models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e.py -x -q -s
```

**(2) Socket test — functional first (clean traceback for the open risks):**
```bash
cd /home/tt-admin/avan/tt-metal && source python_env/bin/activate && \
TT_METAL_HOME=/home/tt-admin/avan/tt-metal PYTHONPATH=/home/tt-admin/avan/tt-metal \
PI0_UPSTREAM_MASKS=1 QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \
pytest /home/tt-admin/avan/tt-metal/models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_socket.py -x -q -s
```

**(3) Socket test under tracy → per-op CSV (after #2 passes):**
```bash
cd /home/tt-admin/avan/tt-metal && source python_env/bin/activate && \
TT_METAL_HOME=/home/tt-admin/avan/tt-metal PYTHONPATH=/home/tt-admin/avan/tt-metal \
PI0_UPSTREAM_MASKS=1 QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \
python -m tracy -p -r -v --op-support-count 100000 \
  -o /home/tt-admin/avan/tt-metal/generated/pi05_socket -n pi0.5_socket \
  -m "pytest /home/tt-admin/avan/tt-metal/models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_socket.py"
```
Output CSV (path printed as `OPs csv generated at:`):
```
/home/tt-admin/avan/tt-metal/generated/pi05_socket/reports/pi0.5_socket/<TIMESTAMP>/ops_perf_results_pi0.5_socket_<TIMESTAMP>.csv
# newest:
ls -t /home/tt-admin/avan/tt-metal/generated/pi05_socket/reports/pi0.5_socket/*/ops_perf_results_*.csv | head -1
```
Confirm `send_direct_async` / `recv_direct_async` rows appear, then diff against the baseline CSV.

> The same env vars + checkpoint apply on a different server, but **paths will differ** — update
> the repo path, `PI05_CHECKPOINT_DIR`, and the venv location for the new box.

---

## 7. Open risks to validate (untested — first real run will surface these)

1. **DRAM payload over the direct socket op.** `prefix_kv_cache` is in **DRAM**; the existing
   micro-benchmark (`test_socket_perf.py`) only exercises **L1** tensors. If `send/recv_direct_async`
   requires L1 (or a specific page layout), either stage the K/V to L1 before transfer or fall back
   to the FIFO `send_async`/`recv_async`. The test does a `comp`/`allclose` before timing to catch
   corruption.
2. **Model on a 1×1 submesh.** The model calls `ttnn.from_torch(device=device)` with no
   `mesh_mapper`. Confirm it builds/runs on a single-chip `MeshDevice`; if not, add a replicate
   mapper or adjust construction.
3. **~36 K/V transfers + 3 artifacts per chunk.** Tune `NUM_CONNECTIONS` (1/2/4) if transfer
   dominates; first goal is correctness + a profiled run, not peak BW.
4. **Fallback if the full split is too costly:** run the full model on submesh A and transfer the
   real K/V A→B→discard once per iter purely to profile the socket op cost on the real payload
   (denoise still on A → valid result). Lower fidelity but de-risks the split.

---

## 8. Hardware blocker hit this session (IMPORTANT for the new server too)

We could not complete the device runs because **chip 0 got wedged** and the reset path on this
Galaxy is broken:

- Symptom 1 (wedge): `Device 0 init: failed to initialize FW! ... Timeout waiting for physical
  cores to finish ... Try resetting the board.` — happens in the `device` fixture, before any model
  code. Cause: a process killed mid-use (a looping benchmark + our own `timeout`-killed retries)
  left chip 0 in a bad FW state.
- `tt-smi -r 0` (targeted reset) is **not supported on this Galaxy** (warns `CPLD FW v1.16+
  required ... otherwise use -glx_reset`). It half-reset device 0 and **desynced the UMD device
  map**.
- Symptom 2 (post-`-r 0`): `ttnn.GetNumPCIeDevices()` / `GetPCIeDeviceID(0)` now throw
  `IndexError: unordered_map::at`, even though the cluster opens all 32 chips and chip-0 telemetry
  is healthy. So every device test errors at `conftest.py:445`.
- **The clean fix is a host-wide `tt-smi -glx_reset_auto` (or `-glx_reset`)** — which resets all 32
  ASICs and must be coordinated with other users. (Auto-mode blocked the agent from running it;
  needs explicit human/admin action.)

**Recovery check after a proper reset (should print `32` then a device id, no exception):**
```bash
source /home/tt-admin/avan/tt-metal/python_env/bin/activate && \
python3 -c "import ttnn; print(ttnn.GetNumPCIeDevices()); print(ttnn.GetPCIeDeviceID(0))"
```

On a fresh server this blocker likely won't exist — just ensure the cluster is healthy/idle before
running.

---

## 9. Key file map

| Path | Role |
|------|------|
| `models/experimental/pi0_5/sockets.md` | Socket API + direct-write mechanism reference |
| `models/experimental/pi0_5/tt/ttnn_pi0_5_model.py` | Model; **edited** — `run_prefix`/`run_denoise`/`sample_actions` |
| `models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e.py` | Single-mesh baseline e2e perf test |
| `models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_socket.py` | **NEW** socket-split e2e perf test |
| `tests/ttnn/distributed/test_socket_perf.py` | Socket async-vs-direct GB/s micro-benchmark (helper source) |
| `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/` | The d2d socket ops (in-tree, already compiled) |
| `tools/tracy/` | tracy profiler (`__main__.py`, `process_ops_logs.py`) |
| `generated/pi05_baseline/.../ops_perf_results_*.csv` | Baseline per-op CSV (done) |

## 10. Immediate next steps for the carryover agent
1. Ensure the cluster is healthy + idle (Section 8 check; reset host-wide if needed).
2. Run command (1) → confirm the refactor still passes single-mesh.
3. Run command (2) → fix any DRAM-over-socket / 1×1-submesh issue (Section 7 fallbacks).
4. Run command (3) → produce the socket CSV; verify `send/recv_direct_async` rows; diff vs baseline.
5. Commit the two changes on `my_flash` once green.
