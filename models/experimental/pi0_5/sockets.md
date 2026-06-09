# Direct-Socket Tensor Transfer for PI0.5

This document describes the **direct-write socket ops** (`send_direct_async` / `recv_direct_async`)
that were added under `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/`, the supporting
`MeshSocket` setup API, **how to integrate them when splitting the PI0.5 model across submeshes**,
and **what the optimization actually does** versus the existing FIFO-based `send_async`/`recv_async`.

> Source ops: `ttnn.experimental.send_direct_async`, `ttnn.experimental.recv_direct_async`
> (added by *"Implement socket direct tensor transfer op"*, Sankar Manoj).

---

## 1. API calls

### 1.1 Submesh + socket setup (prerequisites)

A socket connects a **sender submesh** to a **receiver submesh**. Both are carved out of the same
`MeshDevice` (intra-process) or live in separate processes (inter-process). The transfer ops
themselves do not create sockets — you build the socket once and reuse it across iterations.

```python
import ttnn

# 1. Carve out the two submeshes (sender / receiver) from a MeshDevice.
sender_mesh   = mesh_device.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))
receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(1, 0))

# 2. Describe the sender-core -> receiver-core pairs (one or more parallel "connections"
#    per device; more connections == more parallel ethernet channels == more bandwidth).
def build_connections(mesh_shape, num_connections):
    sender_cores = [ttnn.CoreCoord(i, 0) for i in range(num_connections)]
    recv_cores   = [ttnn.CoreCoord(i, 1) for i in range(num_connections)]   # disjoint cores!
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for s, r in zip(sender_cores, recv_cores):
            connections.append(
                ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, s), ttnn.MeshCoreCoord(coord, r))
            )
    return connections

# 3. Build the socket config. The FIFO is now only used for the handshake + completion
#    token, so the page size can be small (it does NOT carry payload in direct mode).
connections   = build_connections(sender_mesh.shape, num_connections=2)
socket_mem    = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo_size_bytes=socket_page_size * 4)
socket_config = ttnn.SocketConfig(connections, socket_mem)

# 4. Create the paired endpoints (one call returns both halves).
send_socket, recv_socket = ttnn.create_socket_pair(sender_mesh, receiver_mesh, socket_config)
```

### 1.2 The transfer ops

```python
# Sender side — runs on the submesh that owns `input_tensor`.
ttnn.experimental.send_direct_async(input_tensor, send_socket)   # -> [] (empty vector)

# Receiver side — runs on the submesh that owns `output_tensor`.
ttnn.experimental.recv_direct_async(output_tensor, recv_socket)  # -> [output_tensor]
```

| Op | Args | Returns | Notes |
|----|------|---------|-------|
| `send_direct_async` | `input_tensor: ttnn.Tensor`, `mesh_socket: ttnn.MeshSocket` | empty vector | Writes each page **straight into the receiver's output tensor**. Socket only carries the handshake + completion signal. |
| `recv_direct_async` | `output_tensor: ttnn.Tensor`, `mesh_socket: ttnn.MeshSocket` | `[output_tensor]` | Advertises `output_tensor`'s address to the sender over the socket, then waits for the completion page. Performs **no payload data movement** itself. |

**Critical contract:** `recv_direct_async` does *not* allocate the output. You must pre-allocate a
tensor on the receiver submesh with a spec matching the input, and pass it in:

```python
output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, receiver_mesh)
```

Both ops are **async** (non-blocking enqueue). The data is only guaranteed present after a
device sync on *both* submeshes:

```python
ttnn.synchronize_device(sender_mesh)
ttnn.synchronize_device(receiver_mesh)
```

---

## 2. Integrating into PI0.5

PI0.5 today runs on a **single mesh**. The clean place to introduce a socket transfer is a
**pipeline/model-parallel split** along the model's existing two-stage boundary:

```
        submesh A (PREFIX)                         submesh B (SUFFIX / action expert)
  ┌──────────────────────────────┐          ┌────────────────────────────────────────┐
  │ SigLIP vision tower (27 blk)  │          │  10-step flow-matching denoise loop      │
  │ + projector + Gemma 2B VLM    │  ──d2d─► │  Gemma 300M expert (AdaRMSNorm)          │
  │ → prefix embeds + prefix K/V  │  socket  │  reads shared prefix K/V every step      │
  └──────────────────────────────┘          └────────────────────────────────────────┘
```

The prefix (vision + VLM) is computed **once per chunk**; the suffix denoise loop iterates ~10×
reading the prefix's shared attention K/V. That makes the **prefix K/V cache** (and the prefix
embeddings) the natural payload to transfer A→B exactly once per inference, after which submesh B
runs the entire denoise loop locally. This is the same producer→consumer shape the bandwidth
benchmark in `tests/ttnn/distributed/test_socket_perf.py` measures.

### 2.1 Where this hooks into the code

- **Producer:** `Pi0_5PaliGemmaBackboneTTNN` / the VLM prefix path in
  `tt/ttnn_paligemma.py` + `tt/ttnn_prefix.py` produce the prefix K/V and embeddings.
- **Consumer:** the denoise loop in `tt/ttnn_pi0_5_model.py` (`sample_actions` → `forward_*`),
  which currently consumes those tensors in-place. With a split it would consume the *received*
  copies on submesh B.
- **Artifact prep:** `prepare_upstream_artifacts` / `_build_upstream_attn_artifacts` already
  centralizes the prefix-derived tensors (prefix attn mask, RoPE tables, prefix K/V) — this is the
  natural set of tensors to ship across the socket and the natural call site to enqueue the sends.

### 2.2 Integration pattern (build once, transfer per chunk)

```python
# --- one-time setup (model construction) ---
send_socket, recv_socket = ttnn.create_socket_pair(prefix_mesh, suffix_mesh, socket_config)
# pre-allocate receiver-side landing buffers, one per tensor you transfer:
kv_recv = ttnn.allocate_tensor_on_device(prefix_kv.spec, suffix_mesh)

# --- per chunk ---
prefix_kv = run_prefix(prefix_mesh, images, lang, state)      # on submesh A
ttnn.experimental.send_direct_async(prefix_kv, send_socket)   # A -> B, payload bypasses FIFO
ttnn.experimental.recv_direct_async(kv_recv, recv_socket)     # B advertises kv_recv addr, waits

ttnn.synchronize_device(prefix_mesh)
ttnn.synchronize_device(suffix_mesh)

actions = run_denoise_loop(suffix_mesh, kv_recv)              # 10 steps, all local to B
```

### 2.3 Integration checklist / gotchas

1. **Pre-allocate the receiver tensor** with `allocate_tensor_on_device(src.spec, recv_mesh)` and
   reuse it — `recv_direct_async` writes into it, it does not allocate.
2. **Spec must match** between `input_tensor` and `output_tensor` (shape, dtype, layout, page size).
   The sender streams pages directly into the receiver's tensor address, so layouts must agree.
3. **Sender and receiver cores must be disjoint** — the socket runtime forbids a core appearing in
   two connections of the same socket (hence row 0 = senders, row 1 = receivers above).
4. **Issue send + recv every iteration and sync both submeshes** before reading the output.
5. **Keep the socket alive** for the model's lifetime; creating it is the expensive part, the
   transfer ops are cheap to enqueue. Pairs well with PI0.5's existing trace + 2CQ perf path —
   the transfer can be captured in the trace and replayed per chunk.
6. **Tune `num_connections`** (1/2/4) and FIFO `socket_page_size`. In direct mode the FIFO only
   holds the handshake/completion token, so the page size mainly affects pipelining of the small
   control traffic; bandwidth scales with the number of parallel connections.

---

## 3. What the optimization does

### 3.1 The problem with FIFO `send_async` / `recv_async`

The existing socket ops stage the **entire payload through the socket FIFO**, which forces an
extra copy on the receiver. The regular receiver runs **two** kernels:

- `receiver_reader.cpp` — `noc_async_read` from the socket FIFO into a local scratch CB.
- `receiver_writer.cpp` — `noc_async_write` from that scratch CB into the final output tensor.

So a page travels: **sender L1 → socket FIFO (receiver L1) → scratch CB → output tensor** — an
extra full L1 round-trip plus receiver-side data kernels burning cores.

### 3.2 The direct-write path

`send_direct_async` / `recv_direct_async` use the FIFO **only for control**, and stream the
payload straight into the destination tensor. The mechanism (see `sender_direct_writer.cpp` and
`receiver_direct.cpp`):

1. **Handshake.** The sender advertises its handshake-buffer address over the socket. The receiver
   writes back its **output tensor base address, page size, and page count** via fabric inline
   writes, setting a `valid` flag **last** so the sender never observes a partially-written struct.
2. **Direct streaming.** The sender writes tensor pages **directly to the receiver's output-tensor
   NOC address** with fabric unicast writes (`to_noc_unicast_write`). The receiver kernel does
   **zero data movement** — no reader, no writer, no scratch copy.
3. **Completion.** The sender pushes a single page onto the socket as a done token; the receiver
   waits on it and returns.

It handles both page regimes: **small pages** are packed multiple-per-fabric-packet
(`num_pages_per_packet > 0`); **large pages** are split across multiple packets per output page.

### 3.3 Why it is faster (the d2d win)

- **Eliminates the receiver double-copy.** One full L1 round-trip of the payload is removed
  (FIFO→scratch→output collapses to a single sender→output write).
- **Frees the receiver data kernels.** The receiver no longer spends cores reading/writing the
  payload — it just handshakes and waits, so transfer approaches **raw fabric write bandwidth**.
- **FIFO shrinks to a control channel.** No need to size the FIFO to the tensor; it only holds the
  handshake + completion token, freeing L1.
- **Latency drops** because the handshake is a one-time per-transfer cost and the payload is a
  single directed stream rather than a staged copy pipeline.

### 3.4 Measuring it

`tests/ttnn/distributed/test_socket_perf.py` is a bandwidth micro-benchmark that sends a tensor
between two submeshes and reports per-chip / aggregate GB/s, parameterized over
`transfer_mode ∈ {"async", "direct"}`, `num_connections`, FIFO page size, and tensor size. It runs
both modes head-to-head and writes a CSV (`MESH_SOCKET_BW_CSV`, default `mesh_socket_bandwidth.csv`):

```bash
MESH_SOCKET_BW_CSV=/tmp/bw.csv pytest tests/ttnn/distributed/test_socket_perf.py::test_mesh_socket_bandwidth -s
```

> Note: no benchmark numbers are committed in the repo — run the test on a multi-device target to
> get the `async` vs `direct` GB/s comparison for your tensor sizes before wiring the split into the
> PI0.5 perf path.
