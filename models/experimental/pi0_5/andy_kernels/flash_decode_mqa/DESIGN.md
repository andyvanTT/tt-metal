# flash_decode_mqa — design & build log

## Target op
Non-causal MQA attention for the pi0.5 denoise step, K-parallel across cores:
- Q (1, NH=8, Sq=32, D=256), K/V (1, 1, Sk=1056, D=256), additive mask (1,1,Sq,Sk), scale=1/√256.
- Win condition: beat the stock prefill-SDPA composed baseline of **116 µs/call** at PCC ≥ 0.999,
  with K/V resident in L1 (no batch replication). Refs + measurement: `tests/perf/_sdpa_decode_qfold_repro.py`.

Why custom: stock prefill SDPA parallelizes over batch×heads×q_chunks only → 8 cores for this
shape (Phase-0 NO-GO ruled out the sdpa_decode compose route: KV replication 3× slower / L1 overflow).

## Milestones
- **M1 ✅** single-core tile copy via `ttnn.generic_op` — packaging/CB/compile path validated
  (`test_m1_plumbing.py`, exact match).
- **M2 (in progress)** single-core attention correctness, then 8-head fan-out.
- **M3** split Sk(33 tiles) across cores + online-softmax cross-core combine = the speedup.

## M2 recipe — reuse mainline flash primitives (compute_common.hpp)
Path: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`.
The per-(head) flash inner loop (lines ~1822–2050) is exactly:
```
matmul_blocks(cb_q, cb_k, cb_qk_im, Sq_t, Sk_t, DHt, ..., transpose=true)   // QK^T (transpose handles K^T)
add_block_inplace(cb_qk_im, cb_mask, qk_chunk_tiles)                         // += additive mask
reduce_c<MAX,REDUCE_ROW,cb_qk_im,cb_id_scale,Sq_t,Sk_t>(cur_max,prev_max, k>0)  // running row-max
sub_exp_block_bcast_cols_inplace<cb_qk_im,Sq_t,scale,true>(cur_max,cur_sum,Sk_t) // exp((QK-max)*scale)+partial sum
matmul_blocks(cb_qk_im, cb_v, mm2_out, Sq_t, vDHt, Sk_t, ..., transpose=false)  // P@V
// then exp_max_diff correction + accumulate (only when k_num_chunks>1)
```
**Simplification for M2:** set `Sk_chunk_t = Skt = 33` (whole K in ONE chunk) → `k_num_chunks=1` →
the cross-chunk accumulate path is skipped (`processed_k_chunks` always 0). That removes the hardest
(flash-combine) logic for the correctness milestone; final = mm2_out normalized by cur_sum.

### CBs required (mirror sdpa.cpp:76–93)
inputs c_0 q, c_1 k, c_2 v, c_3 mask, c_5 identity_scale (1 tile of 1.0s for reduce), c_7 col_identity;
intermediates c_24 qk_im, c_25/26 out_im A/B, c_27/28 max A/B, c_29/30 sum A/B, c_31 exp_max_diff;
output c_16. For M2 single-chunk we still need qk_im, one max, one sum, one out_im, identity_scale.

### Derived matmul-blocking params (the tedious part — port from C++ host)
`qk_in0_block_w, qk_subblock_w/h, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_num_blocks` and the
`out_*` equivalents. Source of truth: the mainline program factory
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` (find_subblock /
get_matmul_subblock_params). Port that arithmetic into op.py for our tile dims
(Sq_t=1, Sk_t=33, DHt=8, vDHt=8). NOTE 33 is not a clean subblock width — may need Sk padded to 34/36
or split into sub-chunks; validate subblock constraints (out_subblock_w*out_subblock_h ≤ 8, etc.).

## M3 recipe — K-split across cores + combine
- Assign each core a contiguous slice of the 33 K-tiles (e.g. 8 cores × ~4–5 tiles, or more cores).
  Each core runs the M2 inner step over ITS K-slice → partial (out_c, max_c, sum_c) into L1.
- Cross-core combine (online softmax), per deepseek `sdpa_reduce_to_all`/`sdpa_tail` math:
  `m=max(m1,m2); s=s1·e^{(m1-m)·scale}+s2·e^{(m2-m)·scale}; o=o1·e^{(m1-m)scale}+o2·e^{(m2-m)scale}`,
  final `o/=s`. Implement as a reducer-core kernel reading all partials from L1 (semaphore-synced),
  or a tree reduce. This is the genuinely novel kernel and the main M3 risk.
- Fan out the 8 heads across the remaining grid (8 heads × K-split cores), staying ≤120 cores, K/V in L1.

## Status / blockers
- M1 committed + validated. M2/M3 pending.
- PUSH BLOCKED: forwarded SSH agent not reachable in the autonomous shell (origin = github-andy alias);
  work is in LOCAL commits on `single_optim`. Run `git push -u origin single_optim` from a shell with
  the live agent to ship.
