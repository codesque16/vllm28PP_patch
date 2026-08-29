# Commits on top of v0.28.0

Base tag: **`v0.28.0`** (`2cf0a69`).  
Branch: **`dspark-pp`**.  
HEAD: **`6d3da80`**.  
Range: `v0.28.0..HEAD` (7 commits).

These patches make DeepSeek-V4 DSpark speculative decode work with pipeline parallelism on 8× RTX PRO 6000 (SM120), without changing Hopper’s small-batch MHC decode path.

| # | Commit | Subject | Arch impact | Throughput impact |
|---|---|---|---|---|
| 1 | `d9edde6` | Enable DSpark speculative decode with pipeline parallelism | Spec+PP, all GPUs | Enables spec+PP; small last-rank stream wait |
| 2 | `38ba7ea` | Log KV cache capacity math | All | None (logging) |
| 3 | `bf9505e` | Ship MHC mixer state across PP ranks | DeepSeek-V4 PP>1, Hopper and SM120 | ~25% more bytes per PP hop; TP-only unchanged |
| 4 | `9568a4a` | SM120 sparse indexer `torch.topk` prefill fallback | Prefill fallback SM12x only; bounds check all | SM12x prefill slower; Hopper ~unchanged |
| 5 | `26b63d5` | Disable MHC prenorm split-K on SM120 | SM12x (Hopper keeps small-FMA decode) | SM120 1–16 tok decode may be slower; Hopper decode unchanged |
| 6 | `2aec4bf` | Broadcast DSpark draft token ids to earlier PP ranks | Spec+PP, any GPU | Tiny extra NCCL (`num_reqs×7` int64) |
| 7 | `6d3da80` | Ignore `specdec_benchmarking/bench_data` | n/a | None |

---

## 1. `d9edde6` — Enable DSpark speculative decode with pipeline parallelism

**Issue:** Stock v0.28 refused to load DSpark with `PP>1`. The draft is `DeepSeekV4MTP`, which does not implement `SupportsPP`; aux-hidden capture was required on every rank (which would return a `(hidden, aux)` tuple instead of `IntermediateTensors` on earlier ranks); sampled-token PP broadcasts used the raw sampler shape `[num_reqs, 1]` on prefill while receivers expected `[num_reqs, num_spec+1]`, which deadlocks NCCL; MTP checkpoints have no embedding table, so last-rank draft embeddings would be a `PPMissingLayer` and accept rate collapses to ~0%.

**Change:** Skip the draft `SupportsPP` check for `method=dspark`. Capture aux hidden on the last PP rank only. Pad sampled-token broadcasts to `[num_reqs, max_sample_len]` and wait on the last-rank main stream until that collective finishes. Load the **target** embedding table into the last-rank drafter instead of aliasing a missing PP embed. Also adds the specdec bench launch/parse scripts.

**Solves:** The engine can load and run DSpark with pipeline parallelism (e.g. TP2PP4S). This commit is load/handshake plumbing only; later commits fix PP MHC quality and real draft-id broadcast.

**Architectures:** Spec+PP on NVIDIA, AMD, and XPU (dspark wrappers on all three). Greedy and TP-only spec still work. Eagle3/dflash+PP remain rejected.

**Throughput:** No kernel change. Last-rank `main_stream.wait_stream(broadcast_stream)` after the sampled broadcast is required to avoid an NCCL deadlock during mixed warmup; it is a small sync, not a compute tax. Spec+PP itself is extra decode work versus greedy.

**Files:**

- `vllm/config/speculative.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/pp_utils.py`
- `vllm/v1/worker/gpu/spec_decode/dspark/utils.py`
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
- `vllm/models/deepseek_v4/nvidia/dspark.py`
- `vllm/models/deepseek_v4/amd/dspark.py`
- `vllm/models/deepseek_v4/xpu/dspark.py`
- `specdec_benchmarking/.gitignore` (added)
- `specdec_benchmarking/launch_bench_server.sh` (added)
- `specdec_benchmarking/benchmark_random.sh` (added)
- `specdec_benchmarking/new_parse_sweep_results.py` (added)

---

## 2. `38ba7ea` — Log KV cache capacity math

**Issue:** The printed GPU KV token count is `concurrency * max_model_len`. On DeepSeek-V4 hybrid MLA/window cache that hides per-group block accounting, so operators cannot tell whether a long-context config actually fits.

**Change:** Dump per-group KV blocks when logging cache capacity.

**Solves:** Visible hybrid MLA/window KV math in server logs.

**Architectures:** All platforms that use this KV cache logger.

**Throughput:** None. Logging only.

**Files:**

- `vllm/v1/core/kv_cache_utils.py`

---

## 3. `bf9505e` — Fix DeepSeek-V4 PP quality by shipping MHC mixer state across ranks

**Issue:** Stock PP collapsed each pipeline cut with `mhc_post` and NCCL-sent only reconstructed `hidden_states`. The next rank set `residual` / `post_mix` / `res_mix` to `None` and ran standalone `mhc_pre` as if it were layer 0. Inside a rank, MHC defers each sublayer’s post into the next fused post-pre, so that restart is not equivalent. After ~320+ prefill tokens the last-token MHC state is wrong and the first LM-head sample is CJK (`姶` / `姞` / `姍`). Reproduced on PP8 greedy without spec; TP8 greedy is fine.

**Change:** Carry `hidden_states (T, H)` plus `residual`, `post_mix`, and `res_mix` across the PP cut, and skip `mhc_post` on non-last ranks so the next rank continues fused post-pre.

**Solves:** Long-prompt GSM8K/CJK collapse on DeepSeek-V4 pipeline parallel.

**Architectures:** All NVIDIA GPUs that run DeepSeek-V4 with `PP>1` (Hopper SM90 and Blackwell SM120). TP-only and `PP=1` are unchanged.

**Throughput:** About 25% more bytes per PP hop (extra residual `T×4×H` bf16 plus tiny fp32 mixers; `hidden_states` shrinks from `T×4×H` to `T×H`). Decode kernel choice is unchanged. TP-only throughput is unaffected.

**Files:**

- `vllm/models/deepseek_v4/nvidia/model.py`

---

## 4. `9568a4a` — Fix SM120 sparse indexer IMA by falling back to torch.topk in prefill

**Issue:** On SM12x, `ops.top_k_per_row_prefill`’s histogram path emits garbage indices (including NaN bit patterns) when logits contain NaNs. Downstream, `compute_global_topk_indices_and_lens` treats any index `>= 0` as valid and gathers OOB KV block-table entries, which is a CUDA illegal memory access / `EngineDead` on long C128A prefill ([vllm#49896](https://github.com/vllm-project/vllm/issues/49896) / [#49897](https://github.com/vllm-project/vllm/issues/49897)). Decode already excluded SM12x from `cooperative_topk`; prefill still used the CUDA kernel on this v0.28 tree.

**Change:** SM12x prefill uses a `torch.topk` fallback that matches the CUDA kernel’s contract (`ks`-relative indices, ascending order, `-1` pad, skip NaNs). Also reject OOB block indices in the Triton gather as a safety net if a bad index still appears.

**Solves:** 5k-token random-prefill IMA on RTX PRO 6000 (SM120).

**Architectures:** The prefill fallback is SM12x only. Hopper and Ada keep the CUDA kernel. The Triton bounds check runs on every arch; it only changes behavior if an index was already OOB.

**Throughput:** SM12x prefill top-k is slower (`torch.topk` vs fused CUDA). Hopper prefill/decode throughput is unchanged aside from a negligible Triton predicate.

**Files:**

- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/models/deepseek_v4/common/ops/cache_utils.py`
- `tests/kernels/test_top_k_per_row.py`

---

## 5. `26b63d5` — Disable MHC prenorm split-K on SM120 to stop CJK first-token corruption

**Issue:** SM120 `hc_prenorm` split-K (occupancy 31/32/18/…) corrupts last-token MHC state, so the first sample after prefill is CJK. Hopper SM90 is fine with split-K. A second copy of the same bug lived in the small-batch fused FMA path: for `num_tokens <= 16` it overrode the caller’s `n_splits=1` with Hopper heuristics (8 if `tokens<8` and `H<=4096`, else 4). DSpark sequential decode is exactly 8 tokens (1+7 drafts) and hit `n_splits=4`, which produced `姤`/`姫` decode loops. Concurrent spec (64 tokens) used DeepGEMM `n_splits=1` and was clean.

**Change:** `compute_num_split` returns 1 on capability `>= 12`. Skip the small-FMA split-K=4/8 path on SM120 and use DeepGEMM with `n_splits=1`. Hopper still uses small-FMA for 1–16 tokens. Hopper occupancy is also floored to a power of two so odd SM counts tile evenly.

**Solves:** Greedy and spec GSM8K CJK on SM120. The sequential spec `姤`-loop is gone; MAL stays above 4 with real rejection sampling.

**Architectures:** Split-K disable and small-FMA skip are SM12x only. Hopper keeps small-FMA decode. Hopper prefill `n_splits` may drop (e.g. 31→16) from the power-of-two floor.

**Throughput:** SM120 decode of 1–16 tokens may be slightly slower (DeepGEMM `n_splits=1` vs fused FMA split-K 4/8). Hopper decode throughput is unchanged. Hopper large-prefill MHC may lose a bit of split-K occupancy from the floor.

**Files:**

- `vllm/model_executor/kernels/mhc/tilelang.py`
- `vllm/model_executor/kernels/mhc/tilelang_kernels.py`

---

## 6. `2aec4bf` — Broadcast DSpark draft token ids to earlier PP ranks

**Issue:** `propose()` runs only on the last PP rank. Earlier ranks never saw the real draft ids and embedded placeholders or stale slot data on the next pipeline tick. Rejection then compared last-rank drafts against first-rank embeddings of the wrong ids, which shows up as false accepts, ~0% real accept, or GSM8K collapse under spec+PP. Sampled (bonus) tokens were already broadcast; drafts were not.

**Change:** After `propose()`, broadcast padded `[num_reqs, num_spec]` int64 draft ids on the same `pp_broadcast` group, store them in the `T+pp_size` FIFO slot with the bonus token, and apply both in `update_pp_decode_requests`. Skip the extra collective when the sampled broadcast was skipped so NCCL stays matched. Zero `last_sampled_tokens` when a request slot is reused.

**Solves:** Real DSpark accept under PP (MAL ~4 with standard reject) instead of placeholder-id false accepts.

**Architectures:** Spec+PP on any GPU (Hopper and Blackwell). No effect on greedy or TP-only spec.

**Throughput:** One extra NCCL broadcast of `num_reqs×7` int64 per decode step with spec+PP. Negligible vs MHC activation traffic. Greedy and TP-only throughput are unchanged.

**Files:**

- `vllm/v1/worker/gpu/pp_utils.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/states.py`

---

## 7. `6d3da80` — Ignore specdec_benchmarking/bench_data prepared eval dumps

**Issue:** Prepared GSM8K / math JSONL under `specdec_benchmarking/bench_data/` would otherwise show up as untracked or get committed by accident.

**Change:** Add `bench_data/` to `specdec_benchmarking/.gitignore`.

**Solves:** Keeps downloaded/prepared eval dumps out of git.

**Architectures:** n/a.

**Throughput:** None. No runtime change.

**Files:**

- `specdec_benchmarking/.gitignore`
