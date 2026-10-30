# ADR 0003 — Block-parallel (DSpark) speculative decoding for LFM2.5 targets (ticket #114)

**Status:** accepted (design), implementation not started. Depends on #113 (LFM3 tri-hybrid
config/shape) and on the hybrid forward pass that #113 defers.

## Context

`kanso-infer` already ships a speculative decoding path (`src/speculative/`): a **full
autoregressive draft model** proposes `gamma` tokens one at a time on its own paged KV
cache (`src/speculative/draft_cache.rs`), the target verifies the whole candidate block in
**one** batched forward (`src/speculative/verifier.rs` → `Model::forward_batch_all_rows`),
and partial accepts roll the KV cache back (`BlockTable::rollback_blocks`). Greedy requests
accept on argmax match; `temperature > 0` requests use standard rejection sampling
(`min(1, p/q)` with residual correction), so the emitted distribution is the target's.

That machinery is target-agnostic and stays as-is. What it cannot express is **DSpark**,
the drafter family DeepSeek uses in DeepSeek-V4 and Liquid AI ships for LFM2.5:

* a **DFlash-style parallel backbone** — 5 layers, one forward pass over a block of `k`
  tokens, conditioned on **hidden states tapped from the target** (`target_layer_ids`);
* a **Markov head** (rank 256, `markov_w1: Embedding(V, r)` → `markov_w2: Linear(r, V)`)
  that adds a *sequential* token-to-token dependency to the block's logits, which is what
  fixes the "suffix decay" of purely parallel drafters;
* an optional **confidence head** that predicts per-position survival probability and lets
  the verifier truncate the block (confidence-scheduled verification).

Paper: Cheng et al. (2026), *DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation*, [arXiv:2607.05147](https://arxiv.org/abs/2607.05147).
Reported in DeepSeek-V4 serving: **+60–85 % per-user generation speed** over the
production baseline (MTP-1) at matched throughput.

Two facts constrain what we can do here:

1. **The targets are not loadable today.** `src/weights.rs` rejects every non-dense
   architecture (`llama`, `granite` only). DSpark's published targets for LFM are the
   LFM2.5 family (ShortConv + attention hybrid). DSpark is therefore a *consumer* of the
   hybrid forward pass, not a parallel workstream: the hybrid forward pass comes first.
2. **The repo's only speculative drafter is autoregressive.** There is no notion of a
   drafter that consumes target hidden states, no block-parallel proposal path, and no
   Markov/confidence head. `SpeculativeEngine` holds `Option<Model>` — a whole second
   dense model with the *same* vocabulary — and proposes one token at a time.

Measured outcome of DSpark on LFM2.5 (for sizing the prize, not a claim about our
hardware):

| target | drafter | acceptance (of 10) | M4 Max (llama.cpp + Metal) |
| :-- | :-- | :-- | :-- |
| LFM2.5-1.2B-Instruct | `LiquidAI/LFM2.5-1.2B-Instruct-DSpark` | 5.02 | 2.54× (138 → 350 tok/s) |
| LFM2.5-2.6B | `LiquidAI/LFM2.5-2.6B-DSpark` | 4.81 | 2.27× (61 → 139 tok/s) |
| LFM2.5-8B-A1B (MoE) | `LiquidAI/LFM2.5-8B-A1B-DSpark` | 6.95 | 1.18× (MoE verify is expert-bound) |

(mlx-dspark measures higher on the same family — 3.35× for 1.2B bf16, 2.79× for 2.6B bf16 —
because its verify kernels are cheaper; the ratio is a property of the machine, not of the
method.)

## Decision

**Adopt DSpark as a second drafter kind behind a new drafter abstraction, and gate it on
the hybrid forward pass landing first.** The existing autoregressive draft path stays
untouched and remains the default.

### D1 — A `DraftModel` abstraction with two implementations

The engine stops knowing that a drafter is "a `Model` that emits one token per step". A
trait (or enum) exposes one operation — *propose a block for a sequence* — with two
implementations:

* `AutoregressiveDrafter` — today's behaviour, wrapped unchanged (`DraftCache::propose`),
  used for the dense targets the repo supports today;
* `DSparkDrafter` — the new one: tap-driven, block-parallel, Markov-corrected.

Both return the same `Proposals { tokens, probs }`, so the verifier, the profiler, the
rollback logic and the CLI/serve paths do not change. The DSpark drafter additionally
returns its per-position confidence so the verifier can truncate the block (D4).

### D2 — The target needs a hidden-state tap

The DSpark backbone is conditioned on the target's hidden states at `target_layer_ids`
(e.g. `[2, 9, 17, 21, 27]` for LFM2.5-2.6B). This requires:

* the target forward to expose the hidden state at chosen layers (an explicit tap API, not
  a side effect of the layer loop);
* those taps to be captured during the **verify** forward, since the DSpark block for the
  next round is a function of the target's last hidden states — the tap must be part of the
  same forward that produced the committed token, not an extra forward;
* the tapped rows to survive a partial accept: a rejected suffix must not invalidate the
  taps of the accepted prefix.

### D3 — The LFM2 hybrid forward pass is a prerequisite, and so is conv-state rollback

LFM2 (LFM2.5) layers are either **full attention** or **ShortConv** (a causal FIR filter
with a rolling window, `conv_L_cache = 3` for LFM2.5). Rejecting a draft suffix does not
just trim the KV cache — it must also move the conv state back to the accept point.

The state is *not* replayed: llama.cpp's implementation (PR #27383) keeps the last
`n_rs_seq + 1` conv-state **slots** and selects the slot matching the accept point, so a
rollback is a pointer move, not a re-run. We adopt the same shape: per-layer conv state
with a small ring of snapshots, indexed by accept length.

Layer layout is data, not code: `layer_types` from the model config decides which layers are
attention and which are conv (LFM2.5-350M: 6 attention layers at `[2, 5, 8, 10, 12, 14]`
+ 10 conv; LFM2.5-2.6B: 6 + 10; LFM3-1.7B: tri-hybrid, see #113).

### D4 — The verifier stays the single source of truth; the confidence head is optional

Losslessness is non-negotiable: the target verifies every drafted token, so output equals
plain decoding (up to floating-point ties). The confidence head may only *shorten* the
verified block — it can never change which tokens are emitted. It is therefore a
**performance knob, not a correctness mechanism**, and it is behind a flag.

Whether it pays is machine- and target-dependent, and the evidence is mixed: in mlx-dspark
it pays on MoE targets (where the verify curve rises steeply) and *loses* on flat-curve
dense 8-bit targets. We adopt it only after measuring our own verify-cost curve, and we
record the measurement in the bench report — not as a default.

### D5 — Drafter packaging: GGUF, target's tied embed + lm_head, interleaved RoPE

Liquid AI publishes safetensors *and* GGUF for every LFM2.5 drafter
(`LiquidAI/LFM2.5-{1.2B-Instruct,2.6B,8B-A1B}-DSpark-GGUF`), so the drafter can be loaded
through the existing GGUF path. Three properties of that packaging drive the loader work:

* the drafter **reuses the target's tied embedding and `lm_head`** — it ships neither, so
  the loader must resolve them from the target, and the drafter's vocab (128 000 for
  LFM2.5) must match the target's;
* the checkpoint's arch tag is `Lfm2DSparkDraftModel` (`model_type: qwen3`) with the
  DSpark fields nested in `dflash_config` (`target_layer_ids`, `mask_token_id`), *not* the
  top-level `projector_type` layout used by the DeepSpec-native drafters — the loader must
  accept both spellings;
* the drafter uses **interleaved** RoPE (`rope_is_neox_style: false`). Our RoPE path is
  rotate-half (NeoX). llama.cpp permutes the q/k weight rows at conversion time
  (`conversion/qwen.py` in PR #27383) and SGLang added an `IS_NEOX` constexpr to its fused
  kernel (PR #31041). We must do the equivalent **at load time** (permute the drafter's
  q/k projection rows per head) or acceptance collapses — measured at roughly half of the
  interleaved value.

### D6 — Target selection: the two LFM2.5 sizes that have official drafters

The first target is **LFM2.5-1.2B-Instruct (bf16)** and the second **LFM2.5-2.6B (bf16)**:
both have official drafter checkpoints, both are bandwidth-light enough that the verify
width is cheap, and both are measured to gain. We deliberately do **not** target:

* **LFM2.5-8B-A1B** — MoE; the verify row activates fresh experts, so the ratio is bounded
  by the expert traffic (measured 1.18× on M4 Max, and a *net loss* at 8-bit where the
  ~1B-active target step is already too cheap to amortize the drafter);
* **LFM2.5-350M** and **LFM3-1.7B** — no published drafter exists, and the economics are
  poor (a cheap target step has little to amortize: the same argument retires
  Qwen3.5-0.8B with a drafter at 0.96×). Deriving or training a drafter for them is a
  separate research track (ticket #115), not part of this ADR.

### D7 — Losslessness is measured, not asserted

The bench harness must show that DSpark output is **byte-identical** to the baseline decode
of the same target on the same prompts, and the speculative profiler's accept-length must
be reported next to the tok/s delta. A DSpark run that is faster but not byte-identical is
a failure, not a result.

## Alternatives considered

**A. Shell out to an existing DSpark implementation (llama.cpp / mlx-dspark) and proxy.**
Rejected: `kanso-infer` *is* the engine here — its API, paged KV, continuous batching,
prefix cache, Metal decode path and bench gates are the deliverable. Proxying a second
engine would mean two model runtimes, two KV caches and no shared prefix cache, and would
make every performance claim about `kanso-infer` unverifiable. It remains the right
*reference* for measurement (see D7) and the right place to check numbers.

**B. DFlash instead of DSpark (block diffusion, no Markov head).**
Rejected as the primary: DFlash's full 16-block reaches higher acceptance on structured
content but is a net loss on chat and on small targets, and it is the *same* engineering
work (tap, block-parallel drafter, target-side rollback) without the sequential correction.
DSpark is also the lineage Liquid AI ships for LFM, so the checkpoints exist. The
comparison stays in the bench report.

**C. Multi-token prediction (MTP).**
Rejected: LFM models ship no MTP heads (Liquid's own docs say MTP does not work with LFM),
and DSpark's own serving results are measured against an MTP-1 baseline — so the target we
would be competing with is exactly the one that does not exist for this family.

**D. Train our own drafter now (before the hybrid forward pass exists).**
Rejected for this ADR: the training pipeline needs the target's hidden-state cache and the
target's own forward pass to generate it, so it is downstream of the same prerequisite as
the inference path. Deferred to #115, which is scoped as research first.

**E. Keep only the autoregressive drafter and skip DSpark.**
Rejected: the autoregressive drafter pays a full forward per proposed token, which is
exactly the cost DSpark's parallel backbone removes. It is the fallback, not the goal.

## Consequences

**New code (target side)**

* `src/hybrid/`: real ShortConv forward + rolling state, with a snapshot ring for
  partial rollback (replacing the 27-line `recurrent_state.rs` stub, which is not a conv);
* a tap API on the target forward (hidden state per selected layer, per row, surviving a
  partial accept);
* `src/weights.rs`: `lfm2` / `lfm2_moe` architecture acceptance, `layer_types` parsing,
  conv-state tensors, and a drafter arch (`Lfm2DSparkDraftModel`) that borrows the
  target's embed + `lm_head`.

**New code (drafter side)**

* `src/speculative/dspark/`: the 5-layer block-parallel backbone, the hidden-state
  projector, the Markov head, the confidence head, the block proposal (one forward for
  `k` tokens), and the interleaved-RoPE row permutation at load.

**Behavioural changes**

* `SpeculativeEngine` grows a second drafter kind; the CLI/serve surface gains
  `--drafter <path>` (auto-resolve for the LFM2.5 family, mirroring mlx-dspark's
  quant-agnostic registry) and `--max-draft` / `--confidence-threshold` knobs.

**Verification and gates**

* `tests/`: byte-identical output vs. baseline decode (greedy) at several accept lengths;
  a conv-state rollback test (accept prefix length 0, 1, k−1, k) that compares against a
  re-run from scratch; a Markov-head ablation (with/without) to prove the correction is
  wired in; an interleaved-RoPE parity test (permuted vs. reference) ;
* bench: accept-length, tok/s and TTFT/ITL reported for baseline vs. DSpark, plus a
  `gates/` budget entry so a regression in accept length fails CI rather than silently
  shipping.

**Risks**

* *Schedule*: this is the largest single addition to the engine so far, and it is
  downstream of the hybrid forward pass. #113's Phase 1 (config/shape only) is not
  enough — the conv forward and the tap are the real prerequisites.
* *Numerical*: the DSpark backbone adds a second quantisation path; acceptance can be
  sensitive to rope layout, tap position and the markov correction, and a silent
  regression looks like "speculation is just slow". The byte-identity test and the
  accept-length report exist to make that visible.
* *Economics*: on the smallest targets DSpark is a net loss. The engine must be able to
  *park* speculation (fall back to plain steps) rather than make every decode slower.

## References

* Paper: Cheng et al. (2026), *DSpark: Confidence-Scheduled Speculative Decoding with
  Semi-Autoregressive Generation* — <https://arxiv.org/abs/2607.05147>
* Liquid AI blog: *LFM2.5-DSpark: Up to 3.2x Faster Inference from H100 to MacBook* —
  <https://www.liquid.ai/blog/lfm2.5-dspark>
* Liquid AI model card: `LiquidAI/LFM2.5-2.6B-DSpark` —
  <https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark>
* Liquid AI docs: *Speculative Decoding* (only DSpark is supported for LFM; no MTP) —
  <https://docs.liquid.ai/lfm/key-concepts/speculative-decoding>
* mlx-dspark (MLX port, registry + measurement) — <https://github.com/ARahim3/mlx-dspark>
* DeepSeek DeepSpec (training code) — <https://github.com/deepseek-ai/DeepSpec>
* llama.cpp PR #27383 (LFM2 DSpark: conv-state slot rollback, interleaved-rope
  permutation) — <https://github.com/ggml-org/llama.cpp/pull/27383>
* llama.cpp experimental Metal kernels PR #27441 —
  <https://github.com/ggml-org/llama.cpp/pull/27441>
* SGLang PR #31041 (LFM2 / LFM2-MoE DSpark support, `IS_NEOX` in the fused-KV kernel) —
  <https://github.com/sgl-project/sglang/pull/31041>
* LFM3 tri-hybrid architecture (context for #113) —
  <https://github.com/ggml-org/llama.cpp/pull/31252>
