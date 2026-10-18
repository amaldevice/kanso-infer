# ADR 0002 — Scope of the fused Metal layer on the decode path (ticket #87)

**Status:** accepted, with debt recorded.

## Context

`src/metal/fused_layer.rs` carried a tested but caller-less chain: `MetalWeights`
(resident weights, one command buffer for RMSNorm + QKV + RoPE), `encode_fused_qkv_gemv`,
`encode_linear_gemv`, `gpu_chained_pipeline_single_encoder` and the
`gpu_discrete_layer_multi_encoder` baseline. Nothing outside `tests/` called any of it, so
the README row read 🧪 "implemented and tested, but no production caller".

Ticket #87 asks for that kernel to become the decode path, with the fused work chained into
one command buffer per step for attention/O/MLP. Spec #82 puts **new kernels** out of
scope: this work stream connects kernels that already exist and are already tested.

## Decision

The dense Metal decode step runs each layer's **prologue** — RMSNorm, Q, K, V projections
and RoPE — through `MetalWeights::decode_step`, i.e. through the fused kernel, on one
command buffer per layer. The seam lives in `src/model/fused_decode.rs` and is entered from
`Model::forward_batch`, the entry `Engine::step` decodes through.

Attention stays on the #86 path (`paged_attention_v2`, dispatched by `Model::metal_paged_attention`),
and the output projection and MLP stay on the tensor path. There is exactly one Metal
decode path; the fused kernel is the prologue of it, not a second loop.

A model the kernel cannot represent keeps the per-op prologue. `fused_prologue_supported`
rejects the geometry *before* anything is uploaded, and a quantized projection (no dense F32
rows to upload) is likewise a fallback, so the seam never turns a representable-shape
question into a decode error.

## Debt: one command buffer per step, not per layer

Acceptance criterion 4 asks for one command buffer per step covering attention/O/MLP. That
is **not** what this change delivers, and the reason is concrete rather than scheduling:

1. `MetalWeights` / `encode_fused_qkv_gemv` cover the QKV prologue only. Chaining attention
   and the MLP into the same encoder needs a residual-add kernel (`hidden += proj * scale`)
   and an MLP activation kernel (SwiGLU). Neither exists today.
2. The only existing attention entry points — `gpu_paged_attention_v2_buffer` and
   `Model::metal_paged_attention` — create, commit and wait on their own command buffer.
   Re-encoding them into a caller-owned encoder means changing that signature, i.e. a second
   attention encoder path, which is exactly what #86's "no second Metal path" rule forbids.

Both fixes are new kernels or a new attention-encoding seam. Under spec #82's out-of-scope
rule that is a separate ticket, not a silent expansion of this one.

What this change *does* establish, and what `tests/test_fused_metal_decode.rs` asserts on the
production entry point:

* the fused kernel runs in the decode loop (`Model::fused_layer_calls` moves on Metal, stays
  zero on CPU);
* logits match the CPU reference within `1e-3` relative, at ctx 128 and 512;
* the prologue is one command buffer **per layer**, not one per op — the property
  `tests/test_single_command_buffer.rs` proves for the kernel in isolation;
* a hidden size off the float4 geometry (`66`, not a multiple of 4) falls back with
  unchanged logits and zero fused calls.

## Consequence for the benchmark gates

The idle-gap and command-buffer gates (ticket #95) now have a production path to measure, but
the number they will report is "one buffer per layer prologue plus the per-op tail", not "one
buffer per step". Ticket #95 must not calibrate its threshold against the latter until the
debt above is paid.
