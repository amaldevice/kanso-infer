# Decode-path workspace decisions (ticket #93, spec #82 D16/D17)

Two adjacent items in the #82 debt register looked like one piece of work and were not.
This file records what was measured, what was changed, and what was deliberately not.

## D17 — `workspace.clear()` per decode: closed, nothing to build

`ScratchpadWorkspace` was a `Vec<f32>`-per-activation scratchpad owned by `Engine`, whose
`clear()` (`fill(0.0)` on every buffer) ran once per decode step. It was removed in
Fase-2 F12 (commit `1dae3d8`, "drop ScratchpadWorkspace") together with the
`forward_*_with_workspace` entry points that consumed it. There is therefore **no
per-decode workspace clearing left to measure** — the debt register entry was stale.

Verification (both must stay empty of call sites):

```sh
rg -n 'workspace\.clear|ScratchpadWorkspace|with_workspace|_workspace' src tests
```

The only matches today are the AMX workspace added by this ticket (`Model::amx_workspace`,
`AmxWorkspace::clear`), and that `clear()` is called from **one** place — `Model::to_device`
— so a workspace never outlives the device its slots were built for. It is not on the
decode path.

The behavioural half of this is a test rather than a claim:
`decode_step_reuses_workspace_buffers_without_clearing_them`
(`tests/test_amx_workspace.rs`) runs two decode steps and asserts the logits buffer keeps
the same address. If a future change reintroduces a per-step clear or rebuild, the address
moves and the test fails.

## D16 — AMX matmul output allocated per call: fixed with an out-param path

`amx_matmul` returned a freshly allocated `[m, n]` `Vec` for every call. A decode step
issues seven projections per layer, so a 2-layer model paid 14 output allocations per
token, plus three scratch buffers per layer inside `amx_paged_decode_attention`.

`AmxWorkspace` (`src/accelerate/amx.rs`) now holds one reusable output tensor per slot plus
the paged-attention scratch buffers. `matmul_weight_into` runs `C = A * W^T` straight into a
slot through an in-place candle op, and `paged_decode_attention_into` reuses the three
scratch buffers. `Model::linear_into` / `project_qk(.., Some(slots))` /
`forward_mlp_block_into` route **only** the decode path (`forward_decode_step` →
`decode_attention_paged`) through them.

Layouts the in-place GEMM cannot read fall back to `amx_matmul`, so the answer never
changes — only the speed does. Quantized weights never reach this path at all (`Model::linear`
only routes dense F32 through AMX), and every non-AMX backend delegates to `linear`.

Measured on a 2-layer mock at ctx 32 (`tests/test_amx_workspace.rs`, AC1/AC2):

| | allocations / decode step | bytes / decode step |
| :--- | ---: | ---: |
| `--backend cpu` | 874 | 182 032 |
| `--backend amx`, before #93 | 776 | 123 392 |
| `--backend amx`, after #93 | 642 | 90 976 |

A single `matmul_weight_into` call now allocates **0** buffers. The test asserts the
reduction is at least the seven projection outputs per layer, in both count and bytes, so
the saving is measured rather than claimed.

Numbers are unchanged (AC4): max |Δlogits| amx vs cpu decode `5.8e-5`, and decode vs the
monolithic forward `6.4e-6` — both inside the repo's `1e-4` parity tolerance.

### Aliasing contract

A tensor returned by a `*_into` call **aliases** the slot it was written to and is only
valid until the next `*_into` call on that same slot. Each live intermediate of the decode
path owns a distinct slot (`AmxWorkspace::SLOT_*`), which is what keeps the reuse safe:
Q, K, V, the attention output, the three MLP outputs, the logits and the paged-attention
output never share a slot within one layer.

## What is still open

- The batched serving path (`forward_batch`, which handles pure-decode batches) and the
  prefill path still use `linear`, so they still allocate per projection. They were left
  alone: #93 scopes the decode path, and `forward_batch` stacks many sequences into one
  GEMM where a per-call output is a much smaller share of the cost.
- `amx_matmul`'s own `get_f32_vec` fallback (the "unrecognized layout" stderr line) is
  untouched. It fires for layouts the in-place GEMM cannot consume; the workspace path
  bypasses it, but the fallback itself still copies when reached from other callers.
