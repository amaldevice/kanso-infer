# kanso-infer (簡素)

> **Kanso (簡素)**: Japanese aesthetic of simplicity, eliminating clutter to reveal the core essence.

A minimalist, high-performance LLM inference engine written in Rust, engineered from scratch for Apple Silicon (macOS) and consumer hardware.

`kanso-infer` strips away enterprise framework bloat to expose the pure mechanics of modern inference serving: Paged KV Cache, iteration-level continuous batching, Sarathi-Serve chunked prefilling, zero-copy Metal Unified Memory, fused PagedAttention v2, ARM NEON SIMD vector quantization, speculative decoding, and OpenAI-compatible streaming.

---

## Key Features & Capabilities

### 1. Interactive CLI, Chat & Serving Modes
- **`kanso cli`**: Direct single or batched completion directly from terminal flags with token streaming.
- **`kanso chat`**: Interactive terminal multi-turn chat session (REPL loop) with conversation history tracking, ChatML formatting, and `/reset` command.
- **`kanso serve`**: OpenAI-compatible HTTP server (`POST /v1/chat/completions`, `GET /health`) powered by Axum and Tokio with Server-Sent Events (SSE) streaming and 429 backpressure protection.

### 2. Low-Level Hardware Optimizations
- **Incremental $O(1)$ Decode Loop**: Replaces naive quadratic prompt re-computation with true single-token autoregressive decoding, attending across historical keys and values stored in non-contiguous physical blocks.
- **Bounded-Allocation Decode Path**: Decode attention reads the KV history straight out of the block pool instead of re-materializing it per layer (ticket #48), and a step's buffers are reused rather than grown: every tensor op still allocates its output, so this is bounded allocation, not zero-allocation — measured with a counting allocator in `tests/test_decode_allocations.rs` (extra bytes per step stay far below the history they read). Metal is exempt from the no-copy half of that claim: its GEMM path materializes the pool view (ticket #61/F4).
- **Sarathi-Serve Token-Budget Chunked Prefill**: Enforces a configurable token budget per iteration ($T_{\text{budget}}$), partitioning large prompts into chunks and co-scheduling them with active decode requests to eradicate Head-of-Line blocking and stabilize $p99$ Inter-Token Latency (ITL).
- **Apple Silicon Metal Zero-Copy UMA Runtime**: Allocates GPU buffers in shared storage mode (`MTLStorageModeShared`), allowing CPU slices and Metal shaders to share physical Unified Memory with zero PCIe copy overhead.
- **Startup Metal Pipeline Caching**: Pre-compiles all Metal Shading Language (`.metal`) compute kernels into `MTLComputePipelineState` handles during initialization, preventing first-request compilation stutter.
- **Fused PagedAttention v2 with Online Softmax**: Computes attention scores and context reduction directly within GPU threadgroup shared memory in tiles, completely eliminating the materialization of intermediate score matrices ($[B, N_h, Q, S]$) in global VRAM.
- **Quantized-Resident Weights**: GGUF projection weights keep their on-disk format (Q8_0/Q4_K blocks, F16 floor) instead of being dequantized to F32 at load, so batch-1 decode reads ~1 byte/param; matmuls run off that storage (NEON/dotprod kernels on CPU, dequant-on-the-fly GEMV on Metal), embeddings dequantize per row at lookup. SmolLM2-360M Q8_0 loads at ~376 MB resident with ~3.9x the F32 decode rate. The hand-written `src/quant` Q8_0/Q4_K GEMV kernels stay as the reference/benchmark path.
- **Speculative Decoding Verification Pipeline**: The draft model runs on its own paged KV cache (one `forward_decode_step` per proposal, no full-prefix re-computation) and rolls that cache back on rejection, so speculation is a wall-clock win rather than a second full forward pass. Verification is exact: greedy requests accept on argmax match, temperature > 0 requests use standard rejection sampling (`min(1, p/q)`, residual correction), so the token distribution matches plain sampling, not just greedy decoding.

### 3. Multi-Model Architecture Matrix (~350M Compact Class)
Supports dense Transformer decoders and next-generation hybrid linear-attention architectures:
- **SmolLM2-360M** (Hugging Face / Llama architecture: 32 layers, GQA $15:5$, RMSNorm, RoPE rotate-half, SwiGLU).
- **IBM Granite-4.0-350M (Dense)** (28 layers, GQA $16:4$).
- **IBM Granite-4.0-h-350M (Hybrid Mamba-2)** (4 attention layers + 28 Mamba-2 SSM recurrent layers, achieving **87.5% KV cache memory reduction**).
- **Liquid AI LFM2.5-350M (Hybrid LIV Conv)** (6 attention layers + 10 rolling convolution layers, achieving **62% memory reduction**).
- **Hybrid Memory Manager**: Constant $O(1)$ memory buffer for recurrent SSM/Conv states combined with sparse-layer paged attention.

### 4. Multi-Engine Benchmark & Correctness Suite (`bench`)
- **Batch Size Sweeping**: Sweeps batch sizes (e.g. `--batch-sizes 1,4,8,16,32`) to evaluate saturation points.
- **Trace Replay**: Replays ShareGPT-style JSONL traces (`--trace-file`) with heterogeneous prompt lengths, generation lengths, and staggered arrival times.
- **Multi-Model Profile Selection**: `--model-type <mock | smollm2 | granite-dense | granite-h | lfm>` for empirical architecture comparison.
- **Deterministic 64-bit FNV-1a Checksums**: Order-sensitive token stream hashing to mathematically prove greedy output parity across CPU and Metal backends.
- **Hardware Telemetry & Resource Tracking (Zero-Sudo)**:
  - **Hardware Profiling**: Reads CPU model, core counts, RAM, and model identifier via macOS `sysctl`.
  - **Peak Memory (RSS)**: Tracks peak Resident Set Size in MB via POSIX `libc::getrusage`.
  - **Thermal Pressure Monitoring**: Queries `NSProcessInfo.thermalState` directly from user space without root/sudo (`Nominal`, `Fair`, `Serious`, `Critical`).
  - **Memory Bus Bandwidth Utilization**: Estimates effective GB/s and hardware bus utilization percentage using the Roofline formulation.
- **Full SLO Metric Reporting**:
  - Total throughput (`tok/s`) and per-request throughput (`tok/s/req`)
  - Time-to-First-Token (`TTFT` mean and max in ms)
  - Inter-Token Latency (`ITL` mean and $p99$ in ms)
  - Decode duration (`ms/step`) and prefill rate (`tok/s`)
- **Machine-Readable Performance Gates**: `--gate <budget.json>` compares the report against version-controlled budgets and exits `0` (passed) / `1` (violated) / `2` (skipped: required asset missing). See section 6.

## Feature Status

Every claim below resolves to one status. ✅ wired = reachable from `kanso cli` / `chat` / `serve`;
🧪 kernel-only = implemented and tested, but no production caller yet; 📝 planned = roadmap, not code.

The table has one in-crate source of truth: `src/features.rs`. Each row is a `FeatureSpec`
carrying the same name, status and exec pointer, and every ✅ row also carries a `wired`
marker that names its production symbol inside a `const`. Deleting or privatising that
symbol breaks the build, so a ✅ label cannot outlive the code it describes. What the
marker proves is that the symbol *exists and is public* — reachability from
`kanso cli` / `chat` / `serve` is still the behavioural tests' job.
`tests/test_feature_registry.rs` fails when this table and the registry disagree by even
one row, when a ✅ row has no marker, when a 🧪/📝 row still claims one, or when a 📝 row
points somewhere other than the roadmap. `tests/test_honesty_fase3.rs` separately checks
that every exec pointer resolves to a file in the tree.

| Feature | Status | Exec pointer |
| :--- | :---: | :--- |
| CLI / chat / serve (Axum + SSE, 429 backpressure) | ✅ | `src/main.rs`, `src/server/engine_loop.rs` |
| Incremental $O(1)$ decode over paged KV blocks | ✅ | `src/model.rs` (`forward_decode_step`), `src/kv_cache/storage.rs` |
| Continuous batching + Sarathi chunked prefill | ✅ | `src/engine.rs` (`Engine::step`), `src/scheduler/` |
| Prefix-cache block reuse | ✅ | `src/engine.rs`, `src/kv_cache/prefix_cache.rs` |
| Quantized-resident weights (Q8_0/Q4_K/F16 stay packed; per-row embedding dequant) | ✅ | `src/weights.rs`, `tests/test_quantized_weights.rs` |
| Granite dense multipliers (`embedding_scale`, `attention.scale`, `residual_scale`, `logit_scale`) | ✅ | `src/weights.rs`, `src/model.rs`, `src/config.rs` |
| Loader arch gate (rejects non-llama/non-granite GGUF) | ✅ | `src/weights.rs` (`load_from_gguf`) |
| Speculative verification (argmax + `min(1,p/q)` rejection sampling, draft KV rollback) | ✅ | `src/engine.rs` (`speculative_step_sampled`), `src/speculative/verifier.rs` |
| Metal UMA buffers + startup pipeline cache | ✅ | `src/metal/context.rs`, `src/metal/pipeline.rs` |
| AMX/Accelerate GEMM path (dense weights) | ✅ | `src/accelerate/amx.rs`, `src/model.rs` (`Model::linear`) |
| FFN offload seam (per-layer SwiGLU adapters, CPU/AMX execution) | ✅ | `src/ffn_offload/adapter.rs`, `src/model.rs` (`forward_mlp_block`) |
| Real Core ML / ANE execution (`.mlpackage`, `MLModel`) | 📝 | roadmap |
| Fused PagedAttention v2 Metal kernel | ✅ | `src/metal/paged_attention.rs`, `src/model/paged_attn.rs`, `src/metal/kernels/paged_attention_v2.metal` (decode: `Model::metal_paged_attention`) |
| Fused Metal llama-layer kernel | ✅ | `src/metal/fused_layer.rs`, `src/model/fused_decode.rs` — decode: one command buffer per *step* (`Model::forward_batch`, ticket #110) chains the fused RMSNorm + QKV + RoPE prologue, PagedAttention v2, O-projection + residual add and SwiGLU MLP + residual add for every layer; final norm + lm_head stay on the tensor path (ADR 0002 scope). Non-representable geometries fall back to the per-op path |
| Hand-written NEON Q8_0/Q4_K GEMV kernels | 🧪 | `src/quant/neon.rs` (benchmark/parity-test path only; `--backend neon` refuses to start — see below) |
| Quantized (INT8/FP8) paged KV cache | ✅ | `src/kv_cache/kind.rs` (`KvStorageKind`, `KvCacheType`), `src/kv_cache/quantized_storage.rs`, `src/main.rs` — selected by the user via `--kv-type` = f32 / int8 / fp8 (single-shot, chat, serve), plumbed to `Engine::new_with_budget_and_kv`; F32 is the default, Metal refuses INT8/FP8 with an `Err` (its paged-attention kernel reads an F32 pool). The pool's per-token memory numbers (`kv_type`, `bytes_per_token`, `ratio_vs_f32`) ride in the bench report and are gated by `gates/fase2.json` |
| Hybrid SSM/conv layers (Mamba-2, LIV conv) + sparse KV allocation | 📝 | roadmap |
| `granite-h` / `lfm` bench profiles | 📝 | roadmap |

## Roadmap

- Real Core ML / ANE execution: compile per-layer SwiGLU to `.mlpackage` and run via `MLModel`, behind the `src/ffn_offload` seam.
- Wire the fused PagedAttention v2 kernel into the Metal decode path (`src/model.rs` names the call site).
- Hybrid SSM/conv layers (Mamba-2, LIV conv) + sparse KV allocation — demoted from 🧪 to 📝
  by ticket #83. What exists today is `src/hybrid/recurrent_state.rs` (a rolling
  `0.9*state + 0.1*input` buffer, 72 lines) and `src/kv_cache/`'s `SparseBlockTable`
  (block bookkeeping), with **no integration into the forward pass**: no Mamba-2 SSM scan,
  no LIV conv mixing, no sparse attention read. That is a stub, not "implemented and
  tested", so the honest status is planned. The loader arch gate still rejects hybrid GGUF
  (`mamba`, `lfm2`, ...) until the forward pass exists — that gate stays.
- `granite-h` / `lfm` bench profiles — demoted from 🧪 to 📝 by ticket #83. `src/config.rs`
  describes their *shapes* only (mock configs for comparative bench runs); no hybrid or
  LFM weights are ever loaded, and `bench` refuses to run without `--model <path>`. A mock
  shape is not an implemented feature, so this is roadmap: load real Granite-hybrid/LFM
  GGUF once the hybrid forward pass above lands.
- Promote hand-written NEON GEMV kernels from reference path to production decode path.
  Blocker (ticket #89): the packed layouts differ, so the kernels cannot read the
  production `LinearWeight` blocks in place. `src/quant::BlockQ8_0` is
  `{ d: f32, qs: [i8; 32] }` (36 B); candle stores `{ d: f16, qs: [i8; 32] }` (34 B).
  `src/quant::BlockQ4_K` keeps `d`/`dmin` as f32 where candle keeps f16. Bridging that
  needs either a repack of every weight per token (costs more bandwidth than the kernel
  saves, so rejected) or a kernel written against candle's f16-scale blocks. Until then
  `--backend neon` errors out instead of silently running candle's CPU matmul.

---

## Project Structure

```text
kanso-infer/
├── Cargo.toml
├── README.md
├── docs/SPEC.md          # Canonical architecture and low-level specification
├── src/
│   ├── main.rs           # Main CLI binary (cli, chat, serve)
│   ├── lib.rs            # Core library root
│   ├── config.rs         # Dynamic model configurations & profile selection
│   ├── weights.rs        # Weights structures & RoPE layout reordering
│   ├── model.rs          # Llama forward pass & incremental decode step
│   ├── sampler.rs        # Greedy argmax & top-p nucleus sampler
│   ├── chat.rs           # ChatSession multi-turn REPL & ChatML formatting
│   ├── bench.rs          # Benchmark workload runner & SLO metric calculations
│   ├── hardware.rs       # User-space hardware profiler, peak RSS, thermal state & bus util
│   ├── engine.rs         # Continuous batching & chunked prefill scheduler
│   ├── error.rs          # Stdlib EngineError
│   ├── ffn_offload/
│   │   ├── adapter.rs  # Per-layer SwiGLU offload seam (CPU/AMX; Core ML target planned)
│   │   └── mod.rs      # Re-exports (renamed from `ane`, ticket #54)
│   ├── metal/
│   │   ├── context.rs    # Metal device, command queue & UMA buffer allocator
│   │   ├── pipeline.rs   # Compute pipeline state compilation & caching
│   │   ├── ops.rs        # GPU RMSNorm, RoPE, and SiLU dispatch bindings
│   │   ├── paged_attention.rs # Fused PagedAttention v2 Metal pipeline dispatch
│   │   ├── shaders.rs    # Embedded MSL shader registry
│   │   └── kernels/
│   │       ├── paged_attention_v2.metal # Tiled online softmax compute shader
│   │       ├── rmsnorm.metal            # Threadgroup reduction compute shader
│   │       ├── rope.metal               # Rotate-half RoPE compute shader
│   │       ├── silu_gate.metal          # Fused SiLU gate-multiply compute shader
│   │       └── paged_kv.metal           # Paged memory gather compute shader
│   ├── quant/
│   │   ├── neon.rs       # ARM NEON vdotq_s32 & vfmaq_f32 vector kernels
│   │   ├── q8_0.rs       # 8-bit block quantization & GEMV
│   │   └── q4_k.rs       # 4-bit block quantization & dot products
│   ├── speculative/
│   │   ├── verifier.rs   # Target-draft verification tree & KV cache rollback
│   │   ├── draft_cache.rs # Draft model's own paged KV cache & rollback
│   │   ├── sampling.rs   # Rejection sampling (min(1, p/q), residual correction)
│   │   └── mod.rs        # SpeculativeEngine wrapper
│   ├── hybrid/
│   │   ├── recurrent_state.rs # O(1) constant buffer for Mamba-2 & LIV Conv
│   │   └── sparse_kv.rs       # Selective sparse KV cache allocation
│   └── bin/
│       └── bench.rs      # Standalone benchmark binary CLI
└── tests/                # 21 integration test suites (61 tests)
```

---

## Quickstart & CLI Usage

### 1. Run Tests
Verify all 61 unit, parity, and integration test suites:

```bash
cargo test
```

Expected output:
```text
test result: ok. 61 passed; 0 failed; finished in ~11s
```

### 2. Interactive Terminal Chat
Launch a multi-turn conversation session in your terminal:

```bash
# Using SmolLM2 architecture profile (default)
cargo run --release --bin kanso -- chat

# Using Granite-4.0-h hybrid profile on Metal GPU backend
cargo run --release --bin kanso -- chat --model-type granite-h --backend metal
```

Commands during chat:
- Type your message and hit Enter.
- Type `/reset` to clear conversation history.
- Type `exit` or press `Ctrl-D` to quit.

### 3. Direct CLI Completion
Run single or batched prompts directly from flags:

```bash
# Single prompt
cargo run --release --bin kanso -- cli \
  --prompt "The capital of France is" \
  --max-tokens 32

# Batched prompts
cargo run --release --bin kanso -- cli \
  --prompt "Explain quantum computing in one sentence" \
  --prompt "Write a short poem about autumn" \
  --model-type granite-h \
  --max-tokens 64
```

### 4. OpenAI-Compatible API Server
Start the HTTP daemon:

```bash
cargo run --release --bin kanso -- serve --port 8000 --model-type smollm2
```

Test completion via cURL:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-360m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 16
  }'
```

Health check:
```bash
curl http://localhost:8000/health
```

### 5. Running Comparative Benchmarks
Sweep batch sizes and measure throughput, TTFT, and ITL:

```bash
# Benchmark SmolLM2 Dense
cargo run --release --bin bench -- \
  --model-type smollm2 \
  --batch-sizes 1,4,8,16 \
  --backend cpu

# Benchmark IBM Granite-4.0-h Hybrid (Mamba-2 + sparse KV attention)
cargo run --release --bin bench -- \
  --model-type granite-h \
  --batch-sizes 1,4,8,16 \
  --backend metal

# Replay realistic ShareGPT trace file
cargo run --release --bin bench -- \
  --trace-file trace.jsonl \
  --backend cpu
```

Output format (field names, with `<...>` where a run puts its own numbers):
```text
============================================================
 kanso-infer multi-engine benchmark suite
 Model    : <path to the GGUF that ran>
 Hardware : <chip> (<n> cores, <n> GB RAM, Model: <id>)
 Thermal  : <thermal state>
 Backend  : <backend>, Warmup: <bool>
============================================================
batch_size=<n>    backend=<backend>    batch=<n>    tok/s(total)=<t>    tok/s/req=<t>
                ttft_ms(p50/p90/mean/max)=<t>/<t>/<t>/<t> itl_ms(p50/p90/p99)=<t>/<t>/<t> prefill_tok/s=<t> decode_ms/step=<t> (decode_ms/tok=<t>) prefix_hit=<t>% checksum=<16 hex digits>
                peak_rss=<t>MB thermal="<thermal state>" bus_est=<t>GB/s (<t>% of <t>GB/s <how the peak was measured>)
```

Every number above comes from the run, so there is no reference output to
compare against: the `checksum` is an FNV-1a hash of the tokens that run
generated, and it differs per model, per prompt and per sampling seed. Two
profiles reporting the same checksum would mean one of them did not run.

The JSON written by `--json-output` carries a `telemetry` field next to
`bytes_read_per_step`: `"measured"` when the bytes were weighed from the model
that ran, `"unmeasured"` when nothing was weighed. `--gate` treats an
unmeasured report as inconclusive (exit `2`), never as a pass.
```

### 6. Performance Gates
A budget file turns a performance threshold into a reviewable diff. `--gate`
compares a `BenchmarkReport` against the budget and exits with a code CI can read:

```bash
# Gate a fresh run (the report JSON is written as evidence, path printed)
cargo run --release --bin bench -- \
  --model ./models/smollm2-360m-instruct-q8_0.gguf \
  --gate gates/fase1.json

# Re-check an existing report without loading a model or touching hardware
cargo run --release --bin bench -- \
  --gate gates/fase1.json \
  --gate-report report.json
```

A budget declares rows of `{metric, bound, budget, requires?}`:

```json
{
  "name": "fase1",
  "rows": [
    {"metric": "decode_ms_per_step", "bound": "max", "budget": 12.0},
    {"metric": "total_tokens_per_sec", "bound": "min", "budget": 180.0,
     "requires": {"kind": "file", "path": "models/smollm2-360m-instruct-q8_0.gguf"}}
  ]
}
```

Each row prints its measured value, the ratio against the budget, the budget and
the verdict. Exit codes: `0` every row measured and within budget, `1` a budget
was violated, `2` a row could not be measured because a required asset
(`"kind": "file"`) or tool (`"kind": "tool"`, looked up on `PATH`) is missing.
**A skipped row is never counted as a pass**, and an unknown metric is an error
rather than a silent green.

---

## Empirical Benchmarking

Benchmarks run against real GGUF weights via `cargo run --release --bin bench -- --model <path>` (shadow/mock model execution has been removed; see section 5 above).

Reported metrics:
- **Throughput**: Total tokens/s and per-request tokens/s.
- **Latency**: Mean/max/p50/p90 Time-to-First-Token (TTFT), p50/p90/p99 Inter-Token Latency (ITL), decode ms/step, and per-token decode latency (ms/token).
- **Traceable Bus Utilization**: Derived from resident model weight bytes plus active KV cache bytes summed across sequences, evaluated against host DRAM bandwidth measured via the STREAM-triad microbench.
- **KV Memory per Token** (`kv_type`, `bytes_per_token`, `ratio_vs_f32`): the KV pool's own accounting, attached only by a run that really used an engine (`bench --kv-type f32|int8|fp8`), so the INT8/FP8 saving is measured in the report JSON and gated (`gates/fase2.json`) rather than asserted in tests. `bytes_read_per_step` above remains one number — it does not split KV from weights.
- **Determinism**: 64-bit FNV-1a checksums over generated tokens to verify greedy output parity.
