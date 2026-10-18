## Problem Statement

When running local LLM inference on modern consumer hardware—specifically Apple Silicon Macs—developers and ML researchers face a frustrating tradeoff:
1. Production frameworks (like vLLM or TensorRT-LLM) are heavily coupled to NVIDIA CUDA/Linux ecosystems, carrying massive dependency bloat, complex abstractions, and inability to run bare-metal custom kernels natively on macOS Unified Memory.
2. Lightweight educational engines often make naive architectural choices: they either execute decoding via quadratic $O(N^2)$ prompt re-computation, materialize gigantic attention score matrices ($[B, N_h, Q, S]$) in global RAM, fail to leverage ARM NEON SIMD vector hardware for on-the-fly quantization, or dequantize GGUF weights to uncompressed FP32/FP16, hitting the physical memory bandwidth wall and suffering severe generation stalls.
3. Monolithic prompt prefilling causes severe Head-of-Line (HoL) blocking and latency jitter (spikes in Time-to-First-Token and Inter-Token Latency) when serving concurrent streaming users.

Developers need an ultra-lean, verifiable inference engine in Rust that matches production throughput through low-level hardware co-design (Apple Silicon Metal zero-copy UMA, ARM NEON SIMD GEMV, fused PagedAttention v2, chunked prefill, and speculative decoding) without sacrificing code clarity or educational transparency.

---

## Solution

Transform `kanso-infer` into an uncompromising high-performance inference engine that operates directly against bare-metal hardware constraints on macOS Apple Silicon and ARM64 architecture:
1. **True Incremental $O(1)$ Decode Loop**: Replace quadratic prompt recomputation with single-token forward steps that attend over historical Key-Value states stored in non-contiguous physical memory blocks.
2. **Zero-Copy Metal Compute Pipeline**: Bind custom Metal Shading Language (MSL) kernels directly to Apple Silicon Unified Memory using shared storage mode (`MTLStorageModeShared`), with pipeline state compilation caching to eliminate runtime driver overhead.
3. **Fused PagedAttention v2 with Online Softmax**: Compute attention scores and context reduction directly within GPU threadgroup shared memory in tiles, completely eliminating the materialization of intermediate score matrices in global DRAM/VRAM.
4. **ARM NEON SIMD Vector Acceleration**: Utilize AArch64 vector intrinsics (`vdotq_s32`, `vfmaq_f32`) for on-the-fly dequantization of Q8_0 and Q4_K weights in CPU registers during decode GEMV.
5. **Token-Budget Chunked Prefill Scheduler**: Co-schedule small prompt chunks alongside decode steps under a fixed maximum token budget per iteration, eliminating TTFT spikes and ensuring steady streaming generation.
6. **Speculative Decoding Verification Pipeline**: Implement target-draft speculative execution with verifiable KV block rollback to break past the autoregressive memory bandwidth limit on small batch sizes.
7. **Bounded-Allocation Decode Path**: Bound the *extra* bytes a decode step allocates instead of promising zero allocations: the step must not re-materialize the KV history (it reads its context as a view of the block pool, ticket #48), and the buffers it does allocate per step must stay well under a quarter of the history it reads. `tests/test_decode_allocations.rs` is the gate.

---

## User Stories

1. As an inference engineer, I want the engine to execute decode steps incrementally in $O(1)$ time, so that autoregressive generation latency does not degrade quadratically as sequence length grows.
2. As an Apple Silicon user, I want GPU memory buffers to use shared Unified Memory without PCIe copy overhead, so that token activations and KV blocks pass between CPU and Metal shaders with zero copy latency.
3. As a systems programmer, I want Metal compute pipeline states to be compiled once and cached at startup, so that client requests do not suffer compilation jitter during initial forward passes.
4. As an inference engineer, I want attention scores to be normalized via online softmax inside GPU threadgroups, so that the engine never allocates or writes intermediate attention score matrices to global VRAM.
5. As a developer deploying on Mac CPUs, I want on-the-fly Q8_0 and Q4_K matrix-vector multiplications using ARM NEON vector instructions, so that memory bandwidth consumption is reduced up to 4x compared to FP32 without unpacking models into RAM.
6. As a client streaming tokens via SSE, I want long prompt processing to be broken into chunked prefill batches, so that my active generation is not frozen by another user submitting a large document.
7. As an API client, I want steady Inter-Token Latency under multi-user concurrency, so that streaming responses feel smooth and uninterrupted.
8. As a performance researcher, I want speculative decoding with draft-tree verification, so that single-request generation throughput can exceed the theoretical physical memory bandwidth limit of the model.
9. As an inference engineer, I want the speculative decoding pipeline to support atomic KV cache rollback upon token rejection, so that invalid speculative paths never pollute the block pool.
10. As a performance engineer, I want the inner decode loop to reuse its buffers instead of re-materializing the KV history every step, so that per-step *extra* allocation stays bounded (well under a quarter of the history it reads) and allocator tail latency cannot dominate a token.
11. As a systems researcher, I want an end-to-end 64-bit FNV-1a checksum verification harness, so that I can mathematically prove greedy output parity across CPU, NEON SIMD, and Metal backends.
12. As a benchmark operator, I want automated ShareGPT trace replay capabilities, so that I can evaluate continuous batching throughput under realistic prompt/generation length distributions.
13. As an engineer evaluating engines, I want standardized Time-to-First-Token (TTFT) and Inter-Token Latency (ITL) metrics reporting, so that I can directly compare `kanso-infer` against `rvllm` and `llama.cpp`.
14. As an API consumer, I want full OpenAI-compatible `/v1/chat/completions` protocol compliance with Server-Sent Events, so that standard OpenAI SDKs and agent frameworks connect out-of-the-box.
15. As a server operator, I want strict FIFO queuing and HTTP 429 backpressure handling, so that engine memory exhaustion does not cause ungraceful crashes during traffic spikes.
16. As an engineer working with hybrid models (Granite-4.0-h, LFM2.5), I want sparse-layer paged memory allocation, so that non-attention layers consume only $O(1)$ fixed state buffers rather than allocating unnecessary KV blocks.
17. As an ML researcher, I want rotary position embeddings (RoPE) to support dynamic frequency scaling and rotate-half permutations, so that long-context evaluations remain accurate.
18. As a developer writing tests, I want deterministically seeded sampling parameters, so that test suites reliably verify argmax and top-p behaviors across platforms.
19. As an edge device operator, I want graceful fallback to ARM NEON CPU SIMD when a GPU device is unavailable, so that the engine runs everywhere without platform lock-in.
20. As a code maintainer, I want all components to maintain clean public seams and zero unrequested abstractions, so that the codebase remains pedagogical and easily maintainable.

---

## Implementation Decisions

### 1. Two-Tier Modular Architecture
- Decouple the top scheduling and memory management layer completely from device-specific compute backends.
- The top layer operates purely on logical indices, sequence state transitions, token buffers, and virtual block IDs.
- The bottom layer consumes low-level batches and dispatches either to:
  - Apple Silicon Metal Unified Memory backend (primary macOS accelerator).
  - ARM64 NEON SIMD backend (CPU vector path).
  - Reference scalar CPU backend (formal verification baseline).

### 2. Incremental Decode Seam & Block Reading
- Split forward execution into explicit `prefill_chunk` and `decode_step` interfaces.
- The `decode_step` interface receives a batch of active sequences, each contributing exactly one new token ID, along with their mapped physical block indices and current context positions.
- The model computes single-token projections and attends across the historical context directly from physical block memory.

### 3. Fused PagedAttention v2 Compute Kernel
- Attention is computed using threadgroup-level online softmax tiling adapted from FlashAttention-2 and PagedAttention v2.
- The query vector ($Q$) resides in registers; key ($K$) and value ($V$) vectors are gathered tile-by-tile from physical memory blocks into threadgroup memory.
- Running maximums and running normalizers maintain numerical stability in registers without materializing scores in VRAM.

### 4. Zero-Copy Apple Silicon Unified Memory
- All persistent weight buffers and paged KV cache buffers use `MTLStorageModeShared`.
- Metal command buffers and compute command encoders are managed with pre-compiled `MTLComputePipelineState` handles cached in a thread-safe registry.
- Rust slices interact with Metal buffer memory via raw pointers obtained from buffer contents without intermediate IPC or system copies.

### 5. ARM NEON SIMD Intrinsic Vector Kernels
- Implement vectorized GEMV kernels for Q8_0 and Q4_K quantizations in `std::arch::aarch64`.
- Use `vdotq_s32` for 4-way int8 dot products and `vfmaq_f32` for block scale accumulation.
- Weights remain in quantized integer representation until loaded into 128-bit NEON vector registers.

### 6. Sarathi-Serve Chunked Prefill Scheduler
- Implement iteration-level budget scheduling: each engine step is constrained by a maximum token budget (e.g., $T_{\text{budget}} = 512$).
- If active decode requests require $B_{\text{decode}}$ tokens, the remaining budget ($T_{\text{budget}} - B_{\text{decode}}$) is allocated to the earliest waiting prefill request as a partial chunk.
- Large prompts are split across multiple consecutive iterations, eliminating Head-of-Line blocking.

### 7. Speculative Decoding Verification Pipeline
- Support target-draft model speculative loops.
- The small draft model speculates $K$ tokens.
- The target model executes a single verification step over the speculated token tree.
- Accepted tokens are committed; upon the first rejected token, unverified KV blocks are rolled back via atomic block table truncation.

### 8. Static Scratchpad Workspace Allocation
- Eliminate dynamic heap allocations in the inner inference loop.
- Allocate a contiguous workspace memory buffer during engine initialization sized to the maximum batch size and hidden dimensionality.
- Intermediate activations borrow pre-allocated slices within this workspace.

> **Status:** the engine-wide `ScratchpadWorkspace` this section described was removed in
> Fase-2 F12 — a pre-zeroed scratchpad that had to be cleared every step traded
> allocations for `fill()` traffic and an aliasing hazard. What exists instead is the AMX
> decode workspace (`AmxWorkspace`, ticket #93): reused matmul output slots, never cleared
> per step, with a written aliasing contract. See `docs/decode-workspace.md`.

---

## Testing Decisions

### What Makes a Good Test
- Tests must verify observable system behavior through high-level public seams rather than private implementation details.
- Tests must use deterministic inputs and independent sources of truth (exact mathematical formulas, known-good token streams, or worked examples).
- Tests must verify numerical parity: custom Metal and NEON kernels must match the scalar CPU reference path within strict numerical tolerances ($\le 10^{-4}$ for activations, identical 64-bit FNV-1a hash for token streams).

### Tested Seams & Modules
- **Primary Engine Seam**: `Engine::step()` and `Engine::add_request()`. Tests verify continuous batching, chunked prefill admission, token emission, and block pool recovery.
- **Compute Backend Seam**: `Backend::forward_decode()` and `Backend::forward_prefill()`. Tests verify parity between Metal shaders, NEON SIMD, and CPU references.
- **Memory Allocation Seam**: `BlockPool`, `BlockTable`, and `PrefixCache`. Tests verify block ref-counting, Copy-on-Write prefix sharing, and preemption eviction.
- **Benchmark & Correctness Seam**: `BenchmarkReport` and `FNV-1a checksum`. Tests verify steady-state throughput calculations and deterministic greedy reproduction.

### Prior Art
- Unit and parity tests in `inference-engineering/from_scratch/tests/` (`test_forward_parity.rs`, `test_metal_kernel_parity.rs`, `test_paged_kv_cache.rs`).
- Golden reference tests in `rvllm` (`src/bin/bench.rs` checksum validation).

---

## Out of Scope

- Multi-node distributed inference and multi-GPU tensor parallelism (focus is single-device consumer hardware / Apple Silicon).
- Dynamic runtime model fine-tuning or LoRA adapter training.
- Proprietary closed-source quantization formats (AWQ/GPTQ specific to NVIDIA CUDA).
- Continuous speculative drafting with multi-head speculative speculative decoders (Medusa/EAGLE head training).

---

## Further Notes

- The project prioritizes zero unnecessary abstractions ("Ponytail" principle). Every kernel optimization directly serves a verifiable hardware bottleneck.
- Implementations must maintain dual-mode support for dense Transformer decoders (SmolLM2-360M, Granite-4.0 Dense) and hybrid linear-attention models (Granite-4.0-h, LFM2.5).
