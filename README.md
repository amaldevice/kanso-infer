# kanso-infer (簡素)

> **Kanso (簡素)**: Japanese aesthetic of simplicity, eliminating clutter to reveal the core essence.

A minimalist, didactic LLM inference engine written in Rust, engineered from scratch for Apple Silicon (macOS) and consumer hardware.

`kanso-infer` strips away enterprise framework bloat to expose the pure mechanics of modern inference serving: Paged KV Cache, iteration-level continuous batching, prefix caching, OpenAI-compatible streaming API, and Metal compute shaders.

---

## Key Architectural Concepts

### 1. Two-Tier Clean-Seam Design
- **Top Layer (Systems & Scheduling)**: Hardware-agnostic scheduler, paged block memory manager, LRU preemption, and Axum async HTTP server.
- **Bottom Layer (Compute Kernels)**: Pure CPU baseline via `candle-core` for mathematical verification, paired with hand-crafted Metal compute shaders (`.metal`) for Apple Silicon execution.

### 2. Multi-Model Matrix (~350M Compact Class)
Supports dense Transformer decoders and next-generation hybrid linear-attention architectures:
- **SmolLM2-360M** (Hugging Face / Llama architecture: 32 layers, GQA $15:3$, RMSNorm, RoPE, SwiGLU).
- **IBM Granite-4.0-350M (Dense)** (28 layers, GQA $16:4$).
- **IBM Granite-4.0-h-350M (Hybrid Mamba-2)** (4 attention layers + 28 Mamba-2 SSM recurrent layers, achieving 87.5% KV cache memory reduction).
- **Liquid AI LFM2.5-350M (Hybrid LIV Conv)** (6 attention layers + 10 rolling convolution layers).

### 3. Paged Memory & Systems Features
- **Paged KV Cache Block Pool**: Fixed physical blocks ($B_{\text{size}} = 16$ tokens) with reference-counted recycling and virtual-to-physical block table translation.
- **Prefix Caching**: Trie/hash boundary lookup to skip recomputation of shared prompt prefixes.
- **Continuous Batching (Iteration-level Scheduler)**: Dynamic request admission, concurrent prefill/decode sharing forward steps, and automatic block retirement.
- **OpenAI-Compatible HTTP Server**: Axum `/health` and `/v1/chat/completions` endpoints.
- **Custom Apple Silicon Shaders**:
  - `rmsnorm.metal`: Threadgroup parallel reduction for mean of squares.
  - `rope.metal`: Rotary position embedding in rotate-half layout.
  - `silu_gate.metal`: Fused SiLU gate-up multiplication in a single memory pass.
  - `paged_kv.metal`: Paged attention scatter/gather memory layout.
- **Hybrid Memory Manager**: Constant $O(1)$ memory buffer for recurrent SSM/Conv states combined with sparse-layer paged attention.

---

## Project Structure

```text
kanso-infer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── config.rs         # Dynamic GGUF model configurations
│   ├── weights.rs        # Weights structures & RoPE layout reordering
│   ├── model.rs          # Llama forward pass & GQA attention
│   ├── sampler.rs        # Greedy argmax & top-p nucleus sampler
│   ├── kv_cache/         # BlockPool, BlockTable, and PrefixCache
│   ├── scheduler/        # Sequence state machine (Waiting, Running, Finished)
│   ├── engine.rs         # Continuous batching step loop
│   ├── server/           # Axum HTTP routes & OpenAI DTOs
│   ├── metal/            # MSL shaders & CPU numerical parity operators
│   ├── bench.rs          # Synthetic sweeps & throughput profiling
│   └── hybrid/           # RecurrentState (O(1)) & SparseBlockTable
└── tests/                # Comprehensive TDD test suites (15 tests)
```

---

## Quickstart & Testing

Run all unit and integration tests:

```bash
cargo test
```

Expected output:
```text
test result: ok. 15 passed; 0 failed; 0 ignored; finished in 0.08s
```

Run benchmarks:
```bash
cargo test --test test_benchmark_metrics
```
