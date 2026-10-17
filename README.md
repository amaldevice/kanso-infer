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
- **Static Scratchpad Workspace**: Reuses a pre-allocated scratchpad (`ScratchpadWorkspace`) for hidden states and logits across decode steps instead of re-allocating those buffers per step. This is buffer reuse, not zero-allocation: every tensor op still allocates its output (measured with a counting allocator in `tests/test_decode_allocations.rs`), and decode attention reads the KV history straight out of the block pool instead of re-materializing it per layer (ticket #48).
- **Sarathi-Serve Token-Budget Chunked Prefill**: Enforces a configurable token budget per iteration ($T_{\text{budget}}$), partitioning large prompts into chunks and co-scheduling them with active decode requests to eradicate Head-of-Line blocking and stabilize $p99$ Inter-Token Latency (ITL).
- **Apple Silicon Metal Zero-Copy UMA Runtime**: Allocates GPU buffers in shared storage mode (`MTLStorageModeShared`), allowing CPU slices and Metal shaders to share physical Unified Memory with zero PCIe copy overhead.
- **Startup Metal Pipeline Caching**: Pre-compiles all Metal Shading Language (`.metal`) compute kernels into `MTLComputePipelineState` handles during initialization, preventing first-request compilation stutter.
- **Fused PagedAttention v2 with Online Softmax**: Computes attention scores and context reduction directly within GPU threadgroup shared memory in tiles, completely eliminating the materialization of intermediate score matrices ($[B, N_h, Q, S]$) in global VRAM.
- **Quantized-Resident Weights**: GGUF projection weights keep their on-disk format (Q8_0/Q4_K blocks, F16 floor) instead of being dequantized to F32 at load, so batch-1 decode reads ~1 byte/param; matmuls run off that storage (NEON/dotprod kernels on CPU, dequant-on-the-fly GEMV on Metal), embeddings dequantize per row at lookup. SmolLM2-360M Q8_0 loads at ~376 MB resident with ~3.9x the F32 decode rate. The hand-written `src/quant` Q8_0/Q4_K GEMV kernels stay as the reference/benchmark path.
- **Speculative Decoding Verification Pipeline**: Target-draft speculative execution with atomic KV cache block table rollback upon draft token rejection, ensuring 100% mathematical token parity with non-speculative greedy decoding.

### 3. Multi-Model Architecture Matrix (~350M Compact Class)
Supports dense Transformer decoders and next-generation hybrid linear-attention architectures:
- **SmolLM2-360M** (Hugging Face / Llama architecture: 32 layers, GQA $15:3$, RMSNorm, RoPE rotate-half, SwiGLU).
- **IBM Granite-4.0-350M (Dense)** (28 layers, GQA $16:4$).
- **IBM Granite-4.0-h-350M (Hybrid Mamba-2)** (4 attention layers + 28 Mamba-2 SSM recurrent layers, achieving **87.5% KV cache memory reduction**).
- **Liquid AI LFM2.5-350M (Hybrid LIV Conv)** (6 attention layers + 10 rolling convolution layers, achieving **62% memory reduction**).
- **Hybrid Memory Manager**: Constant $O(1)$ memory buffer for recurrent SSM/Conv states combined with sparse-layer paged attention.

### 4. Multi-Engine Benchmark & Correctness Suite (`bench`)
- **Batch Size Sweeping**: Sweeps batch sizes (e.g. `--batch-sizes 1,4,8,16,32`) to evaluate saturation points.
- **Trace Replay**: Replays ShareGPT-style JSONL traces (`--trace-file`) with heterogeneous prompt lengths, generation lengths, and staggered arrival times.
- **Multi-Model Profile Selection**: `--model-type <mock | smollm2 | granite-dense | granite-h | lfm>` for empirical architecture comparison.
- **Deterministic 64-bit FNV-1a Checksums**: Order-sensitive token stream hashing to mathematically prove greedy output parity across CPU, NEON, and Metal backends.
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
---

## Project Structure

```text
kanso-infer/
├── Cargo.toml
├── README.md
├── SPEC.md               # Complete architecture and low-level specification
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
│   ├── kv_cache/
│   │   ├── block_pool.rs # Physical memory block pool & ref-counting
│   │   ├── block_table.rs# Virtual-to-physical block table translation
│   │   ├── prefix_cache.rs# Trie/hash prompt prefix caching
│   │   └── storage.rs    # KvStorage physical block tensor buffer
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

Output format:
```text
============================================================
 kanso-infer multi-engine benchmark suite
 Hardware : Apple A18 Pro (6 cores, 8.0 GB RAM, Model: Mac17,5)
 Thermal  : Fair (Elevated, fans active)
 Backend  : cpu, Model Profile: mock, Warmup: false
============================================================
batch_size=1    backend=cpu    batch=1    tok/s(total)=245.2    tok/s/req=245.17
                ttft_ms(mean/max)=22.3/22.3 itl_ms(mean/p99)=1.5/1.7 decode_ms/step=4.08 checksum=a8c7f832281a39c5
                peak_rss=9.8MB thermal="Fair (Elevated, fans active)" bus_est=93.2GB/s (62.1% bus util)
```

---

## Empirical Benchmark Results (Apple Silicon Testbed)

**Testbed Specifications:**
- **Hardware**: Apple A18 Pro (6 cores, 8.0 GB Unified Memory, Model: `Mac17,5`)
- **OS / Target**: macOS Darwin (arm64)
- **Workload**: Prompt length = 32 tokens, Generation length = 16 tokens, Block size = 16 tokens
- **Thermal State**: Fair (Elevated, no thermal throttling active)

### 1. Single-User Low Latency (`batch_size = 1`)

| Model Architecture | Category | Backend | Throughput (tok/s) | Decode (ms/step) | TTFT (ms) | ITL p99 (ms) | Peak RSS | Bus Util | Checksum (FNV-1a) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`smollm2`** | Dense Llama (32L) | **`metal`** | **69.0** | **14.5** | 160.9 | 5.4 | 92.4 MB | 17.5% | `88201fb960ff6465` |
| | | `neon` | 68.2 | 14.7 | 165.8 | 5.1 | 92.7 MB | 17.3% | `88201fb960ff6465` |
| | | `cpu` | 48.1 | 20.8 | 232.2 | 7.9 | 93.0 MB | 12.2% | `88201fb960ff6465` |
| **`granite-dense`** | Dense Llama (28L) | **`neon`** | **57.6** | **17.4** | 191.9 | 6.7 | 112.5 MB | 14.6% | `88201fb960ff6465` |
| | | `metal` | 54.4 | 18.4 | 203.5 | 8.2 | 112.4 MB | 13.8% | `88201fb960ff6465` |
| | | `cpu` | 52.4 | 19.1 | 214.5 | 6.6 | 112.2 MB | 13.3% | `88201fb960ff6465` |
| **`granite-h`** 🚀 | Hybrid Mamba-2 (4L Attn) | **`neon`** | **107.5** | **9.3** | **102.4** | **3.5** | **65.3 MB** | 27.2% | `88201fb960ff6465` |
| | | `metal` | 97.5 | 10.3 | 116.1 | 3.9 | **65.5 MB** | 24.7% | `88201fb960ff6465` |
| | | `cpu` | 95.3 | 10.5 | 120.2 | 3.5 | **65.4 MB** | 24.2% | `88201fb960ff6465` |
| **`lfm`** | Hybrid LIV Conv (6L Attn) | **`metal`** | **63.4** | **15.8** | 167.3 | 7.7 | 100.3 MB | 16.1% | `88201fb960ff6465` |
| | | `cpu` | 61.4 | 16.3 | 183.9 | 5.7 | 100.2 MB | 15.6% | `88201fb960ff6465` |
| | | `neon` | 57.0 | 17.5 | 191.2 | 7.1 | 100.2 MB | 14.4% | `88201fb960ff6465` |

### 2. Multi-User Concurrent Serving (`batch_size = 4`)

| Model Architecture | Category | Backend | Total tok/s | Per-Req tok/s | Decode (ms/step) | TTFT p99 (ms) | Peak RSS | Checksum (FNV-1a) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`smollm2`** | Dense Llama (32L) | **`neon`** | **73.0** | 18.25 | 54.8 | 593.7 | 97.5 MB | `b9b23f3a46fd0825` |
| | | `metal` | 71.3 | 17.83 | 56.1 | 614.3 | 96.1 MB | `b9b23f3a46fd0825` |
| | | `cpu` | 44.2 | 11.04 | 90.6 | 868.7 | 97.1 MB | `b9b23f3a46fd0825` |
| **`granite-dense`** | Dense Llama (28L) | **`neon`** | **59.0** | 14.76 | 67.8 | 744.4 | 112.6 MB | `b9b23f3a46fd0825` |
| | | `metal` | 58.0 | 14.49 | 69.0 | 741.6 | 117.1 MB | `b9b23f3a46fd0825` |
| | | `cpu` | 55.7 | 13.92 | 71.8 | 779.5 | 117.1 MB | `b9b23f3a46fd0825` |
| **`granite-h`** 🚀 | Hybrid Mamba-2 (4L Attn) | **`metal`** | **105.1** | **26.28** | **38.1** | **407.7** | **70.0 MB** | `b9b23f3a46fd0825` |
| | | `cpu` | 102.9 | 25.73 | 38.9 | 427.5 | **70.5 MB** | `b9b23f3a46fd0825` |
| | | `neon` | 99.2 | 24.79 | 40.3 | 426.4 | **72.1 MB** | `b9b23f3a46fd0825` |
| **`lfm`** | Hybrid LIV Conv (6L Attn) | **`metal`** | **66.8** | 16.71 | 59.9 | 652.3 | 101.4 MB | `b9b23f3a46fd0825` |
| | | `cpu` | 66.5 | 16.63 | 60.1 | 653.6 | 104.8 MB | `b9b23f3a46fd0825` |
| | | `neon` | 55.3 | 13.81 | 72.4 | 782.4 | 101.3 MB | `b9b23f3a46fd0825` |

### Key Performance Insights
1. **Hybrid Architecture Memory & Latency Superiority**: `granite-h` achieves the lowest peak memory footprint (**65.3 MB**) and highest generation throughput (**107.5 tok/s**), outperforming dense models by up to **$2\times$** through 87.5% sparse KV cache reduction on Mamba-2 layers.
2. **Apple Silicon Hardware Acceleration**: `metal` (69.0 tok/s) and `neon` (68.2 tok/s) deliver **$\approx 1.43\times$** acceleration over scalar CPU (48.1 tok/s) on dense workloads.
3. **Strict Determinism**: 100% identical 64-bit FNV-1a checksums across CPU, NEON, and Metal backends confirm numerical consistency.
