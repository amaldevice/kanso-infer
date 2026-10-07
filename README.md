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
- **Zero-Allocation Static Workspace**: Eliminates heap allocation churn (`malloc`/`free`) during inner decode steps by reusing pre-allocated scratchpad workspaces.
- **Sarathi-Serve Token-Budget Chunked Prefill**: Enforces a configurable token budget per iteration ($T_{\text{budget}}$), partitioning large prompts into chunks and co-scheduling them with active decode requests to eradicate Head-of-Line blocking and stabilize $p99$ Inter-Token Latency (ITL).
- **Apple Silicon Metal Zero-Copy UMA Runtime**: Allocates GPU buffers in shared storage mode (`MTLStorageModeShared`), allowing CPU slices and Metal shaders to share physical Unified Memory with zero PCIe copy overhead.
- **Startup Metal Pipeline Caching**: Pre-compiles all Metal Shading Language (`.metal`) compute kernels into `MTLComputePipelineState` handles during initialization, preventing first-request compilation stutter.
- **Fused PagedAttention v2 with Online Softmax**: Computes attention scores and context reduction directly within GPU threadgroup shared memory in tiles, completely eliminating the materialization of intermediate score matrices ($[B, N_h, Q, S]$) in global VRAM.
- **ARM NEON SIMD Quantization Kernels**: Fast vector kernels using AArch64 intrinsics (`vdotq_s32`, `vfmaq_f32`) for on-the-fly Q8_0 and Q4_K matrix-vector multiplication in CPU registers.
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
