//! The probes behind the F12/F15 budget rows that no workload figure answers
//! (#100).
//!
//! Three of them are real measurements taken on the host:
//!
//! - [`probe_gemv_peak_pct`] times the NEON Q8_0 GEMV kernel and divides its
//!   throughput by the host's measured peak bandwidth.
//! - [`probe_paged_attn_over_theoretical`] times the CPU reference paged-attention
//!   kernel at ctx 4096 / batch 1 and divides by its arithmetic floor.
//! - [`probe_sse_stream_consumed`] consumes a real `chat.completion.chunk`
//!   stream off a live `kanso serve` over plain TCP.
//!
//! The fourth, `idle_gap_pct`, is **not** guessed here: a CPU-timed proxy would
//! report the host's scheduling jitter, not the GPU's idle window. Only a Metal
//! command-buffer timestamp (`MTLCommandBuffer.GPUStartTime`/`GPUEndTime`, or a
//! Metal System Trace) can answer it, and this crate's `metal` binding exposes
//! neither. `gates/fase1.json` therefore keeps that row instrumented-but-unavailable
//! with the reason spelled out, and the report carries no number for it.

use crate::bench::KernelProbes;
use crate::metal::paged_attention::{PagedAttentionParams, cpu_paged_attention};
use crate::model::Model;
use crate::quant::neon::neon_gemv_q8_0;
use crate::quant::q8_0::{BlockQ8_0, QK8_0};
use std::time::Instant;

/// Attention context the `paged_attn_ctx4096_b1_over_theoretical` row names.
pub const PROBE_ATTN_CTX: usize = 4096;

/// GEMV shape of the probe: 1024 rows over 1024 columns of Q8_0 weights, i.e.
/// 1024 * 1024 * 34 B ≈ 35 MB read per pass — large enough for the timing to
/// swamp the loop overhead, small enough not to distort the run's own RSS.
const PROBE_GEMV_ROWS: usize = 1024;
const PROBE_GEMV_COLS: usize = 1024;

/// Bytes one Q8_0 block occupies in the packed layout (f32 scale + 32 i8).
const Q8_0_BLOCK_BYTES: f64 = (std::mem::size_of::<f32>() + QK8_0) as f64;

/// The kernel [`probe_gemv_peak_pct`] actually times, so a reader can tell which
/// number the `gemv_peak_pct` row judged (#100).
///
/// On aarch64 this is the hand-written NEON kernel. Elsewhere it is the scalar
/// reference, which cannot approach a memory peak — so the probe declines to run
/// there rather than report a figure the row's `>= 60%` budget would misread.
pub const GEMV_PROBE_KERNEL: &str = if cfg!(target_arch = "aarch64") {
    "neon_gemv_q8_0"
} else {
    "scalar_gemv_q8_0"
};

/// Runs the NEON Q8_0 GEMV kernel over a 1024x1024 matrix and reports its
/// throughput as a percentage of the host's measured peak bandwidth.
///
/// The kernel reads every weight byte exactly once, so
/// `weights_bytes / elapsed / peak` is the share of peak the kernel reached.
///
/// `None` off aarch64: the NEON kernel does not exist there, and timing the
/// scalar fallback would answer a question nobody asked under this name.
pub fn probe_gemv_peak_pct(peak_bandwidth_gb_s: f64) -> Option<f64> {
    if peak_bandwidth_gb_s <= 0.0 || !cfg!(target_arch = "aarch64") {
        return None;
    }
    let blocks_per_row = PROBE_GEMV_COLS / QK8_0;
    let weights: Vec<BlockQ8_0> = vec![
        BlockQ8_0 {
            d: 0.01,
            qs: [1i8; QK8_0],
        };
        PROBE_GEMV_ROWS * blocks_per_row
    ];
    let x = vec![1.0f32; PROBE_GEMV_COLS];
    let mut out = vec![0.0f32; PROBE_GEMV_ROWS];

    // One warm-up pass so the first-touch page faults are not billed as time.
    neon_gemv_q8_0(&weights, &x, &mut out, PROBE_GEMV_ROWS, PROBE_GEMV_COLS);

    let started = Instant::now();
    neon_gemv_q8_0(&weights, &x, &mut out, PROBE_GEMV_ROWS, PROBE_GEMV_COLS);
    let elapsed_s = started.elapsed().as_secs_f64();
    if elapsed_s <= 0.0 {
        return None;
    }

    let weight_bytes = (PROBE_GEMV_ROWS * blocks_per_row) as f64 * Q8_0_BLOCK_BYTES;
    let achieved_gb_s = weight_bytes / elapsed_s / 1e9;
    Some(achieved_gb_s / peak_bandwidth_gb_s * 100.0)
}

/// Bytes the theoretical paged-attention floor assumes: one read of K and one of
/// V per context token, per KV head.
fn attention_floor_bytes(ctx: usize, num_kv_heads: usize, head_dim: usize) -> f64 {
    2.0 * ctx as f64 * num_kv_heads as f64 * head_dim as f64 * std::mem::size_of::<f32>() as f64
}

/// Times the CPU paged-attention kernel at ctx 4096 / batch 1 and reports the
/// elapsed time over the arithmetic floor
/// `2 * ctx * kv_heads * head_dim * 4 B / peak bandwidth`.
///
/// A CPU kernel cannot reach a memory floor, so the figure is expected to be well
/// above 1.0 — that is what the row's `max 2.0` budget is there to catch. The
/// number is real either way: it is this kernel's own measurement.
pub fn probe_paged_attn_over_theoretical(
    peak_bandwidth_gb_s: f64,
    num_kv_heads: usize,
    head_dim: usize,
) -> Option<f64> {
    if peak_bandwidth_gb_s <= 0.0 || num_kv_heads == 0 || head_dim == 0 {
        return None;
    }
    let ctx = PROBE_ATTN_CTX;
    let block_size = 16usize;
    let blocks = ctx.div_ceil(block_size);
    let params = PagedAttentionParams {
        num_seqs: 1,
        num_heads: num_kv_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        block_size: block_size as u32,
        max_num_blocks_per_seq: blocks as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };

    let q = vec![0.01f32; num_kv_heads * head_dim];
    let kv_elems = blocks * block_size * num_kv_heads * head_dim;
    let k_cache = vec![0.01f32; kv_elems];
    let v_cache = vec![0.01f32; kv_elems];
    let block_tables: Vec<u32> = (0..blocks as u32).collect();
    let context_lens = vec![ctx as u32];

    // Warm-up: the first pass pays for the allocation's page faults.
    cpu_paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        &params,
    );

    let started = Instant::now();
    let _ = cpu_paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        &params,
    );
    let elapsed_s = started.elapsed().as_secs_f64();
    if elapsed_s <= 0.0 {
        return None;
    }

    let floor_s = attention_floor_bytes(ctx, num_kv_heads, head_dim) / (peak_bandwidth_gb_s * 1e9);
    (floor_s > 0.0).then(|| elapsed_s / floor_s)
}

/// Consumes one `stream: true` chat completion from a **live** `kanso serve`
/// listening on `host:port`, reading the SSE frames off a plain TCP socket until
/// the `[DONE]` sentinel arrives.
///
/// This is the functional probe behind `sse_sdk_stream_consumed`: it answers
/// "did a client see a whole OpenAI-shaped stream from the real server", not
/// "how fast was it". It talks HTTP/1.1 by hand so the library needs no HTTP
/// client dependency for it.
///
/// `None` when no server is listening — a probe that could not run is not a 0,
/// and the gate skips the row instead of failing a stream nobody opened.
pub fn probe_sse_stream_consumed(host: &str, port: u16) -> Option<f64> {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let body = serde_json::json!({
        "model": "kanso",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "stream": true,
    })
    .to_string();

    let stream = TcpStream::connect((host, port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok()?;
    let mut writer = stream.try_clone().ok()?;
    write!(
        writer,
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
    .ok()?;
    writer.flush().ok()?;

    let mut saw_chunk = false;
    let mut saw_done = false;
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { break };
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            saw_done = true;
            break;
        }
        if payload.contains("\"chat.completion.chunk\"") {
            saw_chunk = true;
        }
    }

    Some(if saw_chunk && saw_done { 1.0 } else { 0.0 })
}

/// Every probe this host can run, gathered into one block for the report.
///
/// `idle_gap_pct` is deliberately left `None`: see the module docs. The gate
/// skips that row rather than judging a number this crate cannot measure.
pub fn run_probes(
    peak_bandwidth_gb_s: f64,
    num_kv_heads: usize,
    head_dim: usize,
    sse_endpoint: Option<(&str, u16)>,
) -> KernelProbes {
    KernelProbes {
        gemv_peak_pct: probe_gemv_peak_pct(peak_bandwidth_gb_s),
        paged_attn_ctx4096_b1_over_theoretical: probe_paged_attn_over_theoretical(
            peak_bandwidth_gb_s,
            num_kv_heads,
            head_dim,
        ),
        idle_gap_pct: None,
        sse_sdk_stream_consumed: sse_endpoint
            .and_then(|(host, port)| probe_sse_stream_consumed(host, port)),
    }
}

/// Model geometry the paged-attention probe needs, read off the loaded model.
pub fn attention_geometry(model: &Model) -> (usize, usize) {
    (model.config.num_key_value_heads, model.config.head_dim())
}
