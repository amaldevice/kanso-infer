//! Ticket #48 / spec #61: the README's "zero-allocation decode step" claim was removed
//! because it does not hold — candle allocates an output for every op, so a decode step
//! always allocates. What this guards instead: a decode step must not allocate the KV
//! history again (the #48 gather-copy regression), so the bytes allocated past a short
//! context stay far below one full history.
//!
//! The counting global allocator lives in this file's own test binary so its counters
//! are not polluted by other test modules running in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

struct CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const HIDDEN: usize = 960;
const Q_HEADS: usize = 15;
const KV_HEADS: usize = 5;
const LAYERS: usize = 32;
const VOCAB: usize = 49_152;
const SHORT_CTX: usize = 128;
const LONG_CTX: usize = 2048;

/// Allocations (count, bytes) charged to the last of `reps` decode steps at `ctx`.
fn step_allocations(model: &Model, ctx: usize, reps: usize) -> (usize, usize) {
    let block_size = model.config.block_size;
    let blocks = ctx.div_ceil(block_size);
    let mut kv_storage = KvStorage::new(
        model.config.num_hidden_layers,
        blocks,
        block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("kv storage");
    let mut block_table = BlockTable::new();
    for b in 0..blocks {
        block_table.push_block(b);
    }
    block_table.set_num_tokens(ctx);
    let pos = ctx - 1;

    for _ in 0..2 {
        let _ = model
            .forward_decode_step(1, pos, &block_table, &mut kv_storage)
            .expect("warmup decode step");
    }

    let mut result = (0, 0);
    for r in 0..reps {
        let allocs = ALLOCS.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);
        let _ = model
            .forward_decode_step(7, pos, &block_table, &mut kv_storage)
            .expect("decode step");
        if r + 1 == reps {
            result = (
                ALLOCS.load(Ordering::Relaxed) - allocs,
                BYTES.load(Ordering::Relaxed) - bytes,
            );
        }
    }
    result
}

#[test]
fn decode_step_does_not_allocate_the_kv_history() {
    let mut model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
        .expect("mock model at SmolLM2-360M dimensions");

    // Bytes one full KV history occupies: K+V per token per layer.
    let history_bytes = 2
        * LONG_CTX
        * model.config.num_key_value_heads
        * model.config.head_dim()
        * 4
        * model.config.num_hidden_layers;

    let mut backends = vec![kanso_infer::Backend::Cpu];
    if !matches!(kanso_infer::Backend::default(), kanso_infer::Backend::Cpu) {
        backends.push(kanso_infer::Backend::default());
    }
    if kanso_infer::accelerate::amx::has_accelerate_support() {
        backends.push(kanso_infer::Backend::Amx);
    }

    for backend in backends {
        model.set_backend(backend);
        let (allocs_short, bytes_short) = step_allocations(&model, SHORT_CTX, 3);
        let (allocs_long, bytes_long) = step_allocations(&model, LONG_CTX, 3);
        let extra = bytes_long.saturating_sub(bytes_short);
        println!(
            "#48 [{backend}] decode allocs: ctx{SHORT_CTX} {allocs_short} allocs / {bytes_short} bytes, \
             ctx{LONG_CTX} {allocs_long} allocs / {bytes_long} bytes, extra {extra} bytes vs history {history_bytes} (limit {})",
            history_bytes / 4,
        );
        // The old path re-gathered the whole history per layer per token: the extra
        // bytes at long context were ~3x one full history. The paged read keeps them
        // to the attention intermediates (scores/probs), far below one history.
        assert!(
            extra < history_bytes / 4,
            "[{backend}] decode at ctx{LONG_CTX} allocates {extra} extra bytes vs {bytes_short} at ctx{SHORT_CTX}: \
             the KV history ({history_bytes} bytes) must not be re-materialized per step"
        );
        // The op count of a step does not depend on the context length.
        assert!(
            allocs_long <= allocs_short + 64,
            "[{backend}] allocation count grows with context: {allocs_long} at ctx{LONG_CTX} vs {allocs_short} at ctx{SHORT_CTX}"
        );
    }
}
