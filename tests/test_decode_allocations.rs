//! Ticket #48 / spec #61: the README claims the inner decode loop is zero-allocation.
//! This is that claim, measured: allocations inside one decode step after warmup.

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
const CTX: usize = 2048;

/// Allocations (count, bytes) charged to the last of `reps` decode steps.
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
fn decode_step_is_allocation_free_after_warmup() {
    let model = Model::create_mock_tiny(HIDDEN, Q_HEADS, KV_HEADS, LAYERS, VOCAB)
        .expect("mock model at SmolLM2-360M dimensions");

    let (allocs, bytes) = step_allocations(&model, CTX, 3);
    println!("#48 decode step at ctx{CTX}: {allocs} allocations, {bytes} bytes");
    assert_eq!(
        allocs, 0,
        "decode step must not allocate after warmup ({bytes} bytes in {allocs} allocations)"
    );
}
