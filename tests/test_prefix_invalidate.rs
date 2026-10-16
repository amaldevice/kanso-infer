//! Ticket #44: prefix-cache use-after-free (invalidate-on-reuse) + LRU bound.
//! Seeded random mock weights (lesson #57: ones mock is blind to KV corruption).

use kanso_infer::engine::Engine;
use kanso_infer::kv_cache::PrefixCache;
use kanso_infer::model::Model;

fn mock() -> Model {
    Model::create_mock_tiny(16, 4, 2, 2, 50).unwrap()
}

fn engine() -> Engine {
    Engine::new_with_budget(
        mock(),
        /* max_running */ 1,
        /* total_blocks */ 8,
        /* block_size */ 4,
        usize::MAX,
    )
}

fn drain(engine: &mut Engine) {
    let mut steps = 0;
    while engine.has_unfinished_requests() && steps < 50 {
        engine.step().expect("step failed");
        steps += 1;
    }
    assert!(
        !engine.has_unfinished_requests(),
        "engine did not drain in 50 steps"
    );
}

fn generated(engine: &Engine, id: u64) -> Vec<u32> {
    engine
        .sequences
        .iter()
        .find(|s| s.id == id)
        .expect("request id not found")
        .generated_tokens
        .clone()
}

/// A finishes (prefix cached), B reallocates A's blocks with a different prompt,
/// C arrives with A's prefix: C must NOT hit the stale blocks, so C's output
/// must equal the no-cache reference.
#[test]
fn test_prefix_stale_blocks_not_reused_after_reallocation() {
    let prompt_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let prompt_b: Vec<u32> = vec![11, 12, 13, 14, 15, 16, 17, 18];
    let mut prompt_c = prompt_a.clone();
    prompt_c.extend_from_slice(&[9, 10]);

    let mut eng = engine();
    let _a = eng.add_request(prompt_a, 1);
    drain(&mut eng);
    let _b = eng.add_request(prompt_b, 1);
    drain(&mut eng);

    let c = eng.add_request(prompt_c.clone(), 1);
    let hits_before = eng.prefix_cache_hits;
    drain(&mut eng);
    let got = generated(&eng, c);
    assert_eq!(
        eng.prefix_cache_hits, hits_before,
        "#44: C must miss — A's blocks were reused by B, entries are stale"
    );

    let mut reference = engine();
    let r = reference.add_request(prompt_c, 1);
    drain(&mut reference);
    assert_eq!(
        got,
        generated(&reference, r),
        "#44: C output diverged from no-cache reference (stale KV reuse)"
    );
}

/// Cache size must stay bounded: oldest entries evicted, and a touched (recently
/// used) entry must survive over untouched older ones (LRU, not FIFO).
#[test]
fn test_prefix_cache_evicts_lru_under_bound() {
    let mut cache = PrefixCache::new(4);
    // 60 distinct one-block prefixes, all fit under any reasonable bound.
    for i in 0u32..60 {
        cache.insert_prefix(&[10 + i, 20 + i, 30 + i, 40 + i], &[i as usize]);
    }
    // Touch the oldest so it becomes most-recently-used.
    let (m, _) = cache.find_longest_prefix(&[10, 20, 30, 40]);
    assert_eq!(m, 4, "oldest prefix must hit before eviction pressure");
    // 10 more inserts force eviction of the least-recently-used entries.
    for i in 60u32..70 {
        cache.insert_prefix(&[10 + i, 20 + i, 30 + i, 40 + i], &[i as usize]);
    }
    // Touched oldest survives...
    let (m_old, _) = cache.find_longest_prefix(&[10, 20, 30, 40]);
    assert_eq!(m_old, 4, "#44: LRU must keep the recently-used entry");
    // ...while untouched older entries are evicted (bound enforced).
    let (m_evict, _) = cache.find_longest_prefix(&[11, 21, 31, 41]);
    assert_eq!(
        m_evict, 0,
        "#44: LRU bound not enforced — stale oldest prefix still cached"
    );
    // Newest entries are served.
    let (m_new, _) = cache.find_longest_prefix(&[79, 89, 99, 109]);
    assert_eq!(m_new, 4, "#44: newest prefix must survive LRU eviction");
}
