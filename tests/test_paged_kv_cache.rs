#[test]
fn test_block_allocation_and_ref_counting() {
    let mut pool = kanso_infer::kv_cache::BlockPool::new(3);
    assert_eq!(pool.free_blocks_count(), 3);

    let b0 = pool.allocate().expect("failed to allocate b0");
    let b1 = pool.allocate().expect("failed to allocate b1");
    let b2 = pool.allocate().expect("failed to allocate b2");
    assert_eq!(pool.free_blocks_count(), 0);
    assert_eq!(pool.allocate(), None);

    // Ref counting check
    pool.increment_ref(b0); // ref count = 2
    pool.free(b0); // ref count = 1, still in use
    assert_eq!(pool.free_blocks_count(), 0);

    pool.free(b0); // ref count = 0, returns to pool
    assert_eq!(pool.free_blocks_count(), 1);

    pool.free(b1);
    pool.free(b2);
    assert_eq!(pool.free_blocks_count(), 3);
}

#[test]
fn test_prefix_cache_matching() {
    let mut prefix_cache = kanso_infer::kv_cache::PrefixCache::new(16);

    // Prompt A has 32 tokens (exactly 2 blocks of 16)
    let tokens_a: Vec<u32> = (0..32).collect();
    let blocks_a = vec![101, 102];
    prefix_cache.insert_prefix(&tokens_a, &blocks_a);

    // Prompt B shares first 32 tokens + 10 new tokens
    let mut tokens_b = tokens_a.clone();
    tokens_b.extend(100..110);

    let (matched_tokens, matched_blocks) = prefix_cache.find_longest_prefix(&tokens_b);
    assert_eq!(matched_tokens, 32);
    assert_eq!(matched_blocks, vec![101, 102]);

    // Prompt C diverges after 10 tokens (< 1 block)
    let tokens_c: Vec<u32> = (0..10).chain(99..120).collect();
    let (matched_c, blocks_c) = prefix_cache.find_longest_prefix(&tokens_c);
    assert_eq!(matched_c, 0);
    assert!(blocks_c.is_empty());
}
