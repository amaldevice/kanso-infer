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

/// Ticket #48: the decode-path pool read returns exactly what `gather_blocks` would,
/// whether the table is one consecutive run (zero-copy view) or fragmented (gather).
#[test]
fn test_paged_kv_tokens_match_gather() {
    use candle_core::{Device, IndexOp, Tensor};
    use kanso_infer::kv_cache::KvStorageKind;

    let dev = Device::Cpu;
    let (layers, blocks, bs, kv_heads, hd) = (2usize, 8usize, 4usize, 2usize, 8usize);
    let mut kv = KvStorageKind::f32_new(layers, blocks, bs, kv_heads, hd, &dev).unwrap();

    // Fill every slot with a value that encodes (block, offset).
    for layer in 0..layers {
        for b in 0..blocks {
            for o in 0..bs {
                let fill = (b * 100 + o) as f32;
                let k = Tensor::full(fill, (kv_heads, hd), &dev).unwrap();
                let v = Tensor::full(-fill, (kv_heads, hd), &dev).unwrap();
                kv.write_kv_token(layer, b, o, &k, &v).unwrap();
            }
        }
    }
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    for layer in 0..layers {
        // Consecutive table: blocks 2,3,4 with 10 tokens (fast path).
        let table = vec![2usize, 3, 4];
        let (pk, pv) = kv.paged_kv_tokens(layer, &table, 10).unwrap();
        let (gk, gv) = kv.gather_blocks(layer, &table, 10).unwrap();
        assert_eq!(pk.dims(), gk.dims());
        assert_eq!(max_diff(&flat(&pk), &flat(&gk)), 0.0);
        assert_eq!(max_diff(&flat(&pv), &flat(&gv)), 0.0);

        // Fragmented table: blocks 5,1,6 with 10 tokens (gather fallback).
        let table = vec![5usize, 1, 6];
        let (pk, pv) = kv.paged_kv_tokens(layer, &table, 10).unwrap();
        let (gk, gv) = kv.gather_blocks(layer, &table, 10).unwrap();
        assert_eq!(max_diff(&flat(&pk), &flat(&gk)), 0.0);
        assert_eq!(max_diff(&flat(&pv), &flat(&gv)), 0.0);
        // Spot-check the token order follows the table, not the pool.
        let k0 = flat(&pk.i(0).unwrap());
        assert!(k0.iter().all(|&x| x == 500.0));
        let k4 = flat(&pk.i(4).unwrap());
        assert!(k4.iter().all(|&x| x == 100.0));
    }
}
