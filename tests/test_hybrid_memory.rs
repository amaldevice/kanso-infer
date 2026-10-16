#[test]
fn test_sparse_kv_cache_allocates_only_for_attention_layers() {
    let config = kanso_infer::config::ModelConfig::granite_4_0_h_350m();
    let mut pool = kanso_infer::kv_cache::BlockPool::new(100);

    // Granite-4.0-h has 32 layers total, but ONLY 4 attention layers: [7, 15, 23, 31]
    let sparse_table = kanso_infer::hybrid::SparseBlockTable::allocate_initial(
        &config, &mut pool, /* num_tokens */ 16,
    );

    assert_eq!(sparse_table.attention_layers().len(), 4);
    assert_eq!(sparse_table.total_allocated_blocks(), 4);

    // If it were a dense 32-layer model, it would have allocated 32 blocks:
    let dense_equivalent_blocks = config.num_hidden_layers;
    let savings_ratio = (dense_equivalent_blocks - sparse_table.total_allocated_blocks()) as f64
        / (dense_equivalent_blocks as f64);
    assert_eq!(savings_ratio, 0.875); // 87.5% memory block savings!
}

#[test]
fn test_recurrent_state_constant_memory_buffer() {
    // 28 Mamba-2 SSM recurrent layers, each state vector dim = 64
    let mut recurrent = kanso_infer::hybrid::RecurrentState::new(28, 64);
    let initial_bytes = recurrent.state_bytes();
    assert_eq!(initial_bytes, 28 * 64 * std::mem::size_of::<f32>());

    // Simulate 1000 autoregressive decode steps
    let dummy_input = vec![0.5f32; 64];
    for _ in 0..1000 {
        for l in 0..28 {
            recurrent.step_update(l, &dummy_input);
        }
    }

    // Crucial invariant: O(1) memory! Memory size does NOT grow with sequence length
    assert_eq!(recurrent.state_bytes(), initial_bytes);
}
