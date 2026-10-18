#[test]
fn test_load_model_configs_smollm2_and_granite() {
    let smol = kanso_infer::config::ModelConfig::smollm2_360m();
    assert_eq!(smol.hidden_size, 960);
    assert_eq!(smol.num_attention_heads, 15);
    assert_eq!(smol.num_key_value_heads, 5);
    assert_eq!(smol.num_hidden_layers, 32);
    assert_eq!(smol.head_dim(), 64);
    // 2 * 32 layers * 5 kv_heads * 64 head_dim * 4 bytes (f32) = 81920 bytes = 80 KB
    assert_eq!(smol.kv_cache_bytes_per_token_f32(), 81920);

    let granite_dense = kanso_infer::config::ModelConfig::granite_4_0_350m_dense();
    assert_eq!(granite_dense.hidden_size, 1024);
    assert_eq!(granite_dense.num_attention_heads, 16);
    assert_eq!(granite_dense.num_key_value_heads, 4);
    assert_eq!(granite_dense.num_hidden_layers, 28);
    assert_eq!(granite_dense.head_dim(), 64);
    // 2 * 28 layers * 4 kv_heads * 64 head_dim * 4 bytes (f32) = 57344 bytes = 56 KB
    assert_eq!(granite_dense.kv_cache_bytes_per_token_f32(), 57344);

    let granite_hybrid = kanso_infer::config::ModelConfig::granite_4_0_h_350m();
    assert_eq!(granite_hybrid.hidden_size, 768);
    assert_eq!(granite_hybrid.num_attention_layers(), 4);
    assert_eq!(granite_hybrid.num_hidden_layers, 32);
    // Only 4 attention layers: 2 * 4 layers * 4 kv_heads * 64 head_dim * 4 bytes = 8192 bytes = 8 KB
    assert_eq!(granite_hybrid.kv_cache_bytes_per_token_f32(), 8192);

    let lfm = kanso_infer::config::ModelConfig::lfm_2_5_350m();
    assert_eq!(lfm.num_attention_layers(), 6);
    assert_eq!(lfm.num_hidden_layers, 16);
    // Only 6 attention layers: 2 * 6 layers * 4 kv_heads * 64 head_dim * 4 bytes = 12288 bytes = 12 KB
    assert_eq!(lfm.kv_cache_bytes_per_token_f32(), 12288);
}

#[test]
fn test_permute_interleaved_to_rotate_half() {
    use candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    // 1 head with head_dim = 4: interleaved [0, 1, 2, 3]
    // Expected rotate-half: [0, 2, 1, 3] (evens first, then odds)
    let input = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (4, 1), &dev).unwrap();
    let permuted = kanso_infer::weights::permute_interleaved_to_rotate_half(&input, 1, 4).unwrap();
    let vals = permuted.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 2.0, 1.0, 3.0]);
}
