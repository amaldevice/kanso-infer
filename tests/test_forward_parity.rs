use candle_core::{Device, Tensor};

#[test]
fn test_mock_model_forward_single_token() {
    let model = kanso_infer::model::Model::create_mock_tiny(
        /* hidden_size */ 16, /* num_attention_heads */ 4, /* num_kv_heads */ 2,
        /* num_layers */ 2, /* vocab_size */ 100,
    )
    .expect("failed to create mock model");

    let prompt = vec![42u32];
    let logits = model
        .forward_single_prompt(&prompt)
        .expect("forward failed");

    // Logits should be [1, vocab_size]
    assert_eq!(logits.dims(), &[1, 100]);

    // Check no NaNs
    let vals = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(!vals.iter().any(|v| v.is_nan() || v.is_infinite()));
}

#[test]
fn test_greedy_sampling_deterministic() {
    let dev = Device::Cpu;
    // Logits with clear maximum at index 7
    let mut data = vec![0.0f32; 10];
    data[7] = 100.0;
    let logits = Tensor::from_vec(data, (1, 10), &dev).unwrap();

    let sampled = kanso_infer::sampler::sample_token(&logits, 0.0, 1.0, 42).unwrap();
    assert_eq!(sampled, 7);
}
