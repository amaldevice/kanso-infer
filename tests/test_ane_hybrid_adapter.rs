use candle_core::Tensor;
use kanso_infer::engine::Engine;
use kanso_infer::ffn_offload::{
    AneLayerAdapter, CoreMlAneAdapter, CpuFallbackAdapter, HybridExecutionPipeline,
    PlannedCoreMlAdapter,
};
use kanso_infer::model::Model;

#[test]
fn test_ane_layer_adapter_interface_and_dispatch() {
    let model = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");
    let layer_idx = 0;
    let layer_weights = &model.weights.layers[layer_idx];
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;

    // 1. CpuFallbackAdapter implements FfnOffloadAdapter
    let mut cpu_adapter = CpuFallbackAdapter::from_layer_weights(
        layer_idx,
        layer_weights,
        hidden_size,
        intermediate_size,
        kanso_infer::Backend::Cpu,
    );
    assert!(cpu_adapter.is_available());
    assert_eq!(cpu_adapter.backend_name(), "cpu");
    assert_eq!(cpu_adapter.layer_index(), 0);

    let dummy_input = vec![0.5f32; hidden_size];
    let cpu_out = cpu_adapter
        .forward_mlp(&dummy_input)
        .expect("CPU adapter forward_mlp failed");
    assert_eq!(cpu_out.len(), hidden_size);

    // Test weight dispatch
    cpu_adapter
        .dispatch_weights(layer_weights)
        .expect("dispatch_weights failed");

    // 2. PlannedCoreMlAdapter implements FfnOffloadAdapter
    let coreml_adapter = PlannedCoreMlAdapter::new(
        layer_idx,
        hidden_size,
        intermediate_size,
        layer_weights,
        kanso_infer::Backend::Cpu,
    );
    assert_eq!(coreml_adapter.layer_index(), 0);

    let coreml_out = coreml_adapter
        .forward_mlp(&dummy_input)
        .expect("CoreML adapter forward_mlp failed");
    assert_eq!(coreml_out.len(), hidden_size);

    // Verify output parity between CPU adapter and CoreML adapter
    for (a, b) in cpu_out.iter().zip(coreml_out.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "Diff between CPU fallback and CoreML output: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_ane_adapter_fallback_mechanism() {
    let model = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");
    let layer_idx = 0;
    let layer_weights = &model.weights.layers[layer_idx];
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;

    let coreml_adapter = CoreMlAneAdapter::new(
        layer_idx,
        hidden_size,
        intermediate_size,
        layer_weights,
        kanso_infer::Backend::Cpu,
    );
    // Honest availability (#62 F1): has_ane_support() is false so adapter defaults unavailable
    assert!(!coreml_adapter.is_available());
    assert_eq!(coreml_adapter.backend_name(), "cpu");
    let dummy_input = vec![0.25f32; hidden_size];
    let fallback_out = coreml_adapter
        .forward_mlp(&dummy_input)
        .expect("fallback forward_mlp should smoothly succeed");
    assert_eq!(fallback_out.len(), hidden_size);

    // Check with AMX fallback if Accelerate is available
    if kanso_infer::accelerate::amx::has_accelerate_support() {
        let amx_fallback_adapter = CoreMlAneAdapter::new(
            layer_idx,
            hidden_size,
            intermediate_size,
            layer_weights,
            kanso_infer::Backend::Amx,
        );
        assert!(!amx_fallback_adapter.is_available());
        assert_eq!(amx_fallback_adapter.backend_name(), "amx_cpu");

        let amx_out = amx_fallback_adapter
            .forward_mlp(&dummy_input)
            .expect("AMX fallback should succeed");
        for (a, b) in fallback_out.iter().zip(amx_out.iter()) {
            let rel_diff = (a - b).abs() / a.abs().max(1.0);
            assert!(
                rel_diff < 1e-4,
                "AMX fallback parity mismatch: {} vs {}, rel_diff={}",
                a,
                b,
                rel_diff
            );
        }
    }
}

#[test]
fn test_hybrid_scheduling_pipeline_attention_amx_metal_mlp_ane() {
    let model = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;

    let attention_backend = if kanso_infer::accelerate::amx::has_accelerate_support() {
        kanso_infer::Backend::Amx
    } else {
        kanso_infer::Backend::Cpu
    };

    let mut adapters: Vec<Box<dyn AneLayerAdapter>> = Vec::new();
    for (idx, layer) in model.weights.layers.iter().enumerate() {
        let adapter = CoreMlAneAdapter::new(
            idx,
            hidden_size,
            intermediate_size,
            layer,
            attention_backend,
        );
        adapters.push(Box::new(adapter));
    }

    let pipeline = HybridExecutionPipeline::new(attention_backend, adapters);
    assert_eq!(pipeline.adapters.len(), 2);
    assert_eq!(pipeline.attention_backend, attention_backend);

    // Execute hybrid layer with attention on Metal/AMX and MLP on adapter
    let dummy_hidden = Tensor::ones((1, hidden_size), candle_core::DType::F32, &model.device)
        .expect("tensor creation failed");
    let out_hidden = pipeline
        .execute_layer(0, &dummy_hidden, &model.weights.layers[0], &model)
        .expect("execute_layer failed");

    assert_eq!(out_hidden.dims(), &[1, hidden_size]);

    let stats = pipeline.stats();
    assert!(
        stats.attention_dispatches > 0,
        "Hybrid pipeline should dispatch attention to Metal/AMX"
    );
    assert!(
        stats.adapter_dispatches > 0,
        "Hybrid pipeline should delegate MLP to ANE adapter"
    );
}

#[test]
fn test_model_forward_parity_hybrid_vs_monolithic() {
    let mut model_monolithic =
        Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");
    let mut model_hybrid =
        Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");

    let attention_backend = if kanso_infer::accelerate::amx::has_accelerate_support() {
        kanso_infer::Backend::Amx
    } else {
        kanso_infer::Backend::Cpu
    };
    model_monolithic.set_backend(attention_backend);
    model_hybrid.set_backend(attention_backend);

    // Enable FFN offload pipeline on model_hybrid
    model_hybrid.enable_ffn_offload();

    let prompt = vec![12u32, 34u32, 56u32];

    let logits_monolithic = model_monolithic
        .forward_single_prompt(&prompt)
        .expect("monolithic prompt forward failed");
    let logits_hybrid = model_hybrid
        .forward_single_prompt(&prompt)
        .expect("hybrid prompt forward failed");

    assert_eq!(logits_monolithic.dims(), logits_hybrid.dims());

    let vals_mono = logits_monolithic
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let vals_hyb = logits_hybrid
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut max_diff = 0.0f32;
    for (a, b) in vals_mono.iter().zip(vals_hyb.iter()) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-4,
            "Activations deviated beyond tolerance: mono={}, hybrid={}, diff={}",
            a,
            b,
            diff
        );
    }
}

#[test]
fn test_engine_hybrid_layer_synchronization_and_decode() {
    let mut model = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("failed to create mock model");
    model.enable_ffn_offload();

    let mut engine = Engine::new(model, 4, 16, 16).expect("engine");
    engine.enable_ffn_offload();

    let request_id = engine.add_request(vec![10, 20, 30], 10);

    let output = engine.step().expect("step failed");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].request_id, request_id);

    // Verify hybrid pipeline stats recorded dispatches during decode step
    if let Some(pipeline) = &engine.model.hybrid_pipeline {
        let stats = pipeline.stats();
        assert!(
            stats.adapter_dispatches > 0,
            "Engine decode step should synchronize through adapter"
        );
    }
}

#[test]
fn test_decode_step_parity_hybrid_vs_monolithic() {
    let model_mono = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("model mono");
    let mut model_hyb = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("model hyb");
    model_hyb.enable_ffn_offload();

    let block_size = 16;
    let mut kv_mono =
        kanso_infer::kv_cache::KvStorageKind::f32_new(2, 4, block_size, 2, 8, &model_mono.device)
            .expect("kv mono");
    let mut kv_hyb =
        kanso_infer::kv_cache::KvStorageKind::f32_new(2, 4, block_size, 2, 8, &model_hyb.device)
            .expect("kv hyb");

    let mut bt_mono = kanso_infer::kv_cache::BlockTable::new();
    bt_mono.push_block(0);
    let mut bt_hyb = kanso_infer::kv_cache::BlockTable::new();
    bt_hyb.push_block(0);

    let token = 42u32;
    let pos = 0;

    let logits_mono = model_mono
        .forward_decode_step(token, pos, &bt_mono, &mut kv_mono)
        .expect("mono decode step");
    let logits_hyb = model_hyb
        .forward_decode_step(token, pos, &bt_hyb, &mut kv_hyb)
        .expect("hyb decode step");

    assert_eq!(logits_mono.dims(), logits_hyb.dims());
    let v_mono = logits_mono.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_hyb = logits_hyb.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for (a, b) in v_mono.iter().zip(v_hyb.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "Decode step mismatch: mono={}, hyb={}",
            a,
            b
        );
    }
}

#[test]
fn test_adapter_input_validation_errors() {
    let model = Model::create_mock_tiny(32, 4, 2, 2, 100).expect("model");
    let adapter = CoreMlAneAdapter::new(
        0,
        32,
        64,
        &model.weights.layers[0],
        kanso_infer::Backend::Cpu,
    );

    // Empty input error
    assert!(adapter.forward_mlp(&[]).is_err());

    // Invalid input length (not multiple of hidden_size 32)
    assert!(adapter.forward_mlp(&[1.0f32; 15]).is_err());
}
