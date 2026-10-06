use candle_core::{DType, Device, Tensor};
use kanso_infer::engine::ScratchpadWorkspace;
use kanso_infer::kv_cache::{BlockTable, KvStorage};
use kanso_infer::model::Model;

#[test]
fn test_incremental_decode_matches_monolithic_forward() {
    let model = Model::create_mock_tiny(
        /* hidden_size */ 16,
        /* num_attention_heads */ 4,
        /* num_kv_heads */ 2,
        /* num_layers */ 2,
        /* vocab_size */ 50,
    )
    .expect("failed to create mock model");

    let prompt = vec![5u32];
    let next_token = 12u32;
    let full_sequence = vec![5u32, 12u32];

    // Reference monolithic forward pass over full sequence [5, 12]
    let monolithic_logits = model
        .forward_single_prompt(&full_sequence)
        .expect("monolithic forward failed");

    // Incremental decoding using KvStorage and BlockTable
    let mut kv_storage = KvStorage::new(
        model.config.num_hidden_layers,
        10,
        model.config.block_size,
        model.config.num_key_value_heads,
        model.config.head_dim(),
        &model.device,
    )
    .expect("failed to create kv storage");

    let mut block_table = BlockTable::new();
    block_table.push_block(0);
    block_table.set_num_tokens(1);

    // Step 0: prompt token 5 at pos 0
    let _ = model
        .forward_decode_step(prompt[0], 0, &block_table, &mut kv_storage)
        .expect("prompt step failed");
    block_table.set_num_tokens(2);

    // Step 1: decode token 12 at pos 1
    let incremental_logits = model
        .forward_decode_step(next_token, 1, &block_table, &mut kv_storage)
        .expect("incremental decode step failed");

    assert_eq!(incremental_logits.dims(), monolithic_logits.dims());

    let mono_vec = monolithic_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let incr_vec = incremental_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(mono_vec.len(), incr_vec.len());
    for (i, (m, inc)) in mono_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (m - inc).abs() < 1e-4,
            "Logits mismatch at index {}: monolithic={}, incremental={}",
            i,
            m,
            inc
        );
    }
}
#[test]
fn test_physical_kv_cache_storage_write_and_gather() {
    let device = Device::Cpu;
    let num_layers = 2;
    let num_blocks = 4;
    let block_size = 2;
    let n_kv_heads = 2;
    let head_dim = 4;

    let mut kv = KvStorage::new(
        num_layers,
        num_blocks,
        block_size,
        n_kv_heads,
        head_dim,
        &device,
    )
    .expect("failed to create kv storage");

    // Write token 0 at block 0, offset 0
    let k0 = Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap();
    let v0 = (Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap() * 2.0).unwrap();
    kv.write_kv_token(0, 0, 0, &k0, &v0).unwrap();

    // Write token 1 at block 0, offset 1
    let k1 = (Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap() * 3.0).unwrap();
    let v1 = (Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap() * 4.0).unwrap();
    kv.write_kv_token(0, 0, 1, &k1, &v1).unwrap();

    // Write token 2 at non-contiguous block 2, offset 0
    let k2 = (Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap() * 5.0).unwrap();
    let v2 = (Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap() * 6.0).unwrap();
    kv.write_kv_token(0, 2, 0, &k2, &v2).unwrap();

    // Read back through block table mapping sequence across block 0 and block 2
    let mut table = BlockTable::new();
    table.push_block(0);
    table.push_block(2);
    table.set_num_tokens(3);

    let (k_ctx, v_ctx) = kv.gather_context(0, &table).expect("gather context failed");
    assert_eq!(k_ctx.dims(), &[3, n_kv_heads, head_dim]);
    assert_eq!(v_ctx.dims(), &[3, n_kv_heads, head_dim]);

    let k_vals = k_ctx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_vals = v_ctx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let head_elements = n_kv_heads * head_dim; // 8

    assert_eq!(&k_vals[0..head_elements], &vec![1.0f32; head_elements][..]);
    assert_eq!(&v_vals[0..head_elements], &vec![2.0f32; head_elements][..]);
    assert_eq!(
        &k_vals[head_elements..2 * head_elements],
        &vec![3.0f32; head_elements][..]
    );
    assert_eq!(
        &v_vals[head_elements..2 * head_elements],
        &vec![4.0f32; head_elements][..]
    );
    assert_eq!(
        &k_vals[2 * head_elements..3 * head_elements],
        &vec![5.0f32; head_elements][..]
    );
    assert_eq!(
        &v_vals[2 * head_elements..3 * head_elements],
        &vec![6.0f32; head_elements][..]
    );
}

#[test]
fn test_scratchpad_workspace_memory_reuse() {
    let max_batch = 4;
    let hidden_dim = 16;
    let intermediate_dim = 32;
    let vocab_size = 50;

    let mut workspace =
        ScratchpadWorkspace::new(max_batch, hidden_dim, intermediate_dim, vocab_size);

    assert_eq!(workspace.hidden_states.len(), max_batch * hidden_dim);
    assert_eq!(workspace.logits.len(), max_batch * vocab_size);

    let ptr_hidden = workspace.hidden_states.as_ptr();
    let ptr_logits = workspace.logits.as_ptr();

    for step in 0..10 {
        workspace.hidden_states[0] = step as f32;
        workspace.logits[0] = (step * 2) as f32;
        workspace.clear();

        // Memory pointer must remain identical, proving no reallocation on heap
        assert_eq!(workspace.hidden_states.as_ptr(), ptr_hidden);
        assert_eq!(workspace.logits.as_ptr(), ptr_logits);
    }
}

#[test]
fn test_multiturn_generation_parity_with_reference() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("failed to create mock model");

    let prompt = vec![3u32, 7u32];
    let num_tokens_to_gen = 4;

    // 1. Reference monolithic generation: at each step, re-forward all tokens seen so far
    let mut reference_tokens = prompt.clone();
    for _ in 0..num_tokens_to_gen {
        let logits = model
            .forward_single_prompt(&reference_tokens)
            .expect("monolithic forward failed");
        let next_tok = kanso_infer::sampler::sample_token(&logits, 0.0, 1.0, 42)
            .expect("sampling failed");
        reference_tokens.push(next_tok);
    }

    // 2. Incremental engine generation
    let mut engine = kanso_infer::engine::Engine::new(
        model,
        /* max_running */ 1,
        /* total_blocks */ 10,
        /* block_size */ 2,
    );
    let req_id = engine.add_request(prompt.clone(), num_tokens_to_gen);

    let mut engine_generated = Vec::new();
    while engine.has_unfinished_requests() {
        let step_outs = engine.step().expect("engine step failed");
        for out in step_outs {
            if out.request_id == req_id {
                engine_generated.push(out.token_id);
            }
        }
    }

    assert_eq!(engine_generated.len(), num_tokens_to_gen);
    let reference_generated = &reference_tokens[prompt.len()..];
    assert_eq!(
        engine_generated,
        reference_generated,
        "Incremental decode generated tokens {:?} do not match reference {:?}",
        engine_generated,
        reference_generated
    );
}
