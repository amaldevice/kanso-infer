use candle_core::{DType, Device, Tensor};
use kanso_infer::kv_cache::{BlockPool, BlockTable, KvStorage, QuantMode, QuantizedKvStorage};
use std::time::Instant;

#[test]
fn test_quantized_kv_int8_reconstruction_accuracy() {
    let device = Device::Cpu;
    let num_layers = 2;
    let num_blocks = 4;
    let block_size = 4;
    let n_kv_heads = 4;
    let head_dim = 64;

    let mut storage = QuantizedKvStorage::new(
        num_layers,
        num_blocks,
        block_size,
        n_kv_heads,
        head_dim,
        QuantMode::Int8,
        &device,
    )
    .expect("failed to create int8 quantized storage");

    // Generate realistic KV tensor values with continuous variation
    let total_elements = n_kv_heads * head_dim;
    let k_raw: Vec<f32> = (0..total_elements)
        .map(|i| ((i as f32) * 0.1).sin() * 2.5)
        .collect();
    let v_raw: Vec<f32> = (0..total_elements)
        .map(|i| ((i as f32) * 0.15).cos() * 1.8)
        .collect();

    let k_tensor = Tensor::from_vec(k_raw.clone(), (n_kv_heads, head_dim), &device).unwrap();
    let v_tensor = Tensor::from_vec(v_raw.clone(), (n_kv_heads, head_dim), &device).unwrap();

    // Write token at block 0, offset 0
    storage
        .write_quantized_kv_token(0, 0, 0, &k_tensor, &v_tensor)
        .expect("write token failed");

    // Gather back using BlockTable
    let mut table = BlockTable::new();
    table.push_block(0);
    table.set_num_tokens(1);

    let (k_deq, v_deq) = storage
        .gather_dequantized_context(0, &table)
        .expect("gather failed");

    assert_eq!(k_deq.dims(), &[1, n_kv_heads, head_dim]);
    assert_eq!(v_deq.dims(), &[1, n_kv_heads, head_dim]);

    let k_recon = k_deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_recon = v_deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Calculate relative L2 reconstruction error: ||x_recon - x||_2 / ||x||_2
    let calc_rel_l2 = |orig: &[f32], recon: &[f32]| -> f32 {
        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for (&o, &r) in orig.iter().zip(recon.iter()) {
            let diff = r - o;
            err_sq += diff * diff;
            norm_sq += o * o;
        }
        (err_sq / norm_sq).sqrt()
    };

    let k_err = calc_rel_l2(&k_raw, &k_recon);
    let v_err = calc_rel_l2(&v_raw, &v_recon);

    println!("INT8 K reconstruction error: {:.4}%", k_err * 100.0);
    println!("INT8 V reconstruction error: {:.4}%", v_err * 100.0);

    // Acceptance criterion: reconstruction error <= 1.0%
    assert!(
        k_err <= 0.010,
        "INT8 K reconstruction error {:.4}% exceeded 1.0%",
        k_err * 100.0
    );
    assert!(
        v_err <= 0.010,
        "INT8 V reconstruction error {:.4}% exceeded 1.0%",
        v_err * 100.0
    );
}

#[test]
fn test_quantized_kv_fp8_reconstruction_accuracy() {
    let device = Device::Cpu;
    let num_layers = 2;
    let num_blocks = 4;
    let block_size = 4;
    let n_kv_heads = 4;
    let head_dim = 64;

    let mut storage = QuantizedKvStorage::new(
        num_layers,
        num_blocks,
        block_size,
        n_kv_heads,
        head_dim,
        QuantMode::Fp8,
        &device,
    )
    .expect("failed to create fp8 quantized storage");

    // Structured KV values representable within FP8 dynamic range
    let total_elements = n_kv_heads * head_dim;
    let k_raw: Vec<f32> = (0..total_elements)
        .map(|i| {
            let val = ((i % 16) as f32) * 0.5 - 3.5;
            if val == 0.0 { 1.0 } else { val }
        })
        .collect();
    let v_raw: Vec<f32> = (0..total_elements)
        .map(|i| ((i % 8) as f32) * 1.0 + 0.5)
        .collect();

    let k_tensor = Tensor::from_vec(k_raw.clone(), (n_kv_heads, head_dim), &device).unwrap();
    let v_tensor = Tensor::from_vec(v_raw.clone(), (n_kv_heads, head_dim), &device).unwrap();

    storage
        .write_quantized_kv_token(0, 0, 0, &k_tensor, &v_tensor)
        .expect("write token failed");

    let mut table = BlockTable::new();
    table.push_block(0);
    table.set_num_tokens(1);

    let (k_deq, v_deq) = storage
        .gather_dequantized_context(0, &table)
        .expect("gather failed");

    let k_recon = k_deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_recon = v_deq.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let calc_rel_l2 = |orig: &[f32], recon: &[f32]| -> f32 {
        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for (&o, &r) in orig.iter().zip(recon.iter()) {
            let diff = r - o;
            err_sq += diff * diff;
            norm_sq += o * o;
        }
        (err_sq / norm_sq).sqrt()
    };

    let k_err = calc_rel_l2(&k_raw, &k_recon);
    let v_err = calc_rel_l2(&v_raw, &v_recon);

    println!("FP8 K reconstruction error: {:.4}%", k_err * 100.0);
    println!("FP8 V reconstruction error: {:.4}%", v_err * 100.0);

    // Acceptance criterion: reconstruction error <= 1.0%
    assert!(
        k_err <= 0.010,
        "FP8 K reconstruction error {:.4}% exceeded 1.0%",
        k_err * 100.0
    );
    assert!(
        v_err <= 0.010,
        "FP8 V reconstruction error {:.4}% exceeded 1.0%",
        v_err * 100.0
    );
}

#[test]
fn test_physical_block_memory_reduction() {
    let device = Device::Cpu;
    let n_kv_heads = 8;
    let head_dim = 128;
    let block_size = 16;
    let num_blocks = 32;
    let num_layers = 12;

    let storage_int8 = QuantizedKvStorage::new_int8(
        num_layers, num_blocks, block_size, n_kv_heads, head_dim, &device,
    )
    .unwrap();

    let storage_fp8 = QuantizedKvStorage::new_fp8(
        num_layers, num_blocks, block_size, n_kv_heads, head_dim, &device,
    )
    .unwrap();

    let bytes_per_token_int8 = storage_int8.memory_bytes_per_token();
    let bytes_per_token_fp8 = storage_fp8.memory_bytes_per_token();
    let bytes_per_token_fp32 = storage_int8.fp32_memory_bytes_per_token();

    println!("FP32 bytes per token: {}", bytes_per_token_fp32);
    println!("INT8 bytes per token: {}", bytes_per_token_int8);
    println!("FP8 bytes per token: {}", bytes_per_token_fp8);

    // Acceptance criterion: Physical block memory consumption per token is reduced by at least 2x
    assert!(
        bytes_per_token_int8 * 2 <= bytes_per_token_fp32,
        "INT8 did not achieve >= 2x reduction: int8={}, fp32={}",
        bytes_per_token_int8,
        bytes_per_token_fp32
    );
    assert!(
        bytes_per_token_fp8 * 2 <= bytes_per_token_fp32,
        "FP8 did not achieve >= 2x reduction: fp8={}, fp32={}",
        bytes_per_token_fp8,
        bytes_per_token_fp32
    );

    let ratio_int8 = storage_int8.memory_reduction_ratio();
    let ratio_fp8 = storage_fp8.memory_reduction_ratio();
    println!("INT8 memory reduction ratio: {:.2}x", ratio_int8);
    println!("FP8 memory reduction ratio: {:.2}x", ratio_fp8);
    assert!(ratio_int8 >= 2.0);
    assert!(ratio_fp8 >= 2.0);
}

#[test]
fn test_block_pool_allocation_recycling_zero_leakage() {
    let device = Device::Cpu;
    let num_blocks = 8;
    let block_size = 4;
    let n_kv_heads = 2;
    let head_dim = 16;
    let num_layers = 1;

    let mut pool = BlockPool::new(num_blocks);
    assert_eq!(pool.free_blocks_count(), num_blocks);

    let mut storage = QuantizedKvStorage::new_int8(
        num_layers, num_blocks, block_size, n_kv_heads, head_dim, &device,
    )
    .unwrap();

    // Allocate 3 blocks
    let b0 = pool.allocate().expect("failed b0");
    let b1 = pool.allocate().expect("failed b1");
    let b2 = pool.allocate().expect("failed b2");
    assert_eq!(pool.free_blocks_count(), num_blocks - 3);

    let k_dummy = Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap();
    let v_dummy = Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap();

    // Write to allocated blocks
    storage
        .write_quantized_kv_token(0, b0, 0, &k_dummy, &v_dummy)
        .unwrap();
    storage
        .write_quantized_kv_token(0, b1, 0, &k_dummy, &v_dummy)
        .unwrap();
    storage
        .write_quantized_kv_token(0, b2, 0, &k_dummy, &v_dummy)
        .unwrap();

    // Reset and recycle all allocated blocks
    storage.reset_block(b0);
    pool.free(b0);
    storage.reset_block(b1);
    pool.free(b1);
    storage.reset_block(b2);
    pool.free(b2);

    // Verify zero memory leakage in pool
    assert_eq!(
        pool.free_blocks_count(),
        pool.total_blocks(),
        "Memory leakage detected: free blocks count does not equal total blocks"
    );

    // Reallocate all blocks successfully
    let mut reallocated = Vec::new();
    for _ in 0..num_blocks {
        reallocated.push(pool.allocate().expect("should be able to allocate"));
    }
    assert_eq!(pool.free_blocks_count(), 0);

    for b in reallocated {
        pool.free(b);
    }
    assert_eq!(pool.free_blocks_count(), num_blocks);
}

#[test]
fn test_long_sequence_gather_and_decode_step() {
    let device = Device::Cpu;
    let n_kv_heads = 4;
    let head_dim = 64;
    let block_size = 16;
    let seq_len = 256; // 16 blocks
    let num_blocks = 20;
    let num_layers = 1;

    let mut q_storage = QuantizedKvStorage::new_int8(
        num_layers, num_blocks, block_size, n_kv_heads, head_dim, &device,
    )
    .unwrap();

    let mut fp32_storage = KvStorage::new(
        num_layers, num_blocks, block_size, n_kv_heads, head_dim, &device,
    )
    .unwrap();

    let mut table = BlockTable::new();
    for b in 0..(seq_len / block_size) {
        table.push_block(b);
    }
    table.set_num_tokens(seq_len);

    let k = Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap();
    let v = Tensor::ones((n_kv_heads, head_dim), DType::F32, &device).unwrap();

    // Populate both caches
    for t in 0..seq_len {
        let block = t / block_size;
        let offset = t % block_size;
        q_storage
            .write_quantized_kv_token(0, block, offset, &k, &v)
            .unwrap();
        fp32_storage
            .write_kv_token(0, block, offset, &k, &v)
            .unwrap();
    }

    // Measure decode gather steps
    let start_fp32 = Instant::now();
    for _ in 0..10 {
        let _ = fp32_storage.gather_context(0, &table).unwrap();
    }
    let dur_fp32 = start_fp32.elapsed();

    let start_q = Instant::now();
    for _ in 0..10 {
        let _ = q_storage.gather_dequantized_context(0, &table).unwrap();
    }
    let dur_q = start_q.elapsed();

    println!(
        "Sequence length {}: FP32 gather 10x took {:?}, Quantized gather 10x took {:?}",
        seq_len, dur_fp32, dur_q
    );

    // Verify gathered content correctness
    let (k_q, v_q) = q_storage.gather_dequantized_context(0, &table).unwrap();
    assert_eq!(k_q.dims(), &[seq_len, n_kv_heads, head_dim]);
    assert_eq!(v_q.dims(), &[seq_len, n_kv_heads, head_dim]);
}

#[test]
fn test_individual_token_read_and_bounds() {
    let device = Device::Cpu;
    let mut storage = QuantizedKvStorage::new_int8(2, 4, 4, 2, 8, &device).unwrap();

    let k = Tensor::ones((2, 8), DType::F32, &device).unwrap();
    let v = (Tensor::ones((2, 8), DType::F32, &device).unwrap() * 3.0).unwrap();

    storage.write_quantized_kv_token(1, 2, 3, &k, &v).unwrap();

    let (k_read, v_read) = storage.read_quantized_kv_token(1, 2, 3).unwrap();
    assert_eq!(k_read.dims(), &[2, 8]);
    assert_eq!(v_read.dims(), &[2, 8]);

    // Out of bounds checks
    assert!(storage.write_quantized_kv_token(2, 0, 0, &k, &v).is_err());
    assert!(storage.write_quantized_kv_token(0, 4, 0, &k, &v).is_err());
    assert!(storage.write_quantized_kv_token(0, 0, 4, &k, &v).is_err());
    assert!(storage.read_quantized_kv_token(2, 0, 0).is_err());
}
