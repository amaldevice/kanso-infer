#[cfg(target_os = "macos")]
mod paged_attention_v2_tests {
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::paged_attention::{
        PagedAttentionParams, cpu_paged_attention, gpu_paged_attention_v2,
        gpu_paged_attention_v2_buffer,
    };

    #[test]
    fn test_cpu_paged_attention_reference_basic() {
        let num_seqs = 1;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let block_size = 2;
        let max_num_blocks_per_seq = 2;
        let context_len = 3;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale,
        };

        let q = vec![1.0f32, 0.0, 1.0, 0.0];
        // 4 physical blocks in pool
        let num_blocks = 4;
        let mut k_cache = vec![0.0f32; num_blocks * block_size * num_kv_heads * head_dim];
        let mut v_cache = vec![0.0f32; num_blocks * block_size * num_kv_heads * head_dim];

        // Logical block 0 -> Physical block 2
        // Logical block 1 -> Physical block 1
        let block_tables = vec![2u32, 1u32];
        let context_lens = vec![context_len as u32];

        // Fill token 0 in block 2, offset 0: K = [1, 0, 0, 0], V = [10, 0, 0, 0]
        let offset_t0 = (2 * block_size) * (num_kv_heads * head_dim);
        k_cache[offset_t0] = 1.0;
        v_cache[offset_t0] = 10.0;

        // Fill token 1 in block 2, offset 1: K = [0, 0, 1, 0], V = [0, 20, 0, 0]
        let offset_t1 = (2 * block_size + 1) * (num_kv_heads * head_dim);
        k_cache[offset_t1 + 2] = 1.0;
        v_cache[offset_t1 + 1] = 20.0;

        // Fill token 2 in block 1, offset 0: K = [1, 0, 1, 0], V = [0, 0, 30, 0]
        let offset_t2 = block_size * (num_kv_heads * head_dim);
        k_cache[offset_t2] = 1.0;
        k_cache[offset_t2 + 2] = 1.0;
        v_cache[offset_t2 + 2] = 30.0;

        let out = cpu_paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );
        assert_eq!(out.len(), num_seqs * num_heads * head_dim);

        // Scores:
        // S0 = Q . K0 * scale = 1.0 * 0.5 = 0.5
        // S1 = Q . K1 * scale = 1.0 * 0.5 = 0.5
        // S2 = Q . K2 * scale = 2.0 * 0.5 = 1.0
        // max = 1.0
        // exp(S0 - 1.0) = exp(-0.5) = 0.60653066
        // exp(S1 - 1.0) = exp(-0.5) = 0.60653066
        // exp(S2 - 1.0) = exp(0.0) = 1.0
        // sum_exp = 2.2130613
        // w0 = 0.2740704
        // w1 = 0.2740704
        // w2 = 0.4518592
        // out[0] = w0 * 10 = 2.740704
        // out[1] = w1 * 20 = 5.481408
        // out[2] = w2 * 30 = 13.555776
        // out[3] = 0
        assert!((out[0] - 2.7407).abs() < 1e-3);
        assert!((out[1] - 5.4814).abs() < 1e-3);
        assert!((out[2] - 13.5558).abs() < 1e-3);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn test_fused_paged_attention_v2_numerical_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let num_seqs = 1;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 64;
        let block_size = 16;
        let max_num_blocks_per_seq = 4;
        let context_len = 35; // spans 3 blocks
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale,
        };

        // Pseudo-random inputs with deterministic seed
        let mut q = vec![0.0f32; num_seqs * num_heads * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 0.125;
        }

        let num_blocks = 8;
        let total_kv_elements = num_blocks * block_size * num_kv_heads * head_dim;
        let mut k_cache = vec![0.0f32; total_kv_elements];
        let mut v_cache = vec![0.0f32; total_kv_elements];

        for i in 0..total_kv_elements {
            k_cache[i] = ((i % 19) as f32 - 9.0) * 0.1;
            v_cache[i] = ((i % 23) as f32 - 11.0) * 0.15;
        }

        // Non-contiguous physical block mapping: logical 0->5, 1->2, 2->7, 3->0
        let block_tables = vec![5u32, 2u32, 7u32, 0u32];
        let context_lens = vec![context_len as u32];

        let cpu_out = cpu_paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );

        let gpu_out = gpu_paged_attention_v2(
            &ctx,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        )
        .expect("GPU paged attention v2 failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        for i in 0..cpu_out.len() {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff <= 1e-4,
                "Numerical parity mismatch at index {}: gpu={}, cpu={}, diff={}",
                i,
                gpu_out[i],
                cpu_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_fused_paged_attention_v2_gqa_implicit_mapping() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // GQA: 8 query heads, 2 KV heads (factor of 4)
        let num_seqs = 1;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 32;
        let block_size = 8;
        let max_num_blocks_per_seq = 4;
        let context_len = 20;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale,
        };

        let mut q = vec![0.0f32; num_seqs * num_heads * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.1;
        }

        let num_blocks = 6;
        let total_kv_elements = num_blocks * block_size * num_kv_heads * head_dim;
        let mut k_cache = vec![0.0f32; total_kv_elements];
        let mut v_cache = vec![0.0f32; total_kv_elements];

        for i in 0..total_kv_elements {
            k_cache[i] = ((i % 29) as f32 - 14.0) * 0.05;
            v_cache[i] = ((i % 31) as f32 - 15.0) * 0.05;
        }

        let block_tables = vec![3u32, 1u32, 4u32, 0u32];
        let context_lens = vec![context_len as u32];

        let cpu_out = cpu_paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );

        let gpu_out = gpu_paged_attention_v2(
            &ctx,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        )
        .expect("GPU paged attention v2 GQA failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        for i in 0..cpu_out.len() {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff <= 1e-4,
                "GQA parity mismatch at {}: gpu={}, cpu={}, diff={}",
                i,
                gpu_out[i],
                cpu_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_fused_paged_attention_v2_multi_seq_batched() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let num_seqs = 3;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let block_size = 16;
        let max_num_blocks_per_seq = 4;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale,
        };

        let mut q = vec![0.0f32; num_seqs * num_heads * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 11) as f32 - 5.0) * 0.1;
        }

        let num_blocks = 12;
        let total_kv = num_blocks * block_size * num_kv_heads * head_dim;
        let mut k_cache = vec![0.0f32; total_kv];
        let mut v_cache = vec![0.0f32; total_kv];
        for i in 0..total_kv {
            k_cache[i] = ((i % 7) as f32 - 3.0) * 0.2;
            v_cache[i] = ((i % 13) as f32 - 6.0) * 0.1;
        }

        // 3 sequences with different block allocations:
        // seq 0: blocks [1, 5] (len 25)
        // seq 1: blocks [0, 8, 2] (len 40)
        // seq 2: blocks [10] (len 8)
        let block_tables = vec![
            1u32, 5, 0, 0, // seq 0
            0, 8, 2, 0, // seq 1
            10, 0, 0, 0, // seq 2
        ];
        let context_lens = vec![25u32, 40, 8];

        let cpu_out = cpu_paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );

        let gpu_out = gpu_paged_attention_v2(
            &ctx,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        )
        .expect("Batched GPU paged attention failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        for i in 0..cpu_out.len() {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff <= 1e-4,
                "Batched parity mismatch at {}: gpu={}, cpu={}, diff={}",
                i,
                gpu_out[i],
                cpu_out[i],
                diff
            );
        }
    }

    /// Ticket #51 acceptance: parity across head_dim {64, 128, 256} x GQA ratio {1, 4, 5, 8}.
    /// head_dim > 128 used to read `q_local`/`s_k` out of bounds silently.
    #[test]
    fn test_paged_attention_v2_parity_matrix() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        for &head_dim in &[64usize, 128, 256] {
            for &group_size in &[1usize, 4, 5, 8] {
                let num_seqs = 2usize;
                let num_kv_heads = 2usize;
                let num_heads = num_kv_heads * group_size;
                let block_size = 16usize;
                let max_num_blocks_per_seq = 4usize;
                let num_blocks = 10usize;
                let context_lens = vec![37u32, 16u32];
                let scale = 1.0f32 / (head_dim as f32).sqrt();

                let params = PagedAttentionParams {
                    num_seqs: num_seqs as u32,
                    num_heads: num_heads as u32,
                    num_kv_heads: num_kv_heads as u32,
                    head_dim: head_dim as u32,
                    block_size: block_size as u32,
                    max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
                    scale,
                };

                let mut q = vec![0.0f32; num_seqs * num_heads * head_dim];
                for (i, v) in q.iter_mut().enumerate() {
                    *v = ((i % 17) as f32 - 8.0) * 0.125;
                }

                let total_kv = num_blocks * block_size * num_kv_heads * head_dim;
                let mut k_cache = vec![0.0f32; total_kv];
                let mut v_cache = vec![0.0f32; total_kv];
                for i in 0..total_kv {
                    k_cache[i] = ((i % 19) as f32 - 9.0) * 0.1;
                    v_cache[i] = ((i % 23) as f32 - 11.0) * 0.15;
                }

                // Non-contiguous physical block mapping (seq 1 only needs its first block).
                let block_tables = vec![5u32, 2, 7, 0, 3, 9, 1, 6];

                let cpu_out = cpu_paged_attention(
                    &q,
                    &k_cache,
                    &v_cache,
                    &block_tables,
                    &context_lens,
                    &params,
                );
                let gpu_out = gpu_paged_attention_v2(
                    &ctx,
                    &q,
                    &k_cache,
                    &v_cache,
                    &block_tables,
                    &context_lens,
                    &params,
                )
                .expect("GPU paged attention v2 dispatch failed");

                assert_eq!(gpu_out.len(), cpu_out.len());
                for i in 0..cpu_out.len() {
                    let diff = (gpu_out[i] - cpu_out[i]).abs();
                    assert!(
                        diff <= 1e-4,
                        "parity mismatch at head_dim={}, kv_group={}, idx={}: gpu={}, cpu={}, diff={}",
                        head_dim,
                        group_size,
                        i,
                        gpu_out[i],
                        cpu_out[i],
                        diff
                    );
                }
            }
        }
    }

    /// Ticket #51: head_dim beyond the kernel's register capacity must be rejected by the host.
    #[test]
    fn test_paged_attention_v2_rejects_over_max_head_dim() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let num_seqs = 1usize;
        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 384usize;
        let block_size = 16usize;
        let max_num_blocks_per_seq = 4usize;

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale: 1.0f32 / (head_dim as f32).sqrt(),
        };

        let q = vec![0.1f32; num_seqs * num_heads * head_dim];
        let kv_len = num_seqs * max_num_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_cache = vec![0.1f32; kv_len];
        let v_cache = vec![0.1f32; kv_len];
        let block_tables = vec![0u32, 1, 2, 3];
        let context_lens = vec![8u32];

        let result = gpu_paged_attention_v2(
            &ctx,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );
        assert!(
            result.is_err(),
            "head_dim={} exceeds the kernel limit and must fail loudly, got {:?}",
            head_dim,
            result.map(|v| v.len())
        );
    }

    /// Ticket #51: the stage-2 merge only matters when >1 partition holds live tokens.
    #[test]
    fn test_paged_attention_v2_multi_partition_merge_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let num_seqs = 1usize;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 128usize;
        let block_size = 16usize;
        let max_num_blocks_per_seq = 40usize; // capacity 640 -> 3 partitions, context spans all
        let num_blocks = 44usize;
        let context_lens = vec![600u32];
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as u32,
            scale,
        };

        let mut q = vec![0.0f32; num_seqs * num_heads * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 0.125;
        }
        let total_kv = num_blocks * block_size * num_kv_heads * head_dim;
        let mut k_cache = vec![0.0f32; total_kv];
        let mut v_cache = vec![0.0f32; total_kv];
        for i in 0..total_kv {
            k_cache[i] = ((i % 19) as f32 - 9.0) * 0.1;
            v_cache[i] = ((i % 23) as f32 - 11.0) * 0.15;
        }
        // Non-contiguous mapping: logical block b -> physical (7 * b + 3) % num_blocks.
        let block_tables: Vec<u32> = (0..max_num_blocks_per_seq as u32)
            .map(|b| (7 * b + 3) % num_blocks as u32)
            .collect();

        let cpu_out = cpu_paged_attention(
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        );
        let gpu_out = gpu_paged_attention_v2(
            &ctx,
            &q,
            &k_cache,
            &v_cache,
            &block_tables,
            &context_lens,
            &params,
        )
        .expect("GPU paged attention v2 dispatch failed");

        assert_eq!(gpu_out.len(), cpu_out.len());
        for i in 0..cpu_out.len() {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff <= 1e-4,
                "merge parity mismatch at idx={}: gpu={}, cpu={}, diff={}",
                i,
                gpu_out[i],
                cpu_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_fused_paged_attention_v2_vram_flat_scaling() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let num_seqs = 1;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 64;
        let block_size = 16;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Check context length 128 vs 2048:
        // For fused PagedAttention v2, the output buffer size is [num_seqs, num_heads, head_dim]
        // regardless of context length (128 vs 2048).
        // No unnormalized intermediate score matrix [num_heads, context_len] is written to VRAM!
        let output_elements_128 = num_seqs * num_heads * head_dim;
        let output_elements_2048 = num_seqs * num_heads * head_dim;

        assert_eq!(
            output_elements_128, output_elements_2048,
            "Output activations VRAM allocation must remain strictly O(1) flat relative to context length"
        );

        // Run with context_len = 2048 on GPU and verify successful execution and correct output shape
        let context_len = 2048;
        let blocks_needed = context_len / block_size; // 128 blocks
        let max_blocks = blocks_needed;
        let num_blocks = blocks_needed + 4;

        let params = PagedAttentionParams {
            num_seqs: num_seqs as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks_per_seq: max_blocks as u32,
            scale,
        };

        let q = vec![0.1f32; num_seqs * num_heads * head_dim];
        let total_kv = num_blocks * block_size * num_kv_heads * head_dim;
        let k_cache = vec![0.05f32; total_kv];
        let v_cache = vec![0.02f32; total_kv];
        let block_tables: Vec<u32> = (0..blocks_needed as u32).collect();
        let context_lens = vec![context_len as u32];

        // Allocate buffers explicitly using buffer API to measure and verify flat VRAM allocation
        let q_buf = ctx.alloc_buffer_with_data(&q);
        let k_buf = ctx.alloc_buffer_with_data(&k_cache);
        let v_buf = ctx.alloc_buffer_with_data(&v_cache);
        let bt_buf = ctx.alloc_buffer_with_data(&block_tables);
        let cl_buf = ctx.alloc_buffer_with_data(&context_lens);
        let out_buf = ctx.alloc_buffer::<f32>(num_seqs * num_heads * head_dim);

        // Verification of VRAM size of the output: exactly 4 * 4 * 64 * 4 bytes = 4096 bytes
        let out_bytes = out_buf.length();
        assert_eq!(out_bytes, (num_seqs * num_heads * head_dim * 4) as u64);

        gpu_paged_attention_v2_buffer(
            &ctx, &q_buf, &k_buf, &v_buf, &bt_buf, &cl_buf, &out_buf, &params,
        )
        .expect("Execution of 2048-token context length paged attention failed");

        let mut result = vec![0.0f32; num_seqs * num_heads * head_dim];
        unsafe {
            let ptr = out_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), result.len());
        }

        // Verify output is non-zero and non-NaN
        for &val in &result {
            assert!(!val.is_nan());
            assert!(!val.is_infinite());
            assert!(val > 0.0);
        }
    }
}
