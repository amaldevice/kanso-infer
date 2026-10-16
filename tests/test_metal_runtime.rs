#[cfg(target_os = "macos")]
mod macos_tests {
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::ops::{
        cpu_rmsnorm, cpu_rope, cpu_silu_gate_multiply, gpu_rmsnorm, gpu_rmsnorm_buffer, gpu_rope,
        gpu_rope_buffer, gpu_silu_gate_multiply,
    };
    use metal::MTLStorageMode;

    #[test]
    fn test_metal_device_init_and_command_queue() {
        let ctx = MetalContext::new().expect("MetalContext initialization should succeed on macOS");
        // Ensure command queue is initialized and can create command buffer
        let cmd_buf = ctx.queue.new_command_buffer();
        assert_eq!(cmd_buf.status(), metal::MTLCommandBufferStatus::NotEnqueued);
    }

    #[test]
    fn test_candle_metal_device_creation() {
        let dev = candle_core::Device::new_metal(0)
            .expect("Candle Metal device initialization should succeed on macOS");
        let t = candle_core::Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let sum = t.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_candle_metal_gguf_loading() {
        let dev = candle_core::Device::new_metal(0)
            .expect("Candle Metal device initialization should succeed on macOS");
        if let Ok(model_path) =
            kanso_infer::weights::resolve_model_path("smollm2-360m-instruct-q8_0.gguf")
        {
            let res = kanso_infer::model::Model::load_gguf(&model_path, dev);
            assert!(
                res.is_ok(),
                "Model::load_gguf with Metal device failed: {:?}",
                res.err()
            );
        }
    }
    #[test]
    fn test_metal_buffer_shared_storage_mode() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // Test uninitialized shared buffer allocation
        let buf = ctx.alloc_buffer::<f32>(128);
        assert_eq!(
            buf.storage_mode(),
            MTLStorageMode::Shared,
            "Buffer storage mode must be MTLStorageModeShared"
        );

        // Zero-copy direct pointer write and read
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, 128) };
        slice[0] = 42.0;
        slice[127] = 99.5;

        let read_slice = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, 128) };
        assert_eq!(read_slice[0], 42.0);
        assert_eq!(read_slice[127], 99.5);

        // Test shared buffer with data initialization
        let input_data = vec![1.5f32, 2.5, 3.5, 4.5];
        let buf_data = ctx.alloc_buffer_with_data(&input_data);
        assert_eq!(
            buf_data.storage_mode(),
            MTLStorageMode::Shared,
            "Allocated buffer must be MTLStorageModeShared"
        );
        let read_data = unsafe { std::slice::from_raw_parts(buf_data.contents() as *const f32, 4) };
        assert_eq!(read_data, &input_data[..]);
    }

    #[test]
    fn test_metal_pipeline_state_caching() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        // Verify pre-compiled pipeline states are cached at startup
        assert!(
            ctx.pipelines.contains("rmsnorm"),
            "Pipeline cache missing rmsnorm"
        );
        assert!(
            ctx.pipelines.contains("rope"),
            "Pipeline cache missing rope"
        );
        assert!(
            ctx.pipelines.contains("silu_gate"),
            "Pipeline cache missing silu_gate"
        );
        assert!(
            ctx.pipelines.contains("paged_kv"),
            "Pipeline cache missing paged_kv"
        );

        let rmsnorm_pipeline = ctx.pipelines.get("rmsnorm");
        assert!(
            rmsnorm_pipeline.is_some(),
            "rmsnorm pipeline state should be cached"
        );

        let rope_pipeline = ctx.pipelines.get("rope");
        assert!(
            rope_pipeline.is_some(),
            "rope pipeline state should be cached"
        );

        let silu_gate_pipeline = ctx.pipelines.get("silu_gate");
        assert!(
            silu_gate_pipeline.is_some(),
            "silu_gate pipeline state should be cached"
        );

        let paged_kv_pipeline = ctx.pipelines.get("paged_kv");
        assert!(
            paged_kv_pipeline.is_some(),
            "paged_kv pipeline state should be cached"
        );
    }

    #[test]
    fn test_metal_rmsnorm_numerical_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let weight = vec![1.0f32; 8];
        let eps = 1e-5f32;

        let cpu_result = cpu_rmsnorm(&input, &weight, eps);
        let gpu_result =
            gpu_rmsnorm(&ctx, &input, &weight, eps).expect("GPU rmsnorm should succeed");

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (&g, &c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff <= 1e-4,
                "RMSNorm mismatch at index {}: gpu={}, cpu={}, diff={}",
                i,
                g,
                c,
                diff
            );
        }
    }

    #[test]
    fn test_metal_rope_numerical_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let head_dim = 64;
        let num_heads = 4;
        let total_elements = num_heads * head_dim;
        let pos = 3;

        // Populate mock input activations
        let mut q_k_cpu: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let mut q_k_gpu = q_k_cpu.clone();

        // Mock cos/sin tables: pos * half_dim + idx
        let half_dim = head_dim / 2;
        let table_len = (pos + 1) * half_dim;
        let cos_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.05).cos()).collect();
        let sin_table: Vec<f32> = (0..table_len).map(|i| (i as f32 * 0.05).sin()).collect();

        // CPU execution
        cpu_rope(&mut q_k_cpu, &cos_table, &sin_table, head_dim, pos);

        // GPU execution
        gpu_rope(&ctx, &mut q_k_gpu, &cos_table, &sin_table, head_dim, pos)
            .expect("GPU rope should succeed");

        for (i, (&g, &c)) in q_k_gpu.iter().zip(q_k_cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff <= 1e-4,
                "RoPE mismatch at index {}: gpu={}, cpu={}, diff={}",
                i,
                g,
                c,
                diff
            );
        }
    }

    #[test]
    fn test_metal_silu_gate_numerical_parity() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let gate = vec![-1.0f32, 0.0, 0.5, 1.0, 2.0];
        let up = vec![0.5f32, 1.0, 1.5, 2.0, 2.5];

        let cpu_result = cpu_silu_gate_multiply(&gate, &up);
        let gpu_result =
            gpu_silu_gate_multiply(&ctx, &gate, &up).expect("GPU silu_gate should succeed");

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (&g, &c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff <= 1e-4,
                "SiLU gate mismatch at index {}: gpu={}, cpu={}, diff={}",
                i,
                g,
                c,
                diff
            );
        }
    }

    #[test]
    fn test_metal_zero_copy_roundtrip() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        let dim = 128;
        let num_rows = 2;
        let count = num_rows * dim;

        // Allocate shared buffers
        let in_buf = ctx.alloc_buffer::<f32>(count);
        let weight_buf = ctx.alloc_buffer::<f32>(dim);
        let out_buf = ctx.alloc_buffer::<f32>(count);

        assert_eq!(in_buf.storage_mode(), MTLStorageMode::Shared);
        assert_eq!(weight_buf.storage_mode(), MTLStorageMode::Shared);
        assert_eq!(out_buf.storage_mode(), MTLStorageMode::Shared);

        // Direct slice write to input and weight buffers (zero host-to-device copy)
        unsafe {
            let in_slice = std::slice::from_raw_parts_mut(in_buf.contents() as *mut f32, count);
            for (i, v) in in_slice.iter_mut().enumerate() {
                *v = (i as f32 + 1.0) * 0.02;
            }

            let w_slice = std::slice::from_raw_parts_mut(weight_buf.contents() as *mut f32, dim);
            for v in w_slice.iter_mut() {
                *v = 1.0;
            }
        }

        // Dispatch directly on shared buffers
        let eps = 1e-5f32;
        gpu_rmsnorm_buffer(&ctx, &in_buf, &weight_buf, &out_buf, num_rows, dim, eps)
            .expect("gpu_rmsnorm_buffer should succeed");

        // Read directly from out_buf (zero device-to-host copy)
        let out_slice =
            unsafe { std::slice::from_raw_parts(out_buf.contents() as *const f32, count) };

        // Verify against CPU reference for each row
        unsafe {
            let in_slice = std::slice::from_raw_parts(in_buf.contents() as *const f32, count);
            let w_slice = std::slice::from_raw_parts(weight_buf.contents() as *const f32, dim);

            for row in 0..num_rows {
                let row_in = &in_slice[row * dim..(row + 1) * dim];
                let row_out = &out_slice[row * dim..(row + 1) * dim];
                let expected = cpu_rmsnorm(row_in, w_slice, eps);

                for (i, (&actual, &exp)) in row_out.iter().zip(expected.iter()).enumerate() {
                    let diff = (actual - exp).abs();
                    assert!(
                        diff <= 1e-4,
                        "Zero-copy RMSNorm mismatch at row {}, idx {}: actual={}, expected={}, diff={}",
                        row,
                        i,
                        actual,
                        exp,
                        diff
                    );
                }
            }
        }

        // Zero-copy RoPE in-place dispatch
        let head_dim = 64;
        let num_heads = 2;
        let total_rope_elements = num_heads * head_dim;
        let rope_buf = ctx.alloc_buffer::<f32>(total_rope_elements);
        assert_eq!(rope_buf.storage_mode(), MTLStorageMode::Shared);

        let mut expected_rope: Vec<f32> = (0..total_rope_elements)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                rope_buf.contents() as *mut f32,
                total_rope_elements,
            );
            slice.copy_from_slice(&expected_rope);
        }

        let half_dim = head_dim / 2;
        let cos_table: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).cos()).collect();
        let sin_table: Vec<f32> = (0..half_dim).map(|i| (i as f32 * 0.05).sin()).collect();
        let cos_buf = ctx.alloc_buffer_with_data(&cos_table);
        let sin_buf = ctx.alloc_buffer_with_data(&sin_table);

        cpu_rope(&mut expected_rope, &cos_table, &sin_table, head_dim, 0);
        gpu_rope_buffer(
            &ctx,
            &rope_buf,
            &cos_buf,
            &sin_buf,
            total_rope_elements,
            head_dim,
            0,
        )
        .expect("gpu_rope_buffer should succeed");

        let actual_rope = unsafe {
            std::slice::from_raw_parts(rope_buf.contents() as *const f32, total_rope_elements)
        };
        for (i, (&actual, &exp)) in actual_rope.iter().zip(expected_rope.iter()).enumerate() {
            let diff = (actual - exp).abs();
            assert!(
                diff <= 1e-4,
                "Zero-copy RoPE mismatch at idx {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
    }
}
