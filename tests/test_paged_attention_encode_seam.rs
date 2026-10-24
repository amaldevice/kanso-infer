//! Ticket #109: the paged-attention Metal seam must split into a pure encoding
//! function driven by a caller-owned command buffer, with the existing entry points
//! keeping today's create-commit-wait behavior — one encoding logic, two submit
//! strategies (#86). This is the seam ticket #110 (one command buffer per step)
//! will encode attention through.
//!
//! What this pins, at the kernel-seam level the split lives at:
//! 1. encoding into a caller-owned command buffer produces output identical to the
//!    self-committing wrapper (bit-equal: both submit strategies run the same
//!    kernels on the same seeded inputs) at ctx 128 (one partition) and ctx 512
//!    (two partitions, so the stage-2 merge runs through the caller-owned path too);
//! 2. encoding creates no command buffer of its own — the only buffer on that path
//!    is the caller's, and commit/wait stays the caller's job;
//! 3. the wrapper still creates, commits and waits on exactly one command buffer
//!    of its own per call, as it does today.
//!
//! Inputs are StdRng-seeded mock tensors, never `ones` (Fase 0 convention): a
//! constant matrix collapses softmax weights and would hide a wrong-block read.

#[cfg(target_os = "macos")]
mod encode_seam {
    use kanso_infer::metal::context::MetalContext;
    use kanso_infer::metal::paged_attention::{
        PagedAttentionParams, encode_paged_attention_v2, gpu_paged_attention_v2_buffer,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2; // GQA group 4, the decode-path shape
    const HEAD_DIM: usize = 64;
    const BLOCK_SIZE: usize = 16;
    /// 512 spans two PARTITION_SIZE (=256) partitions, exercising the merge.
    const CONTEXTS: [usize; 2] = [128, 512];

    fn read_out(buf: &metal::Buffer, len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        unsafe {
            let ptr = buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
        }
        out
    }

    /// One seeded attention case: q/k/v plus the paged geometry both submit
    /// strategies run on.
    struct SeededCase {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        block_tables: Vec<u32>,
        context_lens: Vec<u32>,
        max_blocks: usize,
    }

    /// Seeded q/k/v plus a non-contiguous logical-to-physical block mapping, so
    /// identical output cannot come from reading the wrong blocks.
    fn case(ctx_len: usize, seed: u64) -> SeededCase {
        let blocks_needed = ctx_len.div_ceil(BLOCK_SIZE);
        let num_blocks = blocks_needed + 5; // slack keeps the mapping non-contiguous
        let mut rng = StdRng::seed_from_u64(seed);

        let q: Vec<f32> = (0..NUM_HEADS * HEAD_DIM)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let kv_len = num_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
        let k: Vec<f32> = (0..kv_len)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        let v: Vec<f32> = (0..kv_len)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        // logical block b -> physical (7b + 3) mod num_blocks
        let block_tables: Vec<u32> = (0..blocks_needed as u32)
            .map(|b| (7 * b + 3) % num_blocks as u32)
            .collect();
        SeededCase {
            q,
            k,
            v,
            block_tables,
            context_lens: vec![ctx_len as u32],
            max_blocks: blocks_needed,
        }
    }

    #[test]
    fn caller_owned_command_buffer_matches_self_committing_wrapper() {
        let ctx = MetalContext::new().expect("MetalContext initialization failed");

        for (i, &ctx_len) in CONTEXTS.iter().enumerate() {
            let case = case(ctx_len, 0x109 + i as u64);
            let params = PagedAttentionParams {
                num_seqs: 1,
                num_heads: NUM_HEADS as u32,
                num_kv_heads: NUM_KV_HEADS as u32,
                head_dim: HEAD_DIM as u32,
                block_size: BLOCK_SIZE as u32,
                max_num_blocks_per_seq: case.max_blocks as u32,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            };

            let q_buf = ctx.alloc_buffer_with_data(&case.q);
            let k_buf = ctx.alloc_buffer_with_data(&case.k);
            let v_buf = ctx.alloc_buffer_with_data(&case.v);
            let bt_buf = ctx.alloc_buffer_with_data(&case.block_tables);
            let cl_buf = ctx.alloc_buffer_with_data(&case.context_lens);
            let out_len = NUM_HEADS * HEAD_DIM;

            // Submit strategy 1 (today's public path): the wrapper creates, commits
            // and waits on its own command buffer.
            let wrapper_out = ctx.alloc_buffer::<f32>(out_len);
            let before = ctx.command_buffers_created();
            gpu_paged_attention_v2_buffer(
                &ctx,
                &q_buf,
                &k_buf,
                &v_buf,
                &bt_buf,
                &cl_buf,
                &wrapper_out,
                &params,
            )
            .expect("wrapper dispatch");
            assert_eq!(
                ctx.command_buffers_created() - before,
                1,
                "ctx {ctx_len}: the wrapper must keep create-commit-wait behavior: \
                 exactly one command buffer of its own per call"
            );

            // Submit strategy 2: encoding is driven by a caller-owned command
            // buffer. Poisoned output first, so a pass cannot be zeros inherited
            // from allocation.
            let caller_out = ctx.alloc_buffer::<f32>(out_len);
            unsafe {
                std::slice::from_raw_parts_mut(caller_out.contents() as *mut f32, out_len)
                    .fill(f32::NAN);
            }
            let before = ctx.command_buffers_created();
            let command_buffer = ctx.new_command_buffer(); // caller-owned
            let after_alloc = ctx.command_buffers_created();
            encode_paged_attention_v2(
                &ctx,
                command_buffer,
                &q_buf,
                &k_buf,
                &v_buf,
                &bt_buf,
                &cl_buf,
                &caller_out,
                &params,
            )
            .expect("caller-owned encoding");
            assert_eq!(
                ctx.command_buffers_created(),
                after_alloc,
                "ctx {ctx_len}: encoding must not create a command buffer of its own"
            );
            // Submission is the caller's job: the seam neither commits nor waits.
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(
                ctx.command_buffers_created() - before,
                1,
                "ctx {ctx_len}: the caller-owned path must spend exactly the one \
                 command buffer the caller created"
            );

            let wrapper = read_out(&wrapper_out, out_len);
            let caller = read_out(&caller_out, out_len);
            assert!(
                !caller.iter().any(|x| x.is_nan()),
                "ctx {ctx_len}: the kernel must have written through the caller-owned \
                 encoding (output still holds the poison value)"
            );
            assert_eq!(
                caller, wrapper,
                "ctx {ctx_len}: caller-owned encoding must be bit-identical to the \
                 self-committing wrapper (one encoding logic, #86)"
            );
        }
    }
}
