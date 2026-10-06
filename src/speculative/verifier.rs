use candle_core::IndexOp;
use crate::error::EngineError;
use crate::kv_cache::{BlockPool, BlockTable, KvStorage};
use crate::model::Model;
use crate::sampler::sample_token;

#[derive(Debug, Clone, PartialEq)]
pub struct VerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub emitted_token: u32,
    pub num_accepted: usize,
    pub num_rejected: usize,
    pub committed_tokens: Vec<u32>,
    pub rolled_back_blocks: Vec<usize>,
}

pub struct SpeculativeVerifier;

impl SpeculativeVerifier {
    pub fn verify(
        model: &Model,
        current_token: u32,
        candidates: &[u32],
        start_pos: usize,
        block_table: &mut BlockTable,
        block_pool: &mut BlockPool,
        kv_storage: &mut KvStorage,
        block_size: usize,
    ) -> Result<VerificationResult, EngineError> {
        if candidates.is_empty() {
            let needed_blocks = (start_pos + 1 + block_size - 1) / block_size;
            while block_table.blocks().len() < needed_blocks {
                let blk = block_pool.allocate().ok_or_else(|| {
                    EngineError::Model("Out of memory: cannot allocate KV block".to_string())
                })?;
                block_table.push_block(blk);
            }
            let logits = model.forward_decode_step(current_token, start_pos, block_table, kv_storage)?;
            let emitted = sample_token(&logits, 0.0, 1.0, 42)?;
            return Ok(VerificationResult {
                accepted_tokens: Vec::new(),
                emitted_token: emitted,
                num_accepted: 0,
                num_rejected: 0,
                committed_tokens: vec![emitted],
                rolled_back_blocks: Vec::new(),
            });
        }

        let k = candidates.len();
        let total_candidate_tokens = k + 1;
        let verify_tokens: Vec<u32> = std::iter::once(current_token)
            .chain(candidates.iter().copied())
            .collect();

        // Ensure sufficient blocks are allocated for speculative tokens
        let needed_blocks = (start_pos + total_candidate_tokens + block_size - 1) / block_size;
        while block_table.blocks().len() < needed_blocks {
            let blk = block_pool.allocate().ok_or_else(|| {
                EngineError::Model("Out of memory: cannot allocate KV block".to_string())
            })?;
            block_table.push_block(blk);
        }

        // Execute single forward pass verification
        let logits = model.forward_verify_tokens(
            &verify_tokens,
            start_pos,
            block_table,
            kv_storage,
        )?;

        // Verify draft candidates greedily
        let mut accepted_tokens = Vec::new();
        let mut emitted_token = None;
        let mut num_accepted = 0;
        let mut num_rejected = 0;

        for (idx, &cand) in candidates.iter().enumerate() {
            let token_logits = logits.i(idx)?;
            let target_pred = sample_token(&token_logits, 0.0, 1.0, 42)?;
            if cand == target_pred {
                accepted_tokens.push(cand);
                num_accepted += 1;
            } else {
                emitted_token = Some(target_pred);
                num_rejected = candidates.len() - num_accepted;
                break;
            }
        }

        if emitted_token.is_none() {
            // All K candidates accepted! Emit bonus token from index K
            let bonus_logits = logits.i(k)?;
            let bonus_pred = sample_token(&bonus_logits, 0.0, 1.0, 42)?;
            emitted_token = Some(bonus_pred);
        }

        let emitted = emitted_token.unwrap();

        // Valid tokens: start_pos + 1 (current_token) + num_accepted + 1 (emitted token)
        let valid_tokens = start_pos + 1 + num_accepted + 1;
        let rolled_back_blocks = block_table.rollback_blocks(valid_tokens, block_size, block_pool);

        let mut committed_tokens = accepted_tokens.clone();
        committed_tokens.push(emitted);

        Ok(VerificationResult {
            accepted_tokens,
            emitted_token: emitted,
            num_accepted,
            num_rejected,
            committed_tokens,
            rolled_back_blocks,
        })
    }
}
