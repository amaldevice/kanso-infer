use crate::error::EngineError;
use crate::kv_cache::kind::slot_for as kanso_slot_for;
use crate::kv_cache::{BlockPool, BlockTable, KvStorageKind, PrefixCache};
use crate::model::Model;
use crate::sampler::{greedy_token, sample_from_probs, softmax_probs};
use crate::scheduler::SamplingParams;
use crate::speculative::sampling::{acceptance_prob, residual_probs};
use candle_core::{IndexOp, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    /// Verifies `candidates` against the target model in one batched forward pass.
    ///
    /// Greedy requests (`sampling.temperature <= 0`) accept a candidate when it equals the
    /// target's argmax. Temperature > 0 requests run standard rejection sampling instead
    /// (#55): accept with `min(1, p/q)`, otherwise emit a draw from `max(0, p - q)`, which
    /// reproduces the target distribution exactly — that needs `draft_probs[i]`, the
    /// distribution the draft drew `candidates[i]` from.
    #[allow(clippy::too_many_arguments)] // #57: public API; regrouping params out of scope
    pub fn verify(
        model: &Model,
        current_token: u32,
        candidates: &[u32],
        draft_probs: &[Vec<f32>],
        sampling: SamplingParams,
        start_pos: usize,
        block_table: &mut BlockTable,
        block_pool: &mut BlockPool,
        prefix_cache: &mut PrefixCache,
        kv_storage: &mut KvStorageKind,
        block_size: usize,
    ) -> Result<VerificationResult, EngineError> {
        if candidates.is_empty() {
            let needed_blocks = (start_pos + 1).div_ceil(block_size);
            while block_table.blocks().len() < needed_blocks {
                let blk = block_pool.allocate().ok_or_else(|| {
                    EngineError::Model("Out of memory: cannot allocate KV block".to_string())
                })?;
                // #44/F3: block is about to be overwritten — drop stale prefix entries.
                prefix_cache.invalidate_block(blk);
                block_table.push_block(blk);
            }
            let logits =
                model.forward_decode_step(current_token, start_pos, block_table, kv_storage)?;
            let mut rng = StdRng::seed_from_u64(sampling.seed);
            let (emitted, _) = draw(&logits, sampling, &mut rng)?;
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
        let needed_blocks = (start_pos + total_candidate_tokens).div_ceil(block_size);
        while block_table.blocks().len() < needed_blocks {
            let blk = block_pool.allocate().ok_or_else(|| {
                EngineError::Model("Out of memory: cannot allocate KV block".to_string())
            })?;
            // #44/F3: block is about to be overwritten — drop stale prefix entries.
            prefix_cache.invalidate_block(blk);
            block_table.push_block(blk);
        }

        // Execute single forward pass verification. Ticket #90: the candidate block rides
        // `forward_batch` as one sequence of the step's batch — `[current, candidates...]`
        // is contiguous and ascending from `start_pos`, and the whole block needs its own
        // logits row (one per candidate plus the bonus position).
        let positions: Vec<usize> = (start_pos..start_pos + total_candidate_tokens).collect();
        let physical_blocks = block_table.blocks();
        let slot_mapping: Vec<usize> = positions
            .iter()
            .map(|&pos| {
                kanso_slot_for(
                    physical_blocks[pos / block_size],
                    pos % block_size,
                    block_size,
                )
            })
            .collect();
        let logits = model.forward_batch_all_rows(
            &verify_tokens,
            &positions,
            &slot_mapping,
            kv_storage,
            &[physical_blocks.to_vec()],
            &[start_pos + total_candidate_tokens],
            &[total_candidate_tokens],
        )?;

        let mut accepted_tokens = Vec::new();
        let mut emitted_token = None;
        let mut num_accepted = 0;
        let mut num_rejected = 0;
        // Seeded per request (ticket #61 F9): the same round replays exactly instead of
        // drawing from whatever entropy the thread happened to have.
        let mut rng = StdRng::seed_from_u64(sampling.seed);

        for (idx, &cand) in candidates.iter().enumerate() {
            let (target_pred, p) = draw(&logits.i(idx)?, sampling, &mut rng)?;
            if sampling.temperature <= 0.0 {
                if cand == target_pred {
                    accepted_tokens.push(cand);
                    num_accepted += 1;
                } else {
                    emitted_token = Some(target_pred);
                    num_rejected = candidates.len() - num_accepted;
                    break;
                }
                continue;
            }

            let q = draft_probs.get(idx).ok_or_else(|| {
                EngineError::Model(format!(
                    "temperature > 0 speculation needs the draft distribution of candidate {idx}"
                ))
            })?;
            if q.len() != p.len() {
                return Err(EngineError::Model(format!(
                    "draft distribution of candidate {idx} covers {} tokens, the target's {}",
                    q.len(),
                    p.len()
                )));
            }
            if rng.random::<f32>() <= acceptance_prob(&p, q, cand) {
                accepted_tokens.push(cand);
                num_accepted += 1;
            } else {
                // Rejected: correct the draft with the residual max(0, p - q).
                emitted_token = Some(sample_from_probs(&residual_probs(&p, q)?, &mut rng));
                num_rejected = candidates.len() - num_accepted;
                break;
            }
        }

        if emitted_token.is_none() {
            // All K candidates accepted! Emit bonus token from index K
            let (bonus_pred, _) = draw(&logits.i(k)?, sampling, &mut rng)?;
            emitted_token = Some(bonus_pred);
        }

        let emitted = emitted_token.unwrap();

        // Valid tokens: start_pos + 1 (current_token) + num_accepted + 1 (emitted token)
        let valid_tokens = start_pos + 1 + num_accepted + 1;
        let rolled_back_blocks = block_table.rollback_blocks(valid_tokens, block_size, block_pool);

        // Upon rejection, write target model's corrected token K and V at the rollback position
        if num_rejected > 0 {
            let rollback_pos = start_pos + 1 + num_accepted;
            let _ = model.forward_decode_step(emitted, rollback_pos, block_table, kv_storage)?;
        }
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

/// Draws one token from `logits` under `sampling`, plus the distribution it came from when
/// the request is not greedy (the rejection test and the residual both need `p`). The draw
/// comes from the caller's seeded stream, so a round is reproducible (ticket #61 F9).
fn draw(
    logits: &Tensor,
    sampling: SamplingParams,
    rng: &mut StdRng,
) -> Result<(u32, Vec<f32>), EngineError> {
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    if sampling.temperature <= 0.0 {
        return Ok((greedy_token(&flat), Vec::new()));
    }
    let probs = softmax_probs(&flat, sampling.temperature, sampling.top_p)?;
    Ok((sample_from_probs(&probs, rng), probs))
}
