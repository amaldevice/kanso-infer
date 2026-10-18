mod draft_cache;
pub mod profiler;
pub mod sampling;
pub mod verifier;

pub use profiler::{SpeculativeProfiler, SpeculativeReport};
pub use verifier::{SpeculativeVerifier, VerificationResult};

use crate::engine::{Engine, StepOutput};
use crate::error::EngineError;
use crate::model::Model;
use crate::scheduler::SequenceStatus;
use draft_cache::DraftCache;

pub struct SpeculativeEngine {
    pub engine: Engine,
    pub draft_model: Option<Model>,
    pub gamma: usize,
    pub profiler: SpeculativeProfiler,
    /// The draft's own paged KV cache (#55), built on first use.
    draft_cache: Option<DraftCache>,
}

impl SpeculativeEngine {
    pub fn new(
        target_model: Model,
        draft_model: Option<Model>,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
        gamma: usize,
    ) -> Result<Self, EngineError> {
        let engine = Engine::new(target_model, max_running, total_blocks, block_size)?;
        Ok(Self {
            engine,
            draft_model,
            gamma,
            profiler: SpeculativeProfiler::new(),
            draft_cache: None,
        })
    }

    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        self.engine.add_request(prompt_tokens, max_tokens)
    }

    /// Same as [`Engine::add_request_with_sampling`] (#56 knobs). The sequence's sampling
    /// params drive both sides of speculation: the draft draws from `q` with them, and the
    /// verifier runs rejection sampling against the target's `p` when temperature > 0.
    pub fn add_request_with_sampling(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
        sampling: crate::scheduler::SamplingParams,
    ) -> u64 {
        self.engine
            .add_request_with_sampling(prompt_tokens, max_tokens, stop_token_ids, sampling)
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.engine.has_unfinished_requests()
    }

    pub fn waiting_count(&self) -> usize {
        self.engine.waiting_count()
    }

    pub fn running_count(&self) -> usize {
        self.engine.running_count()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.engine.free_blocks_count()
    }

    /// Free blocks in the draft model's own KV pool (`None` before the first speculative
    /// step, when the cache does not exist yet). After every request finishes it must be
    /// back to the full pool: rollback and release both return draft blocks.
    pub fn draft_free_blocks_count(&self) -> Option<usize> {
        self.draft_cache.as_ref().map(DraftCache::free_blocks_count)
    }

    pub fn step(&mut self) -> Result<Vec<StepOutput>, EngineError> {
        if self.draft_model.is_none() || self.gamma == 0 {
            return self.engine.step();
        }

        self.release_dead_draft_slots();

        let has_prefill = self.engine.sequences.iter().any(|s| {
            s.status == SequenceStatus::Waiting
                || (s.status == SequenceStatus::Running
                    && s.prefilled_tokens < s.prompt_tokens.len())
        });
        if has_prefill {
            return self.engine.step();
        }

        let mut outputs = Vec::new();
        let running_ids: Vec<u64> = self
            .engine
            .sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running && !s.is_finished())
            .map(|s| s.id)
            .collect();

        for req_id in running_ids {
            let (prefix, sampling) = {
                let seq = self
                    .engine
                    .sequences
                    .iter()
                    .find(|s| s.id == req_id)
                    .unwrap();
                (
                    [
                        seq.prompt_tokens.as_slice(),
                        seq.generated_tokens.as_slice(),
                    ]
                    .concat(),
                    seq.sampling,
                )
            };

            self.ensure_draft_cache()?;
            let proposals = {
                let gamma = self.gamma;
                let cache = self.draft_cache.as_mut().unwrap();
                let draft = self.draft_model.as_ref().unwrap();
                cache.propose(draft, req_id, &prefix, gamma, sampling)?
            };

            let start = std::time::Instant::now();
            let verif = self.engine.speculative_step_sampled(
                req_id,
                &proposals.tokens,
                &proposals.probs,
                sampling,
            )?;
            let duration = start.elapsed();
            self.profiler.record_step(
                proposals.tokens.len(),
                verif.num_accepted,
                duration,
                verif.num_rejected > 0,
            );

            let seq_after = self
                .engine
                .sequences
                .iter()
                .find(|s| s.id == req_id)
                .unwrap();
            let is_finished = seq_after.is_finished();
            if is_finished {
                self.draft_cache.as_mut().unwrap().release(req_id);
            } else {
                // Drop the rejected proposals; the last committed token stays uncached so
                // the next round feeds it and gets that round's first proposal for free.
                let keep = seq_after.prompt_tokens.len() + seq_after.generated_tokens.len() - 1;
                self.draft_cache.as_mut().unwrap().rollback(req_id, keep);
            }

            for (i, &tok) in verif.committed_tokens.iter().enumerate() {
                let tok_finished = is_finished && (i == verif.committed_tokens.len() - 1);
                outputs.push(StepOutput {
                    request_id: req_id,
                    token_id: tok,
                    is_finished: tok_finished,
                });
            }
        }

        // Retirement stage: free KV blocks for finished sequences
        for seq in self.engine.sequences.iter_mut() {
            if seq.is_finished() && !seq.block_table.blocks().is_empty() {
                for &blk in seq.block_table.blocks() {
                    self.engine.block_pool.free(blk);
                }
                seq.block_table = crate::kv_cache::BlockTable::new();
            }
        }

        Ok(outputs)
    }

    /// Builds the draft cache on first use: it needs the target's block geometry and the
    /// draft's own layer/head shape, plus one vocabulary shared with the target (rejection
    /// sampling compares the two models' distributions token by token).
    fn ensure_draft_cache(&mut self) -> Result<(), EngineError> {
        if self.draft_cache.is_none() {
            let draft = self.draft_model.as_ref().unwrap();
            if draft.config.vocab_size != self.engine.model.config.vocab_size {
                return Err(EngineError::Model(format!(
                    "draft vocab {} does not match target vocab {}",
                    draft.config.vocab_size, self.engine.model.config.vocab_size
                )));
            }
            self.draft_cache = Some(DraftCache::new(
                draft,
                self.engine.block_pool.total_blocks(),
                self.engine.block_size,
            )?);
        }
        Ok(())
    }

    /// Returns the draft cache of every request that is no longer live to the draft pool:
    /// finished sequences, and sequences the caller already dropped with
    /// `Engine::drain_finished` (a finished cache would otherwise hold its blocks until the
    /// engine itself is gone).
    fn release_dead_draft_slots(&mut self) {
        let live: Vec<u64> = self
            .engine
            .sequences
            .iter()
            .filter(|s| !s.is_finished())
            .map(|s| s.id)
            .collect();
        if let Some(cache) = self.draft_cache.as_mut() {
            let dead: Vec<u64> = cache.slot_ids().filter(|id| !live.contains(id)).collect();
            for id in dead {
                cache.release(id);
            }
        }
    }
}
