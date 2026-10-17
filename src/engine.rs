use crate::error::EngineError;
use crate::kv_cache::{BlockPool, BlockTable, KvStorage, PrefixCache};
use crate::model::Model;
use crate::sampler::sample_token;
use crate::scheduler::{SamplingParams, Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: u64,
    pub token_id: u32,
    pub is_finished: bool,
}
#[derive(Debug, Clone)]
pub struct ScratchpadWorkspace {
    pub max_batch: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub hidden_states: Vec<f32>,
    pub normed_states: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ScratchpadWorkspace {
    pub fn new(
        max_batch: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            max_batch,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            hidden_states: vec![0.0f32; max_batch * hidden_dim],
            normed_states: vec![0.0f32; max_batch * hidden_dim],
            q_proj: vec![0.0f32; max_batch * hidden_dim],
            k_proj: vec![0.0f32; max_batch * hidden_dim],
            v_proj: vec![0.0f32; max_batch * hidden_dim],
            attn_out: vec![0.0f32; max_batch * hidden_dim],
            ffn_gate: vec![0.0f32; max_batch * intermediate_dim],
            ffn_up: vec![0.0f32; max_batch * intermediate_dim],
            ffn_down: vec![0.0f32; max_batch * hidden_dim],
            logits: vec![0.0f32; max_batch * vocab_size],
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.hidden_states.fill(0.0);
        self.normed_states.fill(0.0);
        self.q_proj.fill(0.0);
        self.k_proj.fill(0.0);
        self.v_proj.fill(0.0);
        self.attn_out.fill(0.0);
        self.ffn_gate.fill(0.0);
        self.ffn_up.fill(0.0);
        self.ffn_down.fill(0.0);
        self.logits.fill(0.0);
    }
}

pub struct Engine {
    pub model: Model,
    pub block_pool: BlockPool,
    pub prefix_cache: PrefixCache,
    pub sequences: Vec<Sequence>,
    pub max_running: usize,
    pub block_size: usize,
    pub token_budget: usize,
    pub current_step: usize,
    pub next_request_id: u64,
    pub kv_storage: KvStorage,
    pub workspace: ScratchpadWorkspace,
    pub prefix_matched_tokens: usize,
    pub total_prompt_tokens_admitted: usize,
    pub prefix_cache_hits: usize,
    pub prefix_cache_lookups: usize,
}

impl Engine {
    pub fn new(model: Model, max_running: usize, total_blocks: usize, block_size: usize) -> Self {
        Self::new_with_budget(model, max_running, total_blocks, block_size, usize::MAX)
    }

    pub fn new_with_budget(
        model: Model,
        max_running: usize,
        total_blocks: usize,
        block_size: usize,
        token_budget: usize,
    ) -> Self {
        let kv_storage = KvStorage::new(
            model.config.num_hidden_layers,
            total_blocks,
            block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            &model.device,
        )
        .expect("failed to initialize kv storage");

        let workspace = ScratchpadWorkspace::new(
            max_running,
            model.config.hidden_size,
            model.config.intermediate_size,
            model.config.vocab_size,
        );

        Self {
            model,
            block_pool: BlockPool::new(total_blocks),
            prefix_cache: PrefixCache::new(block_size),
            sequences: Vec::new(),
            max_running,
            block_size,
            token_budget,
            current_step: 0,
            next_request_id: 1,
            kv_storage,
            workspace,
            prefix_matched_tokens: 0,
            total_prompt_tokens_admitted: 0,
            prefix_cache_hits: 0,
            prefix_cache_lookups: 0,
        }
    }
    pub fn enable_ane_hybrid(&mut self) {
        self.model.enable_ane_hybrid();
    }

    pub fn set_hybrid_pipeline(&mut self, pipeline: crate::ane::HybridExecutionPipeline) {
        self.model.set_hybrid_pipeline(pipeline);
    }

    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> u64 {
        self.add_request_with_stops(prompt_tokens, max_tokens, Vec::new())
    }

    /// Same as [`Engine::add_request`], but generation also stops on any id in `stop_token_ids`.
    pub fn add_request_with_stops(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
    ) -> u64 {
        self.add_request_with_sampling(
            prompt_tokens,
            max_tokens,
            stop_token_ids,
            SamplingParams::default(),
        )
    }

    /// Full constructor: stop ids plus the per-request sampling knobs the engine's sample
    /// sites read (#56, CLI/server user knobs).
    pub fn add_request_with_sampling(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        stop_token_ids: Vec<u32>,
        sampling: SamplingParams,
    ) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        let mut seq = Sequence::new(id, prompt_tokens, max_tokens);
        seq.stop_token_ids = stop_token_ids;
        seq.sampling = sampling;
        self.sequences.push(seq);
        id
    }

    pub fn prefix_cache_hit_rate(&self) -> f64 {
        if self.total_prompt_tokens_admitted == 0 {
            0.0
        } else {
            (self.prefix_matched_tokens as f64 / self.total_prompt_tokens_admitted as f64).min(1.0)
        }
    }

    pub fn reset_prefix_cache_stats(&mut self) {
        self.prefix_matched_tokens = 0;
        self.total_prompt_tokens_admitted = 0;
        self.prefix_cache_hits = 0;
        self.prefix_cache_lookups = 0;
    }

    pub fn waiting_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Waiting)
            .count()
    }

    pub fn running_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running)
            .count()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.block_pool.free_blocks_count()
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.sequences.iter().any(|s| !s.is_finished())
    }

    pub fn step(&mut self) -> Result<Vec<StepOutput>, EngineError> {
        self.current_step += 1;
        let mut outputs = Vec::new();
        let mut finished_indices = Vec::new();
        let mut remaining_budget = self.token_budget;

        // 1. Priority: Active decode requests execute without stalling
        for idx in 0..self.sequences.len() {
            let seq = &self.sequences[idx];
            if seq.status != SequenceStatus::Running
                || seq.is_finished()
                || seq.prefilled_tokens < seq.prompt_tokens.len()
            {
                continue;
            }
            if remaining_budget == 0 {
                break;
            }

            // #56: the block need is settled before the forward runs. If the pool is
            // short, blocks are taken from the lowest-priority (newest) sequence, which
            // then recomputes; if there is nothing left to take, this sequence waits a
            // step instead of decoding past the end of its block table.
            if self.decode_needs_block(idx) && !self.ensure_blocks(idx, 1) {
                continue;
            }

            let seq = &mut self.sequences[idx];
            let last_token = *seq.generated_tokens.last().unwrap();
            let pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;
            self.workspace.clear();
            let logits = self.model.forward_decode_step_with_workspace(
                last_token,
                pos,
                &seq.block_table,
                &mut self.kv_storage,
                Some(&mut self.workspace),
            )?;

            let next_token = sample_token(
                &logits,
                seq.sampling.temperature,
                seq.sampling.top_p,
                seq.sampling.seed,
            )?;
            seq.generated_tokens.push(next_token);
            seq.block_table.advance_tokens(1);

            // #56: EOS (any configured stop id) retires the sequence exactly like
            // max_tokens, so its blocks return to the pool in this same step.
            let is_finished = seq.is_done_after(next_token);
            if is_finished {
                seq.status = SequenceStatus::Finished;
                finished_indices.push(idx);
            }

            outputs.push(StepOutput {
                request_id: seq.id,
                token_id: next_token,
                is_finished,
            });

            remaining_budget = remaining_budget.saturating_sub(1);
        }

        // 2. Admission stage: Check waiting requests & prefix cache lookup before first chunk
        let running_now = self.running_count();
        let available_slots = self.max_running.saturating_sub(running_now);
        let mut admitted_count = 0;

        for seq in self.sequences.iter_mut() {
            if seq.status == SequenceStatus::Waiting
                && admitted_count < available_slots
                && (self.block_pool.free_blocks_count() > 0 || seq.prefilled_tokens > 0)
            {
                // Prefix cache lookup runs before the first chunk is scheduled
                if seq.prefilled_tokens == 0 && seq.generated_tokens.is_empty() {
                    self.prefix_cache_lookups += 1;
                    self.total_prompt_tokens_admitted += seq.prompt_tokens.len();
                    let (matched_tokens, matched_blocks) =
                        self.prefix_cache.find_longest_prefix(&seq.prompt_tokens);
                    if matched_tokens > 0 {
                        let mut all_claimed = true;
                        let mut claimed = Vec::new();
                        for &blk in &matched_blocks {
                            if self.block_pool.claim_block(blk) {
                                claimed.push(blk);
                            } else {
                                all_claimed = false;
                                break;
                            }
                        }
                        if all_claimed {
                            for &blk in &matched_blocks {
                                seq.block_table.push_block(blk);
                            }
                            let effective_matched = if matched_tokens == seq.prompt_tokens.len() {
                                matched_tokens.saturating_sub(1)
                            } else {
                                matched_tokens
                            };
                            seq.block_table.set_num_tokens(effective_matched);
                            seq.prefilled_tokens = effective_matched;
                            self.prefix_cache_hits += 1;
                            self.prefix_matched_tokens += matched_tokens;
                        } else {
                            for blk in claimed {
                                self.block_pool.free(blk);
                            }
                        }
                    }
                }

                seq.status = SequenceStatus::Running;
                admitted_count += 1;
            }
        }

        // 3. Prefill chunks stage: Process partial prefill chunks with remaining budget
        for idx in 0..self.sequences.len() {
            let seq = &self.sequences[idx];
            if seq.status != SequenceStatus::Running
                || seq.is_finished()
                || seq.prefilled_tokens >= seq.prompt_tokens.len()
            {
                continue;
            }
            if remaining_budget == 0 {
                continue;
            }

            let needed = seq.prompt_tokens.len() - seq.prefilled_tokens;
            let chunk_len = needed.min(remaining_budget);
            if chunk_len == 0 {
                continue;
            }

            let start_pos = seq.prefilled_tokens;
            let end_pos = start_pos + chunk_len;
            // The chunk that completes the prompt also samples its first token, so
            // reserve a slot for that token before any forward runs (#56).
            let needed_blocks = (end_pos + usize::from(end_pos == seq.prompt_tokens.len()))
                .div_ceil(self.block_size);
            let current_blocks = seq.block_table.blocks().len();
            if needed_blocks > current_blocks
                && !self.ensure_blocks(idx, needed_blocks - current_blocks)
            {
                continue;
            }

            let seq = &self.sequences[idx];
            let chunk_tokens = &seq.prompt_tokens[start_pos..end_pos];
            let logits = self.model.forward_prefill(
                chunk_tokens,
                start_pos,
                &seq.block_table,
                &mut self.kv_storage,
            )?;

            let seq = &mut self.sequences[idx];
            seq.prefilled_tokens = end_pos;
            seq.block_table.set_num_tokens(end_pos);
            remaining_budget = remaining_budget.saturating_sub(chunk_len);

            // If prefill completed, sample the first generated token
            if seq.prefilled_tokens == seq.prompt_tokens.len() {
                let next_token = sample_token(
                    &logits,
                    seq.sampling.temperature,
                    seq.sampling.top_p,
                    seq.sampling.seed,
                )?;
                seq.generated_tokens.push(next_token);
                seq.block_table.advance_tokens(1);

                // #56: EOS retires the sequence in this same step.
                let is_finished = seq.is_done_after(next_token);
                if is_finished {
                    seq.status = SequenceStatus::Finished;
                    finished_indices.push(idx);
                }

                outputs.push(StepOutput {
                    request_id: seq.id,
                    token_id: next_token,
                    is_finished,
                });

                // Insert prompt prefix into prefix cache
                self.prefix_cache
                    .insert_prefix(&seq.prompt_tokens, seq.block_table.blocks());
            }
        }

        // 4. Retirement stage: Free KV blocks for finished sequences
        for idx in finished_indices {
            let seq = &mut self.sequences[idx];
            for &blk in seq.block_table.blocks() {
                self.block_pool.free(blk);
            }
        }

        Ok(outputs)
    }

    /// #56: drops finished requests from the active vector so a long-lived engine
    /// (server, CLI) does not rescan unbounded history every step. Returns how many
    /// were dropped. Finished sequences stay visible until the caller (chat/CLI/server)
    /// calls this; their KV blocks are already back in the pool by then.
    pub fn drain_finished(&mut self) -> usize {
        let before = self.sequences.len();
        self.sequences.retain(|s| !s.is_finished());
        before - self.sequences.len()
    }

    /// Would the next decoded token spill past the sequence's allocated blocks?
    fn decode_needs_block(&self, idx: usize) -> bool {
        let table = &self.sequences[idx].block_table;
        table.num_tokens() + 1 > table.blocks().len() * self.block_size
    }

    /// #56: grows sequence `idx`'s block table by `want` blocks, taking them from the
    /// lowest-priority (newest) running sequence when the pool is short — that victim is
    /// preempted and recomputes later. `false` = there was nothing left to take, so the
    /// caller must defer the sequence instead of running it outside its block table.
    fn ensure_blocks(&mut self, idx: usize, want: usize) -> bool {
        while self.block_pool.free_blocks_count() < want {
            if !self.preempt_lowest_priority_after(idx) {
                return false;
            }
        }
        let seq = &mut self.sequences[idx];
        for _ in 0..want {
            let Some(blk) = self.block_pool.allocate() else {
                // Unreachable: the loop above just secured `want` free blocks.
                return false;
            };
            // #44: block is about to be overwritten — drop stale prefix entries.
            self.prefix_cache.invalidate_block(blk);
            seq.block_table.push_block(blk);
        }
        true
    }

    /// Picks the newest running sequence after `idx` (lowest priority) and preempts it.
    fn preempt_lowest_priority_after(&mut self, idx: usize) -> bool {
        let victim = (idx + 1..self.sequences.len()).rev().find(|&j| {
            self.sequences[j].status == SequenceStatus::Running && !self.sequences[j].is_finished()
        });
        match victim {
            Some(v) => {
                self.preempt(v);
                true
            }
            None => false,
        }
    }

    /// Recompute preemption (#56): release the sequence's KV blocks and fold the tokens it
    /// already emitted back into its prompt, so it re-prefills from a clean block table when
    /// rescheduled. Already-reported tokens are not replayed — `max_tokens` shrinks by the
    /// folded count, so the caller still observes exactly `max_tokens` outputs.
    fn preempt(&mut self, idx: usize) {
        let seq = &mut self.sequences[idx];
        for &blk in seq.block_table.blocks() {
            self.block_pool.free(blk);
        }
        let emitted = seq.generated_tokens.len();
        seq.prompt_tokens.append(&mut seq.generated_tokens);
        seq.max_tokens = seq.max_tokens.saturating_sub(emitted);
        seq.block_table = BlockTable::new();
        seq.prefilled_tokens = 0;
        seq.status = SequenceStatus::Waiting;
    }

    /// Greedy (temperature 0) speculation: candidates are verified by argmax match.
    pub fn speculative_step(
        &mut self,
        request_id: u64,
        draft_candidates: &[u32],
    ) -> Result<crate::speculative::VerificationResult, EngineError> {
        self.speculative_step_sampled(request_id, draft_candidates, &[], SamplingParams::default())
    }

    /// Full speculation step. `draft_probs[i]` is the distribution the draft drew
    /// `draft_candidates[i]` from — required when `sampling.temperature > 0`, where the
    /// candidates are verified by rejection sampling instead of argmax match (#55).
    pub fn speculative_step_sampled(
        &mut self,
        request_id: u64,
        draft_candidates: &[u32],
        draft_probs: &[Vec<f32>],
        sampling: SamplingParams,
    ) -> Result<crate::speculative::VerificationResult, EngineError> {
        let seq_idx = self
            .sequences
            .iter()
            .position(|s| s.id == request_id)
            .ok_or_else(|| EngineError::Model(format!("Request {} not found", request_id)))?;

        let seq = &self.sequences[seq_idx];
        if seq.status != SequenceStatus::Running || seq.is_finished() {
            return Err(EngineError::Model(format!(
                "Sequence {} is not running or already finished",
                request_id
            )));
        }
        if seq.prefilled_tokens < seq.prompt_tokens.len() || seq.generated_tokens.is_empty() {
            return Err(EngineError::Model(format!(
                "Sequence {} has not completed prefill yet",
                request_id
            )));
        }

        let last_token = *seq.generated_tokens.last().unwrap();
        let start_pos = seq.prompt_tokens.len() + seq.generated_tokens.len() - 1;

        let seq_mut = &mut self.sequences[seq_idx];
        let verify_result = crate::speculative::SpeculativeVerifier::verify(
            &self.model,
            last_token,
            draft_candidates,
            draft_probs,
            sampling,
            start_pos,
            &mut seq_mut.block_table,
            &mut self.block_pool,
            &mut self.prefix_cache,
            &mut self.kv_storage,
            self.block_size,
        )?;

        let seq_mut = &mut self.sequences[seq_idx];
        let mut committed = Vec::new();
        for &tok in &verify_result.committed_tokens {
            if seq_mut.generated_tokens.len() < seq_mut.max_tokens {
                seq_mut.generated_tokens.push(tok);
                committed.push(tok);
                if seq_mut.stop_token_ids.contains(&tok) {
                    break; // #56: EOS — never commit draft tokens past the stop token
                }
            }
        }

        let total_seq_tokens = seq_mut.prompt_tokens.len() + seq_mut.generated_tokens.len();
        seq_mut.block_table.rollback_blocks(
            total_seq_tokens,
            self.block_size,
            &mut self.block_pool,
        );

        // #56/F1: same termination rule as the decode/prefill paths.
        let done = seq_mut
            .generated_tokens
            .last()
            .is_some_and(|&tok| seq_mut.is_done_after(tok));
        if done {
            seq_mut.status = SequenceStatus::Finished;
            // #56/F3: a speculative finish never reaches stage-4 retirement, so the
            // blocks would otherwise stay out of the pool until process end.
            for &blk in seq_mut.block_table.blocks() {
                self.block_pool.free(blk);
            }
        }

        let mut final_result = verify_result;
        final_result.committed_tokens = committed;
        Ok(final_result)
    }
}
