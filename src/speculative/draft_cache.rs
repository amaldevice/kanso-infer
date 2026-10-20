use crate::error::EngineError;
use crate::kv_cache::{BlockPool, BlockTable, KvStorageKind, allocate_zeroed};
use crate::model::Model;
use crate::sampler::{greedy_token, sample_from_probs, softmax_probs};
use crate::scheduler::SamplingParams;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// One round of draft proposals: the tokens and, for temperature > 0, the distribution
/// each was drawn from (the verifier's rejection test needs `q`).
pub struct Proposals {
    pub tokens: Vec<u32>,
    pub probs: Vec<Vec<f32>>,
}

/// The draft model's own paged KV cache (#55).
///
/// Before this the draft re-ran `forward_single_prompt` over the whole prefix per draft
/// token — O(ctx) forward work per proposal, which made drafting slower than the target
/// step it was supposed to skip. Now the draft owns a `KvStorage` + block table and
/// advances one `forward_decode_step` per proposal, and the cache is truncated when the
/// target rejects proposals.
pub struct DraftCache {
    storage: KvStorageKind,
    pool: BlockPool,
    block_size: usize,
    slots: HashMap<u64, Slot>,
}

struct Slot {
    block_table: BlockTable,
    /// The tokens whose K/V this cache holds, in order. The last committed token is never
    /// in here: the target produced it, the draft has not computed it yet.
    tokens: Vec<u32>,
}

impl DraftCache {
    pub fn new(model: &Model, total_blocks: usize, block_size: usize) -> Result<Self, EngineError> {
        let storage = KvStorageKind::f32_new(
            model.config.num_hidden_layers,
            total_blocks,
            block_size,
            model.config.num_key_value_heads,
            model.config.head_dim(),
            &model.device,
        )?;
        Ok(Self {
            storage,
            pool: BlockPool::new(total_blocks),
            block_size,
            slots: HashMap::new(),
        })
    }

    /// Free blocks left in the draft pool.
    pub fn free_blocks_count(&self) -> usize {
        self.pool.free_blocks_count()
    }

    /// Request ids that currently hold a draft cache.
    pub fn slot_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.slots.keys().copied()
    }

    /// Drafts up to `gamma` tokens past `prefix` (the sequence's prompt + generated tokens).
    ///
    /// A cache already holding a prefix of `prefix` is reused; each round after the first
    /// therefore feeds exactly one uncached token (the one the target emitted) before
    /// proposing. What the target does with the proposals afterwards decides how far the
    /// cache is kept — see [`DraftCache::rollback`].
    pub fn propose(
        &mut self,
        model: &Model,
        request_id: u64,
        prefix: &[u32],
        gamma: usize,
        sampling: SamplingParams,
    ) -> Result<Proposals, EngineError> {
        let mut proposals = Proposals {
            tokens: Vec::with_capacity(gamma),
            probs: Vec::with_capacity(gamma),
        };
        if gamma == 0 || prefix.is_empty() {
            return Ok(proposals);
        }

        let Self {
            storage,
            pool,
            block_size,
            slots,
        } = self;
        let block_size = *block_size;
        let slot = slots.entry(request_id).or_insert_with(|| Slot {
            block_table: BlockTable::new(),
            tokens: Vec::new(),
        });

        if !prefix.starts_with(&slot.tokens) {
            // Stale cache (request id reuse): rebuild from scratch.
            for &blk in slot.block_table.blocks() {
                pool.free(blk);
            }
            slot.block_table = BlockTable::new();
            slot.tokens.clear();
        }

        let mut logits = if slot.tokens.is_empty() {
            ensure_blocks(
                pool,
                &mut slot.block_table,
                block_size,
                prefix.len(),
                storage,
            )?;
            let logits = model.forward_prefill(prefix, 0, &slot.block_table, storage)?;
            slot.tokens.extend_from_slice(prefix);
            logits
        } else {
            let mut logits = None;
            let pending = prefix[slot.tokens.len()..].to_vec();
            for tok in pending {
                let pos = slot.tokens.len();
                ensure_blocks(pool, &mut slot.block_table, block_size, pos + 1, storage)?;
                logits = Some(model.forward_decode_step(tok, pos, &slot.block_table, storage)?);
                slot.tokens.push(tok);
            }
            logits.ok_or_else(|| {
                EngineError::Model("draft cache is ahead of the sequence prefix".to_string())
            })?
        };

        // One seeded stream per round (ticket #61 F9), so a temperature > 0 request drafts
        // the same proposals on every replay of the same prefix.
        let mut rng = StdRng::seed_from_u64(sampling.seed);
        for _ in 0..gamma {
            let flat = logits.flatten_all()?.to_vec1::<f32>()?;
            let (token, probs) = if sampling.temperature <= 0.0 {
                (greedy_token(&flat), Vec::new())
            } else {
                let p = softmax_probs(&flat, sampling.temperature, sampling.top_p)?;
                (sample_from_probs(&p, &mut rng), p)
            };
            proposals.tokens.push(token);
            if !probs.is_empty() {
                proposals.probs.push(probs);
            }

            let pos = slot.tokens.len();
            ensure_blocks(pool, &mut slot.block_table, block_size, pos + 1, storage)?;
            logits = model.forward_decode_step(token, pos, &slot.block_table, storage)?;
            slot.tokens.push(token);
        }

        Ok(proposals)
    }

    /// Keeps only the first `keep` cached tokens, returning the blocks that held rejected
    /// proposals to the draft pool. The K/V of kept positions stays valid — a rejection
    /// only invalidates positions past the accepted prefix, which the next round rewrites.
    pub fn rollback(&mut self, request_id: u64, keep: usize) {
        let Self {
            pool,
            block_size,
            slots,
            ..
        } = self;
        if let Some(slot) = slots.get_mut(&request_id) {
            slot.tokens.truncate(keep);
            slot.block_table.rollback_blocks(keep, *block_size, pool);
        }
    }

    /// Drops a request's cache and returns its blocks to the draft pool.
    pub fn release(&mut self, request_id: u64) {
        let Self { pool, slots, .. } = self;
        if let Some(slot) = slots.remove(&request_id) {
            for &blk in slot.block_table.blocks() {
                pool.free(blk);
            }
        }
    }
}

fn ensure_blocks(
    pool: &mut BlockPool,
    table: &mut BlockTable,
    block_size: usize,
    up_to: usize,
    storage: &mut KvStorageKind,
) -> Result<(), EngineError> {
    let needed = up_to.div_ceil(block_size);
    while table.blocks().len() < needed {
        let blk = allocate_zeroed(pool, storage)?.ok_or_else(|| {
            EngineError::Model("Draft KV: out of memory: cannot allocate block".to_string())
        })?;
        table.push_block(blk);
    }
    table.set_num_tokens(up_to);
    Ok(())
}
