pub mod block_pool;
pub mod block_table;
pub mod kind;
pub mod prefix_cache;
pub mod quantized_storage;
pub mod storage;

pub use block_pool::BlockPool;
pub use block_table::BlockTable;
pub use kind::{KvCacheType, KvGeometry, KvSlotPlan, KvStorageKind, PaddedBatch, slot_for};
pub use prefix_cache::PrefixCache;
pub use quantized_storage::{QuantMode, QuantType, QuantizedKvBlock, QuantizedKvStorage};
pub use storage::KvStorage;

use crate::error::EngineError;

/// Allocates a physical block and erases what the previous request left in it (#106).
///
/// `BlockPool` only tracks ref counts — it knows nothing about tensors — so the erase
/// happens here, where the pool and the storage it backs are both in hand. Every caller
/// that pairs a block with a KV pool must allocate through this: both pools keep a
/// block's rows until overwritten, so a reused block's tail past the new request's
/// `ctx_len` is stale KV that attention can still read.
///
/// `Ok(None)` = the pool is empty; each caller keeps its own out-of-memory wording.
pub fn allocate_zeroed(
    pool: &mut BlockPool,
    storage: &mut KvStorageKind,
) -> Result<Option<usize>, EngineError> {
    let Some(block) = pool.allocate() else {
        return Ok(None);
    };
    storage.reset_block(block)?;
    Ok(Some(block))
}
