use std::collections::HashMap;

/// Default bound on cached prefix entries; past this the least-recently-used
/// entry is evicted. Keeps prefix memory O(1) in sequence count.
pub const DEFAULT_MAX_ENTRIES: usize = 64;

pub struct PrefixCache {
    block_size: usize,
    max_entries: usize,
    cached_prefixes: HashMap<Vec<u32>, Vec<usize>>,
    /// Reverse index: physical block -> token-prefix keys referencing it.
    /// Lets block reuse drop exactly the entries that went stale.
    block_to_keys: HashMap<usize, Vec<Vec<u32>>>,
    last_used: HashMap<Vec<u32>, u64>,
    tick: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self::with_capacity(block_size, DEFAULT_MAX_ENTRIES)
    }

    pub fn with_capacity(block_size: usize, max_entries: usize) -> Self {
        Self {
            block_size,
            max_entries: max_entries.max(1),
            cached_prefixes: HashMap::new(),
            block_to_keys: HashMap::new(),
            last_used: HashMap::new(),
            tick: 0,
        }
    }

    /// Stores prefix tokens in increments of block_size with their physical block IDs
    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) {
        let num_complete_blocks = tokens.len() / self.block_size;
        for b in 1..=num_complete_blocks {
            let token_prefix = tokens[0..(b * self.block_size)].to_vec();
            let block_prefix = blocks[0..b].to_vec();
            self.tick += 1;
            let tick = self.tick;
            if let Some(old) = self
                .cached_prefixes
                .insert(token_prefix.clone(), block_prefix.clone())
            {
                for blk in old {
                    if !block_prefix.contains(&blk) {
                        self.detach(&token_prefix, blk);
                    }
                }
            }
            for &blk in &block_prefix {
                let keys = self.block_to_keys.entry(blk).or_default();
                if !keys.contains(&token_prefix) {
                    keys.push(token_prefix.clone());
                }
            }
            self.last_used.insert(token_prefix, tick);
        }
        while self.cached_prefixes.len() > self.max_entries {
            let oldest = self
                .last_used
                .iter()
                .min_by_key(|(_, t)| **t)
                .map(|(k, _)| k.clone());
            match oldest {
                Some(key) => self.remove_entry(&key),
                None => break,
            }
        }
    }

    /// Drops every entry referencing `block`. Call when the block is handed to
    /// a different sequence: the cached KV no longer matches recorded tokens.
    /// Returns the number of entries dropped.
    pub fn invalidate_block(&mut self, block: usize) -> usize {
        let keys = self.block_to_keys.remove(&block).unwrap_or_default();
        let n = keys.len();
        for key in &keys {
            self.remove_entry(key);
        }
        n
    }

    /// Finds the longest cached prefix matching the given tokens on block boundaries
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<usize>) {
        let num_complete_blocks = tokens.len() / self.block_size;
        for b in (1..=num_complete_blocks).rev() {
            let slice = &tokens[0..(b * self.block_size)];
            if let Some(blocks) = self.cached_prefixes.get(slice).cloned() {
                self.tick += 1;
                self.last_used.insert(slice.to_vec(), self.tick);
                return (b * self.block_size, blocks);
            }
        }
        (0, Vec::new())
    }

    fn remove_entry(&mut self, key: &[u32]) {
        if let Some(blocks) = self.cached_prefixes.remove(key) {
            for blk in blocks {
                self.detach(key, blk);
            }
        }
        self.last_used.remove(key);
    }

    fn detach(&mut self, key: &[u32], block: usize) {
        if let Some(keys) = self.block_to_keys.get_mut(&block) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.block_to_keys.remove(&block);
            }
        }
    }
}
