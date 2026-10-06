pub struct BlockPool {
    free_blocks: Vec<usize>,
    ref_counts: Vec<usize>,
    total_blocks: usize,
}

impl BlockPool {
    pub fn new(total_blocks: usize) -> Self {
        let free_blocks = (0..total_blocks).rev().collect();
        let ref_counts = vec![0; total_blocks];
        Self {
            free_blocks,
            ref_counts,
            total_blocks,
        }
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn free_blocks_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn allocate(&mut self) -> Option<usize> {
        let block = self.free_blocks.pop()?;
        self.ref_counts[block] = 1;
        Some(block)
    }

    pub fn increment_ref(&mut self, block: usize) {
        if block < self.total_blocks {
            self.ref_counts[block] += 1;
        }
    }

    pub fn free(&mut self, block: usize) {
        if block < self.total_blocks && self.ref_counts[block] > 0 {
            self.ref_counts[block] -= 1;
            if self.ref_counts[block] == 0 {
                self.free_blocks.push(block);
            }
        }
    }
}
