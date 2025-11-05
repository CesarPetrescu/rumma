use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct BatchHandle {
    sequence_ids: Vec<u64>,
}

impl BatchHandle {
    pub fn sequence_ids(&self) -> &[u64] {
        &self.sequence_ids
    }
}

#[derive(Debug, Default)]
pub struct Scheduler {
    next_sequence: u64,
    active: HashSet<u64>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_batch(&mut self, batch_size: usize) -> BatchHandle {
        let mut ids = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let id = self.next_sequence;
            self.next_sequence += 1;
            self.active.insert(id);
            ids.push(id);
        }
        BatchHandle { sequence_ids: ids }
    }

    pub fn retire_sequence(&mut self, id: u64) {
        self.active.remove(&id);
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}
