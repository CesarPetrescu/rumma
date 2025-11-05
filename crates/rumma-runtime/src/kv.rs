use std::collections::HashMap;

use parking_lot::RwLock;

#[derive(Debug)]
pub struct PagedKvCache {
    hidden_size: usize,
    entries: RwLock<HashMap<u64, Vec<f32>>>,
}

#[derive(Debug)]
pub struct KeyValueHandle {
    pub sequence_id: u64,
}

impl PagedKvCache {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn insert(&self, id: u64, values: Vec<f32>) {
        assert_eq!(values.len(), self.hidden_size);
        self.entries.write().insert(id, values);
    }

    pub fn update(&self, id: u64, values: Vec<f32>) {
        assert_eq!(values.len(), self.hidden_size);
        self.entries.write().insert(id, values);
    }

    pub fn get(&self, id: u64) -> Option<Vec<f32>> {
        self.entries.read().get(&id).cloned()
    }

    pub fn remove(&self, id: u64) {
        self.entries.write().remove(&id);
    }
}

impl KeyValueHandle {
    pub fn new(sequence_id: u64) -> Self {
        Self { sequence_id }
    }
}
