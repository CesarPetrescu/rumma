use std::sync::Arc;

use anyhow::{anyhow, Result};

use rumma_core::Model;
use rumma_kernels::{gemm_prefill, gemv_decode, relu_inplace, softmax};

use crate::{
    graph::GraphRegistry,
    kv::PagedKvCache,
    scheduler::{BatchHandle, Scheduler},
};

#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

pub struct Engine {
    model: Arc<Model>,
    cache: PagedKvCache,
    scheduler: Scheduler,
    graphs: GraphRegistry,
}

impl Engine {
    pub fn new(model: Arc<Model>) -> Self {
        let hidden = model.hidden_size();
        Self {
            model,
            cache: PagedKvCache::new(hidden),
            scheduler: Scheduler::new(),
            graphs: GraphRegistry::new(),
        }
    }

    pub fn scheduler(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    pub fn capture_decode_graph(&mut self) {
        self.graphs.mark_decode_captured();
    }

    pub fn decode_graph_captured(&self) -> bool {
        self.graphs.decode_captured()
    }

    pub fn prefill_batch(&mut self, handle: &BatchHandle, inputs: &[f32]) -> Result<()> {
        let batch = handle.sequence_ids().len();
        let hidden = self.model.hidden_size();
        if inputs.len() != batch * hidden {
            return Err(anyhow!(
                "prefill expects {} elements, received {}",
                batch * hidden,
                inputs.len()
            ));
        }
        let mut activations = inputs.to_vec();
        for layer in self.model.layers() {
            let mut next = gemm_prefill(layer, &activations, batch)?;
            relu_inplace(&mut next);
            activations = next;
        }
        for (idx, &sequence_id) in handle.sequence_ids().iter().enumerate() {
            let start = idx * hidden;
            let end = start + hidden;
            self.cache
                .insert(sequence_id, activations[start..end].to_vec());
        }
        Ok(())
    }

    pub fn decode_step(&mut self, sequence_id: u64, input: &[f32]) -> Result<DecodeOutput> {
        let hidden = self.model.hidden_size();
        if input.len() != hidden {
            return Err(anyhow!(
                "decode expects {} elements, received {}",
                hidden,
                input.len()
            ));
        }
        let mut state = input.to_vec();
        for layer in self.model.layers() {
            let mut next = gemv_decode(layer, &state)?;
            relu_inplace(&mut next);
            state = next;
        }
        let logits = state.clone();
        let mut probabilities = logits.clone();
        softmax(&mut probabilities);
        self.cache.update(sequence_id, state);
        Ok(DecodeOutput {
            logits,
            probabilities,
        })
    }

    pub fn cached_hidden(&self, sequence_id: u64) -> Option<Vec<f32>> {
        self.cache.get(sequence_id)
    }

    pub fn retire_sequence(&mut self, sequence_id: u64) {
        self.scheduler.retire_sequence(sequence_id);
        self.cache.remove(sequence_id);
    }
}
