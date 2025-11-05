use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::quant::{QuantizationConfig, QuantizedLinear};

/// Simple sequential container for quantized linear layers.
#[derive(Clone, Debug)]
pub struct Model {
    layers: Arc<Vec<QuantizedLinear>>,
    hidden_size: usize,
}

impl Model {
    pub fn new(layers: Vec<QuantizedLinear>) -> Result<Self> {
        if layers.is_empty() {
            return Err(anyhow!("model must contain at least one layer"));
        }
        let hidden = layers[0].cols();
        if !layers
            .iter()
            .all(|layer| layer.cols() == hidden && layer.rows() == hidden)
        {
            return Err(anyhow!(
                "only square layers with uniform hidden size are supported in this demo"
            ));
        }
        Ok(Self {
            layers: Arc::new(layers),
            hidden_size: hidden,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn layers(&self) -> &[QuantizedLinear] {
        self.layers.as_ref()
    }
}

/// Helper that constructs toy models for smoke testing and CLI experimentation.
#[derive(Clone, Debug)]
pub struct ModelBuilder {
    hidden_size: usize,
    depth: usize,
    quant_cfg: QuantizationConfig,
}

impl ModelBuilder {
    pub fn new(hidden_size: usize, depth: usize, quant_cfg: QuantizationConfig) -> Self {
        Self {
            hidden_size,
            depth,
            quant_cfg,
        }
    }

    pub fn build_from_weights(&self, weights: &[Vec<f32>]) -> Result<Model> {
        if weights.len() != self.depth {
            return Err(anyhow!(
                "expected {} layers, got {}",
                self.depth,
                weights.len()
            ));
        }
        let layers = weights
            .iter()
            .enumerate()
            .map(|(idx, data)| {
                QuantizedLinear::from_dense(
                    self.hidden_size,
                    self.hidden_size,
                    data,
                    &self.quant_cfg,
                )
                .map_err(|err| anyhow!("layer {}: {err}", idx))
            })
            .collect::<Result<Vec<_>>>()?;
        Model::new(layers)
    }

    pub fn build_random(&self, seed: u64) -> Result<Model> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Vec::with_capacity(self.depth);
        for _ in 0..self.depth {
            let mut layer = Vec::with_capacity(self.hidden_size * self.hidden_size);
            for _ in 0..self.hidden_size * self.hidden_size {
                // keep values small to emulate normalized transformer weights
                layer.push(rng.gen_range(-0.5f32..0.5f32));
            }
            weights.push(layer);
        }
        self.build_from_weights(&weights)
    }
}
