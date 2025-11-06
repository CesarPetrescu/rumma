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

        // Note: We do not validate that layers chain properly here, as real AWQ models
        // may contain layers from different parts of a transformer (attention, FFN, etc.)
        // that are not meant to be chained sequentially. Dimension mismatches will be
        // caught at runtime during GEMM operations if the layers are used incorrectly.

        // For compatibility with the Engine, we use the output dimension of the last layer
        // as the hidden_size (this is what gets stored in the cache)
        let hidden = layers.last().unwrap().rows();

        Ok(Self {
            layers: Arc::new(layers),
            hidden_size: hidden,
        })
    }

    /// Creates a Model with an explicitly specified hidden_size.
    /// This is useful for AWQ models where the hidden_size cannot be reliably
    /// inferred from the layer dimensions alone.
    pub fn with_hidden_size(layers: Vec<QuantizedLinear>, hidden_size: usize) -> Result<Self> {
        if layers.is_empty() {
            return Err(anyhow!("model must contain at least one layer"));
        }

        Ok(Self {
            layers: Arc::new(layers),
            hidden_size,
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
        self.build_from_quantized_layers(layers)
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

    pub fn build_from_quantized_layers(&self, layers: Vec<QuantizedLinear>) -> Result<Model> {
        if layers.len() != self.depth {
            return Err(anyhow!(
                "expected {} layers, got {}",
                self.depth,
                layers.len()
            ));
        }

        // Validate that the model's output dimension matches the expected hidden_size
        // This ensures compatibility with the test/benchmark setup
        if let Some(last_layer) = layers.last() {
            if last_layer.rows() != self.hidden_size {
                return Err(anyhow!(
                    "model output dimension {} does not match expected hidden_size {}",
                    last_layer.rows(),
                    self.hidden_size
                ));
            }
        }

        Model::new(layers)
    }
}
