use std::any::Any;

use anyhow::{bail, Result};
use crate::quant::QuantizedLinear;

pub trait Model: Send + Sync + 'static {
    fn layers(&self) -> Vec<QuantizedLinear>;
    fn as_any(&self) -> &dyn Any;
}

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
}

pub struct GenericModel {
    layers: Vec<QuantizedLinear>,
}

impl Model for GenericModel {
    fn layers(&self) -> Vec<QuantizedLinear> {
        self.layers.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Dense {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}

impl Dense {
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, rows: usize, cols: usize) -> Result<Self> {
        if weight.len() != rows * cols {
            bail!("weight size does not match dimensions");
        }
        if let Some(ref bias) = bias {
            if bias.len() != rows {
                bail!("bias size does not match output dimension");
            }
        }
        Ok(Self {
            weight,
            bias,
            rows,
            cols,
        })
    }
}

use crate::quant::QuantizationConfig;
use rand::{rngs::StdRng, Rng, SeedableRng};

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

    pub fn build_random(self, seed: u64) -> Result<GenericModel> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::new();
        for _ in 0..self.depth {
            let mut dense = vec![0.0; self.hidden_size * self.hidden_size];
            for item in dense.iter_mut() {
                *item = rng.gen_range(-1.0..1.0);
            }
            let q = QuantizedLinear::from_dense(
                self.hidden_size,
                self.hidden_size,
                &dense,
                &self.quant_cfg,
            )?;
            layers.push(q);
        }
        GenericModel::with_hidden_size(layers, self.hidden_size)
    }
}

impl GenericModel {
    pub fn with_hidden_size(layers: Vec<QuantizedLinear>, _hidden_size: usize) -> Result<Self> {
        if layers.is_empty() {
            bail!("model must have at least one layer");
        }
        Ok(Self { layers })
    }
}
