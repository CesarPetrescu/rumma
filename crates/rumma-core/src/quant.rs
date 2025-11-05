use std::fmt;

use anyhow::{anyhow, Result};
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub symmetric: bool,
    pub nibble_map: [u8; 8],
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: true,
            nibble_map: [0, 1, 2, 3, 4, 5, 6, 7],
        }
    }
}

impl QuantizationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.group_size == 0 {
            return Err(anyhow!("group size must be > 0"));
        }
        let mut seen = [false; 8];
        for &slot in &self.nibble_map {
            if slot as usize >= 8 {
                return Err(anyhow!("nibble indices must be < 8"));
            }
            if seen[slot as usize] {
                return Err(anyhow!("nibble mapping must be a permutation"));
            }
            seen[slot as usize] = true;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedLinear {
    rows: usize,
    cols: usize,
    group_size: usize,
    words_per_row: usize,
    groups_per_row: usize,
    nibble_map: [u8; 8],
    nibble_unmap: [u8; 8],
    weights: Vec<u32>,
    scales: Vec<f32>,
    zero_points: Option<Vec<u8>>,
}

impl QuantizedLinear {
    pub fn from_dense(
        rows: usize,
        cols: usize,
        dense: &[f32],
        cfg: &QuantizationConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        if dense.len() != rows * cols {
            return Err(anyhow!(
                "dense weight tensor has {} elements, expected {}",
                dense.len(),
                rows * cols
            ));
        }
        let words_per_row = (cols + 7) / 8;
        let groups_per_row = (cols + cfg.group_size - 1) / cfg.group_size;
        let mut weights = vec![0u32; rows * words_per_row];
        let mut scales = vec![0f32; rows * groups_per_row];
        let mut zeros = if cfg.symmetric {
            None
        } else {
            Some(vec![8u8; rows * groups_per_row])
        };
        let mut nibble_unmap = [0u8; 8];
        for (logical, &slot) in cfg.nibble_map.iter().enumerate() {
            nibble_unmap[slot as usize] = logical as u8;
        }
        for row in 0..rows {
            let row_offset = row * cols;
            for group in 0..groups_per_row {
                let group_start = group * cfg.group_size;
                let group_end = usize::min(group_start + cfg.group_size, cols);
                let slice = &dense[row_offset + group_start..row_offset + group_end];
                let max = slice.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
                let scale = if max == 0.0 { 1e-8 } else { max / 7.0 };
                scales[row * groups_per_row + group] = scale;
                if let Some(ref mut zp) = zeros {
                    zp[row * groups_per_row + group] = 8;
                }
                for (local_idx, chunk) in slice.chunks(8).enumerate() {
                    let mut packed = 0u32;
                    for (logical, &value) in chunk.iter().enumerate() {
                        let slot = cfg.nibble_map[logical] as usize;
                        let scaled = if scale == 0.0 {
                            0.0
                        } else {
                            (value / scale).round().clamp(-8.0, 7.0)
                        };
                        let q = (scaled as i32 + 8) as u32;
                        packed |= (q & 0xF) << (slot * 4);
                    }
                    let word_index = row * words_per_row + group_start / 8 + local_idx;
                    weights[word_index] = packed;
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            group_size: cfg.group_size,
            words_per_row,
            groups_per_row,
            nibble_map: cfg.nibble_map,
            nibble_unmap,
            weights,
            scales,
            zero_points: zeros,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn weights(&self) -> &[u32] {
        &self.weights
    }

    pub fn zero_points(&self) -> Option<&[u8]> {
        self.zero_points.as_deref()
    }

    pub fn quantized_row(&self, row: usize) -> &[u32] {
        let start = row * self.words_per_row;
        &self.weights[start..start + self.words_per_row]
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        assert!(output.len() >= self.cols);
        output.fill(0.0);
        for group in 0..self.groups_per_row {
            let group_start = group * self.group_size;
            let group_len = usize::min(self.group_size, self.cols - group_start);
            let scale = self.scales[row * self.groups_per_row + group];
            let zero = self
                .zero_points
                .as_ref()
                .map(|zp| zp[row * self.groups_per_row + group] as i32)
                .unwrap_or(8);
            let base_word = row * self.words_per_row + group_start / 8;
            let words_in_group = (group_len + 7) / 8;
            for word_offset in 0..words_in_group {
                let word = self.weights[base_word + word_offset];
                for slot in 0..8 {
                    let logical = self.nibble_unmap[slot] as usize + word_offset * 8;
                    if logical >= group_len {
                        continue;
                    }
                    let q = ((word >> (slot * 4)) & 0xF) as i32;
                    let value = (q - zero) as f32 * scale;
                    let col = group_start + logical;
                    if col < self.cols {
                        output[col] = value;
                    }
                }
            }
        }
    }

    pub fn dot_row(&self, row: usize, input: &[f32]) -> f32 {
        assert_eq!(input.len(), self.cols);
        let mut acc = 0.0f32;
        for group in 0..self.groups_per_row {
            let group_start = group * self.group_size;
            let group_len = usize::min(self.group_size, self.cols - group_start);
            let scale = self.scales[row * self.groups_per_row + group];
            let zero = self
                .zero_points
                .as_ref()
                .map(|zp| zp[row * self.groups_per_row + group] as i32)
                .unwrap_or(8);
            let base_word = row * self.words_per_row + group_start / 8;
            let words_in_group = (group_len + 7) / 8;
            for word_offset in 0..words_in_group {
                let word = self.weights[base_word + word_offset];
                for slot in 0..8 {
                    let logical = self.nibble_unmap[slot] as usize + word_offset * 8;
                    if logical >= group_len {
                        continue;
                    }
                    let q = ((word >> (slot * 4)) & 0xF) as i32;
                    let val = (q - zero) as f32 * scale;
                    let col = group_start + logical;
                    acc += val * input[col];
                }
            }
        }
        acc
    }
}

impl fmt::Display for QuantizedLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizedLinear[rows={}, cols={}, group={}, words_per_row={}, nibble_map={:?}]",
            self.rows, self.cols, self.group_size, self.words_per_row, self.nibble_map
        )
    }
}

#[derive(Error, Debug)]
pub enum QuantizationError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}
