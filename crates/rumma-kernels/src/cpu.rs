use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rumma_core::QuantizedLinear;

pub fn gemm_prefill(layer: &QuantizedLinear, input: &[f32], batch: usize) -> Result<Vec<f32>> {
    if input.len() != layer.cols() * batch {
        return Err(anyhow!(
            "prefill input has {} elements, expected {}",
            input.len(),
            layer.cols() * batch
        ));
    }
    let cols = layer.cols();
    let rows = layer.rows();
    let mut output = vec![0f32; rows * batch];
    output
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(b, out)| {
            let vector = &input[b * cols..(b + 1) * cols];
            for row in 0..rows {
                out[row] = layer.dot_row(row, vector);
            }
        });
    Ok(output)
}

pub fn gemv_decode(layer: &QuantizedLinear, input: &[f32]) -> Result<Vec<f32>> {
    if input.len() != layer.cols() {
        return Err(anyhow!(
            "decode input has {} elements, expected {}",
            input.len(),
            layer.cols()
        ));
    }
    let rows = layer.rows();
    let mut output = vec![0f32; rows];
    output.par_iter_mut().enumerate().for_each(|(row, slot)| {
        *slot = layer.dot_row(row, input);
    });
    Ok(output)
}

pub fn relu_inplace(tensor: &mut [f32]) {
    tensor.iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut sum = 0.0f32;
    for value in logits.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum == 0.0 {
        return;
    }
    for value in logits.iter_mut() {
        *value /= sum;
    }
}
