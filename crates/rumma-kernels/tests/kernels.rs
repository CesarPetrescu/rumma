use rumma_core::{QuantizationConfig, QuantizedLinear};
use rumma_kernels::{gemm_prefill, gemv_decode};

fn dense_gemm(weights: &[f32], rows: usize, cols: usize, input: &[f32], batch: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * batch];
    for b in 0..batch {
        for row in 0..rows {
            let mut acc = 0.0;
            for col in 0..cols {
                acc += weights[row * cols + col] * input[b * cols + col];
            }
            output[b * rows + row] = acc;
        }
    }
    output
}

#[test]
fn gemm_prefill_matches_dense() {
    let rows = 4;
    let cols = 8;
    let cfg = QuantizationConfig::default();
    let mut weights = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            weights.push(((row + col) as f32 * 0.1) - 0.3);
        }
    }
    let layer = QuantizedLinear::from_dense(rows, cols, &weights, &cfg).unwrap();
    let batch = 3;
    let mut input = vec![0.0f32; cols * batch];
    for (idx, value) in input.iter_mut().enumerate() {
        *value = ((idx % cols) as f32 * 0.05) - 0.2;
    }
    let baseline = dense_gemm(&weights, rows, cols, &input, batch);
    let quantized = gemm_prefill(&layer, &input, batch).unwrap();
    for (a, b) in baseline.iter().zip(quantized.iter()) {
        assert!((a - b).abs() < 0.1, "{} vs {}", a, b);
    }
}

#[test]
fn gemv_decode_matches_dense() {
    let rows = 4;
    let cols = 8;
    let cfg = QuantizationConfig::default();
    let mut weights = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            weights.push(((row * cols + col) as f32 * 0.02) - 0.2);
        }
    }
    let layer = QuantizedLinear::from_dense(rows, cols, &weights, &cfg).unwrap();
    let mut input = vec![0.0f32; cols];
    for (idx, value) in input.iter_mut().enumerate() {
        *value = (idx as f32 * 0.03) - 0.1;
    }
    let baseline = dense_gemm(&weights, rows, cols, &input, 1);
    let quantized = gemv_decode(&layer, &input).unwrap();
    for (a, b) in baseline.iter().zip(quantized.iter()) {
        assert!((a - b).abs() < 0.1);
    }
}
