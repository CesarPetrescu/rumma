use rumma_core::{QuantizationConfig, QuantizedLinear};

#[test]
fn quantize_roundtrip_preserves_structure() {
    let cfg = QuantizationConfig::default();
    let rows = 2;
    let cols = 16;
    let mut weights = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let value = ((row * cols + col) as f32 * 0.03125) - 0.25;
            weights.push(value);
        }
    }
    let quant = QuantizedLinear::from_dense(rows, cols, &weights, &cfg).unwrap();
    let mut buffer = vec![0.0f32; cols];
    quant.dequantize_row(1, &mut buffer);
    let dense_slice = &weights[cols..];
    let mut max_error = 0.0f32;
    for (a, b) in buffer.iter().zip(dense_slice.iter()) {
        max_error = max_error.max((a - b).abs());
    }
    assert!(max_error < 0.05, "max error {max_error}");

    let input = vec![0.5f32; cols];
    let dense_dot: f32 = dense_slice.iter().zip(&input).map(|(w, x)| w * x).sum();
    let quant_dot = quant.dot_row(1, &input);
    assert!((dense_dot - quant_dot).abs() < 0.05);
}
