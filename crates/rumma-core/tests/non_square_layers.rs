use rumma_core::{Model, QuantizationConfig, QuantizedLinear};

#[test]
fn model_accepts_non_square_layers_that_chain() {
    let cfg = QuantizationConfig::default();

    // Create a chain of non-square layers: 128 -> 256 -> 128
    // This mimics an MLP: input -> expand -> contract

    // Layer 1: 256 x 128 (input: 128, output: 256)
    let layer1_weights = vec![0.01f32; 256 * 128];
    let layer1 = QuantizedLinear::from_dense(256, 128, &layer1_weights, &cfg).unwrap();

    // Layer 2: 128 x 256 (input: 256, output: 128)
    let layer2_weights = vec![0.01f32; 128 * 256];
    let layer2 = QuantizedLinear::from_dense(128, 256, &layer2_weights, &cfg).unwrap();

    // Verify dimensions
    assert_eq!(layer1.cols(), 128);
    assert_eq!(layer1.rows(), 256);
    assert_eq!(layer2.cols(), 256);
    assert_eq!(layer2.rows(), 128);

    // Create model - should succeed since layers chain properly
    let model = Model::new(vec![layer1, layer2]).unwrap();

    // Verify hidden_size is the output of the last layer
    assert_eq!(model.hidden_size(), 128);
}

#[test]
fn model_rejects_non_chaining_layers() {
    let cfg = QuantizationConfig::default();

    // Layer 1: 256 x 128 (input: 128, output: 256)
    let layer1_weights = vec![0.01f32; 256 * 128];
    let layer1 = QuantizedLinear::from_dense(256, 128, &layer1_weights, &cfg).unwrap();

    // Layer 2: 128 x 128 (input: 128, output: 128)
    // This doesn't chain! Layer1 output is 256, but Layer2 expects input of 128
    let layer2_weights = vec![0.01f32; 128 * 128];
    let layer2 = QuantizedLinear::from_dense(128, 128, &layer2_weights, &cfg).unwrap();

    // Create model - should fail
    let result = Model::new(vec![layer1, layer2]);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("output dimension"));
    assert!(error_msg.contains("input dimension"));
}

#[test]
fn model_accepts_square_layers() {
    let cfg = QuantizationConfig::default();

    // Create two square layers (backwards compatibility test)
    let layer1_weights = vec![0.01f32; 128 * 128];
    let layer1 = QuantizedLinear::from_dense(128, 128, &layer1_weights, &cfg).unwrap();

    let layer2_weights = vec![0.01f32; 128 * 128];
    let layer2 = QuantizedLinear::from_dense(128, 128, &layer2_weights, &cfg).unwrap();

    let model = Model::new(vec![layer1, layer2]).unwrap();
    assert_eq!(model.hidden_size(), 128);
}
