use std::sync::Arc;

use rumma_core::{ModelBuilder, QuantizationConfig};
use rumma_runtime::Engine;

#[test]
fn engine_prefill_decode_flow() {
    let cfg = QuantizationConfig::default();
    let builder = ModelBuilder::new(8, 2, cfg);
    let model = Arc::new(builder.build_random(7).unwrap());
    let mut engine = Engine::new(model);

    let batch_handle = engine.scheduler().register_batch(2);
    let mut inputs = vec![0.0f32; 16];
    for (idx, value) in inputs.iter_mut().enumerate() {
        *value = (idx as f32 * 0.02) - 0.3;
    }
    engine.prefill_batch(&batch_handle, &inputs).unwrap();

    for &sequence_id in batch_handle.sequence_ids() {
        assert!(engine.cached_hidden(sequence_id).is_some());
    }

    let seq_id = batch_handle.sequence_ids()[0];
    let cached = engine.cached_hidden(seq_id).unwrap();
    let decode = engine.decode_step(seq_id, &cached).unwrap();
    assert_eq!(decode.logits.len(), cached.len());
    assert_eq!(decode.probabilities.len(), cached.len());

    engine.retire_sequence(seq_id);
    assert!(engine.cached_hidden(seq_id).is_none());
}
