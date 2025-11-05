use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, Rng, SeedableRng};

use rumma_core::{ModelBuilder, QuantizationConfig};
use rumma_runtime::Engine;

#[derive(Parser, Debug)]
#[command(author, version, about = "rumma demo CLI", long_about = None)]
struct Args {
    #[arg(long, default_value_t = 4096)]
    hidden_size: usize,

    #[arg(long, default_value_t = 4)]
    layers: usize,

    #[arg(long, default_value_t = 128)]
    group_size: usize,

    #[arg(long, default_value_t = 1)]
    batch: usize,

    #[arg(long, default_value_t = 4)]
    decode_tokens: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let quant_cfg = QuantizationConfig {
        group_size: args.group_size,
        symmetric: true,
        nibble_map: [0, 1, 2, 3, 4, 5, 6, 7],
    };
    let builder = ModelBuilder::new(args.hidden_size, args.layers, quant_cfg);
    let model = Arc::new(builder.build_random(args.seed)?);
    let mut engine = Engine::new(model.clone());
    engine.capture_decode_graph();

    let batch_handle = {
        let scheduler = engine.scheduler();
        scheduler.register_batch(args.batch)
    };

    let mut rng = StdRng::seed_from_u64(args.seed + 1);
    let mut prefill_inputs = vec![0f32; args.batch * args.hidden_size];
    for value in prefill_inputs.iter_mut() {
        *value = rng.gen_range(-1.0..1.0);
    }
    let start = Instant::now();
    engine.prefill_batch(&batch_handle, &prefill_inputs)?;
    let prefill_duration = start.elapsed();

    println!(
        "Prefill complete: batch={} hidden={} layers={} duration={:.2?}",
        args.batch, args.hidden_size, args.layers, prefill_duration
    );

    let mut next_inputs: HashMap<u64, Vec<f32>> = batch_handle
        .sequence_ids()
        .iter()
        .map(|&id| {
            let cached = engine
                .cached_hidden(id)
                .expect("prefill should populate cache");
            (id, cached)
        })
        .collect();

    let bar = ProgressBar::new((args.decode_tokens * args.batch) as u64);
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} decode {pos}/{len} tokens")?
            .tick_chars("⠁⠃⠇⠧⠷⠿⠷⠧⠇⠃"),
    );

    let mut sampled_prob = 0.0f32;
    for _step in 0..args.decode_tokens {
        for &sequence_id in batch_handle.sequence_ids() {
            let input = next_inputs
                .get(&sequence_id)
                .expect("missing cached state")
                .clone();
            let output = engine.decode_step(sequence_id, &input)?;
            sampled_prob = output
                .probabilities
                .first()
                .copied()
                .unwrap_or(sampled_prob);
            next_inputs.insert(sequence_id, output.logits);
            bar.inc(1);
        }
    }
    bar.finish_with_message("decode complete");

    println!(
        "Decode finished: steps={} sampled_prob={:.6}",
        args.decode_tokens, sampled_prob
    );

    for &sequence_id in batch_handle.sequence_ids() {
        engine.retire_sequence(sequence_id);
    }

    Ok(())
}
