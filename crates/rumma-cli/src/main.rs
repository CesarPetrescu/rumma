use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueHint};
use dirs::cache_dir;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, Rng, SeedableRng};

use rumma_core::{load_awq_model, Model, ModelBuilder, QuantizationConfig};
use rumma_runtime::Engine;

#[derive(Parser, Debug)]
#[command(author, version, about = "rumma demo CLI", long_about = None)]
struct Args {
    #[arg(
        long,
        default_value_t = 4096,
        help = "Hidden size for the random demo model"
    )]
    hidden_size: usize,

    #[arg(
        long,
        default_value_t = 4,
        help = "Number of layers for the random demo model"
    )]
    layers: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Quantization group size for the random demo model"
    )]
    group_size: usize,

    #[arg(long, default_value_t = 1)]
    batch: usize,

    #[arg(long, default_value_t = 4)]
    decode_tokens: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(
        long,
        help = "Context size (for informational purposes)"
    )]
    ctx: Option<usize>,

    #[arg(
        long,
        conflicts_with_all = ["hf_repo", "random"],
        help = "Path to an AWQ safetensors checkpoint or HuggingFace model URL"
    )]
    model: Option<String>,

    #[arg(
        long,
        conflicts_with_all = ["model", "random"],
        help = "Hugging Face repo id containing an AWQ checkpoint"
    )]
    hf_repo: Option<String>,

    #[arg(
        long,
        conflicts_with = "random",
        help = "Revision to download from the Hugging Face repo (default: main)"
    )]
    revision: Option<String>,

    #[arg(
        long,
        conflicts_with = "random",
        help = "File to download from the Hugging Face repo (default: model.safetensors)"
    )]
    hf_file: Option<String>,

    #[arg(
        long,
        conflicts_with = "random",
        help = "Token for private Hugging Face repositories"
    )]
    hf_token: Option<String>,

    #[arg(
        long,
        conflicts_with = "random",
        help = "Download the entire Hugging Face repo instead of a single file"
    )]
    hf_download_repo: bool,

    #[arg(long, value_hint = ValueHint::DirPath, help = "Override the Hugging Face cache directory")]
    cache_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Force using the random demo model even if a checkpoint is provided"
    )]
    random: bool,
}

enum ModelSelection {
    Random,
    Awq { path: PathBuf, origin: String },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let selection = resolve_model_selection(&args)?;

    let (model, hidden_size, depth, origin, awq_layers) = match selection {
        ModelSelection::Random => {
            let quant_cfg = QuantizationConfig {
                group_size: args.group_size,
                symmetric: true,
                nibble_map: [0, 1, 2, 3, 4, 5, 6, 7],
            };
            let builder = ModelBuilder::new(args.hidden_size, args.layers, quant_cfg);
            let model = builder.build_random(args.seed)?;
            (
                Arc::new(model),
                args.hidden_size,
                args.layers,
                String::from("random demo weights"),
                Vec::new(),
            )
        }
        ModelSelection::Awq { path, origin } => {
            let (model, hidden, depth, layer_names) = build_awq_model(&path)?;
            (
                Arc::new(model),
                hidden,
                depth,
                format!("{origin} ({})", path.display()),
                layer_names,
            )
        }
    };

    print!(
        "Loaded model: hidden_size={} layers={} source={}",
        hidden_size, depth, origin
    );
    if let Some(ctx) = args.ctx {
        println!(" ctx={}", ctx);
    } else {
        println!();
    }
    if !awq_layers.is_empty() {
        for name in &awq_layers {
            println!("  layer {name}");
        }
    }

    let mut engine = Engine::new(model.clone());
    engine.capture_decode_graph();

    let batch_handle = {
        let scheduler = engine.scheduler();
        scheduler.register_batch(args.batch)
    };

    let mut rng = StdRng::seed_from_u64(args.seed + 1);
    let mut prefill_inputs = vec![0f32; args.batch * hidden_size];
    for value in prefill_inputs.iter_mut() {
        *value = rng.gen_range(-1.0..1.0);
    }
    let start = Instant::now();
    engine.prefill_batch(&batch_handle, &prefill_inputs)?;
    let prefill_duration = start.elapsed();

    println!(
        "Prefill complete: batch={} hidden={} layers={} duration={:.2?}",
        args.batch, hidden_size, depth, prefill_duration
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

fn resolve_model_selection(args: &Args) -> Result<ModelSelection> {
    if args.random {
        return Ok(ModelSelection::Random);
    }

    if let Some(model_arg) = &args.model {
        // Check if it's a HuggingFace URL
        if let Some(repo_id) = parse_huggingface_url(model_arg) {
            let (path, origin) = download_hf_checkpoint(&repo_id, args)?;
            return Ok(ModelSelection::Awq { path, origin });
        } else {
            // Treat as file path
            return Ok(ModelSelection::Awq {
                path: PathBuf::from(model_arg),
                origin: String::from("local checkpoint"),
            });
        }
    }

    if let Some(repo_id) = &args.hf_repo {
        let (path, origin) = download_hf_checkpoint(repo_id, args)?;
        return Ok(ModelSelection::Awq { path, origin });
    }

    Ok(ModelSelection::Random)
}

/// Parse a HuggingFace URL and extract the repo ID
/// Examples:
///   https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ -> Some("Qwen/Qwen2.5-3B-Instruct-AWQ")
///   https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ/tree/main -> Some("Qwen/Qwen2.5-3B-Instruct-AWQ")
///   /path/to/model.safetensors -> None
fn parse_huggingface_url(s: &str) -> Option<String> {
    if !s.starts_with("http://") && !s.starts_with("https://") {
        return None;
    }

    // Parse URL to extract repo ID
    if let Some(url_path) = s.strip_prefix("https://huggingface.co/")
                              .or_else(|| s.strip_prefix("http://huggingface.co/")) {
        // Extract the repo ID (org/model-name)
        // Handle cases like:
        //   Qwen/Qwen2.5-3B-Instruct-AWQ
        //   Qwen/Qwen2.5-3B-Instruct-AWQ/tree/main
        //   Qwen/Qwen2.5-3B-Instruct-AWQ/resolve/main/model.safetensors
        let parts: Vec<&str> = url_path.split('/').collect();
        if parts.len() >= 2 {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    }

    None
}

fn build_awq_model(path: &Path) -> Result<(Model, usize, usize, Vec<String>)> {
    let awq = load_awq_model(path)?;
    let depth = awq.depth();
    let layer_names = awq
        .layers()
        .iter()
        .map(|layer| layer.name.clone())
        .collect::<Vec<_>>();

    // Get the hidden_size from the AWQ model before consuming it
    let hidden_size = awq.hidden_size()
        .context("failed to determine hidden_size from AWQ model")?;

    let quant_layers = awq.into_quantized_layers();

    // Create model directly without ModelBuilder (which is designed for square test models)
    let model = Model::new(quant_layers)?;

    Ok((model, hidden_size, depth, layer_names))
}

fn download_hf_checkpoint(repo_id: &str, args: &Args) -> Result<(PathBuf, String)> {
    let revision = args
        .revision
        .clone()
        .unwrap_or_else(|| String::from("main"));
    let filename = args
        .hf_file
        .clone()
        .unwrap_or_else(|| String::from("model.safetensors"));

    let cache = resolve_cache_dir(args)?;
    fs::create_dir_all(&cache)
        .with_context(|| format!("failed to create Hugging Face cache at {:?}", cache))?;

    let mut builder = ApiBuilder::new();
    if let Some(token) = args.hf_token.clone() {
        builder = builder.with_token(Some(token));
    }
    builder = builder.with_cache_dir(cache.clone());
    let api = builder.build()?;

    let repo_spec = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.clone());
    let repo = api.repo(repo_spec.clone());

    if args.hf_download_repo {
        let info = repo.info()?;
        for sibling in &info.siblings {
            repo.get(&sibling.rfilename)?;
        }

        let repo_dir = cache
            .join(repo_spec.folder_name())
            .join("snapshots")
            .join(&info.sha);
        let full_path = repo_dir.join(&filename);
        if !full_path.exists() {
            bail!("{repo_id} revision {revision} did not contain expected file {filename}");
        }
        let origin = format!("{repo_id}@{revision} (full repo)");
        Ok((full_path, origin))
    } else {
        let path = repo.get(&filename)?;
        let origin = format!("{repo_id}/{filename}@{revision}");
        Ok((path, origin))
    }
}

fn resolve_cache_dir(args: &Args) -> Result<PathBuf> {
    if let Some(dir) = &args.cache_dir {
        return Ok(dir.clone());
    }
    if let Some(base) = cache_dir() {
        return Ok(base.join("rumma").join("checkpoints"));
    }
    Ok(std::env::temp_dir().join("rumma-checkpoints"))
}
