use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueHint};
use dirs::cache_dir;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rumma_core::{load_awq_model, AwqModel, Model, ModelBuilder, QuantizationConfig};
use rumma_runtime::Engine;

#[cfg(feature = "gui")]
mod gui;

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

    #[arg(
        long,
        help = "Enter interactive chat mode instead of running benchmark"
    )]
    interactive: bool,

    #[arg(long, help = "Launch the application in GUI mode")]
    gui: bool,
}

enum ModelSelection {
    Random,
    Awq {
        paths: Vec<PathBuf>,
        origin: String,
        repo_id: Option<String>,
    },
}

#[cfg(feature = "gui")]
use std::sync::mpsc;

#[cfg(feature = "gui")]
pub fn resolve_and_load_model(
    repo_id: &str,
    status_tx: Option<mpsc::Sender<gui::GuiMessage>>,
) -> Result<AwqModel> {
    // This function will be a stripped-down version of the CLI's model loading
    // It needs to be callable from the GUI thread.

    let mut args = Args::parse_from(Vec::<String>::new());
    args.hf_repo = Some(repo_id.to_string());

    let selection = resolve_model_selection(&args)?;

    let model = match selection {
        ModelSelection::Random => {
            bail!("Random model not supported in GUI mode");
        }
        ModelSelection::Awq { paths, .. } => {
            if let Some(tx) = &status_tx {
                tx.send(gui::GuiMessage::Status(
                    "Loading model from disk...".to_string(),
                ))
                .unwrap();
            }
            let model = build_awq_model(&paths)?;
            if let Some(tx) = &status_tx {
                tx.send(gui::GuiMessage::Status(
                    "Model loaded successfully.".to_string(),
                ))
                .unwrap();
            }
            model
        }
    };
    Ok(model)
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.gui {
        #[cfg(feature = "gui")]
        return gui::run().map_err(|e| anyhow::anyhow!("GUI Error: {}", e));
        #[cfg(not(feature = "gui"))]
        bail!("GUI feature is not enabled.");
    }

    let selection = resolve_model_selection(&args)?;

    let (model, hidden_size, depth, origin, awq_layers, repo_id): (
        Arc<dyn Model>,
        usize,
        usize,
        String,
        Vec<String>,
        Option<String>,
    ) = match selection {
        ModelSelection::Random => {
            let quant_cfg = QuantizationConfig::default();
            let builder = ModelBuilder::new(args.hidden_size, args.layers, quant_cfg);
            let model = builder.build_random(args.seed)?;
            (
                Arc::new(model) as Arc<dyn Model>,
                args.hidden_size,
                args.layers,
                String::from("random demo weights"),
                Vec::new(),
                None,
            )
        }
        ModelSelection::Awq {
            paths,
            origin,
            repo_id,
        } => {
            println!("🔧 Loading model from disk...");
            let awq_model = build_awq_model(&paths)?;
            let hidden = awq_model.config.hidden_size;
        let depth = awq_model.depth();
            let layer_names = awq_model
            .clone()
            .into_layers()
            .into_iter()
            .map(|l| l.name)
                .collect();
            println!("✓ Model loaded successfully\n");
            (
                Arc::new(awq_model) as Arc<dyn Model>,
                hidden,
                depth,
                format!("{origin} ({} files)", paths.len()),
                layer_names,
                repo_id,
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

    // Handle interactive mode
    if args.interactive {
        if repo_id.is_some() {
            println!("\n🤖 Entering interactive chat mode...\n");
            let awq_model = model
                .as_any()
                .downcast_ref::<AwqModel>()
                .cloned()
                .context("failed to downcast model to AwqModel")?;
            run_interactive_mode(awq_model)?;
            return Ok(());
        } else {
            bail!("Interactive mode requires a HuggingFace model (use --model with a HuggingFace URL)");
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
            let (paths, origin) = download_hf_checkpoint(&repo_id, args)?;
            return Ok(ModelSelection::Awq {
                paths,
                origin,
                repo_id: Some(repo_id),
            });
        } else {
            // Treat as file path
            return Ok(ModelSelection::Awq {
                paths: vec![PathBuf::from(model_arg)],
                origin: String::from("local checkpoint"),
                repo_id: None,
            });
        }
    }

    if let Some(repo_id) = &args.hf_repo {
        let (paths, origin) = download_hf_checkpoint(repo_id, args)?;
        return Ok(ModelSelection::Awq {
            paths,
            origin,
            repo_id: Some(repo_id.clone()),
        });
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

fn build_awq_model(paths: &[PathBuf]) -> Result<AwqModel> {
    load_awq_model(paths)
}

fn download_hf_checkpoint(repo_id: &str, args: &Args) -> Result<(Vec<PathBuf>, String)> {
    let revision = args
        .revision
        .clone()
        .unwrap_or_else(|| String::from("main"));
    let hf_file = args
        .hf_file
        .clone()
        .unwrap_or_else(|| String::from("model.safetensors"));

    println!("📦 Downloading from HuggingFace: {}", repo_id);
    println!("   Revision: {}", revision);

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
        println!("   Downloading entire repository...");
        let info = repo.info()?;
        for sibling in &info.siblings {
            println!("   Fetching: {}", sibling.rfilename);
            repo.get(&sibling.rfilename)?;
        }
        println!("✓ Download complete\n");

        // After downloading, find all the .safetensors files
        let repo_dir = cache
            .join(repo_spec.folder_name())
            .join("snapshots")
            .join(&info.sha);

        let mut safetensor_files = Vec::new();
        for entry in fs::read_dir(repo_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                safetensor_files.push(path);
            }
        }

        if safetensor_files.is_empty() {
            bail!("no .safetensors files found in {repo_id} revision {revision}");
        }

        // Sort for deterministic order
        safetensor_files.sort();

        let origin = format!("{repo_id}@{revision} (full repo)");
        Ok((safetensor_files, origin))
    } else {
        // Download a single file, or if a sharded model, download all shards
        println!("   Checking for sharded model...");
        let info = repo.info()?;
        let is_sharded = info
            .siblings
            .iter()
            .any(|s| s.rfilename.ends_with(".safetensors") && s.rfilename.contains("-of-"));

        let mut paths = Vec::new();
        if is_sharded {
            println!("   Sharded model detected, downloading all .safetensors files...");
            for sibling in info.siblings {
                if sibling.rfilename.ends_with(".safetensors") {
                    println!("   Fetching: {}", sibling.rfilename);
                    paths.push(repo.get(&sibling.rfilename)?);
                }
            }
            // Sort for deterministic order
            paths.sort();
        } else {
            println!("   Fetching model file: {}", hf_file);
            paths.push(repo.get(&hf_file)?);
        }

        let origin = format!("{repo_id}@{revision}");
        println!("✓ Download complete\n");
        Ok((paths, origin))
    }
}

fn run_interactive_mode(model: AwqModel) -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Welcome to Rumma Interactive Chat!");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Type your message and press Enter to chat.");
    println!("Type 'quit' or 'exit' to leave.\n");
    println!(
        "Model: hidden_size={} layers={}",
        model.config.hidden_size,
        model.layers().len()
    );
    println!("═══════════════════════════════════════════════════════════\n");

    let mut engine = Engine::new(Arc::new(model.clone()) as Arc<dyn Model>);
    engine.capture_decode_graph();

    loop {
        // Show prompt
        print!("You: ");
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit commands
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!("\nGoodbye! 👋\n");
            break;
        }

        // Tokenize input
        let encoding = model
            .tokenizer
            .encode(input, false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {}", e))?;
        let token_ids = encoding.get_ids();

        // Generate response
        print!("\nAssistant: ");
        io::stdout().flush()?;
        let response = generate_response(&mut engine, Arc::new(model.clone()), token_ids)?;
        println!("{}\n", response);
        println!("───────────────────────────────────────────────────────────\n");
    }

    Ok(())
}

fn generate_response(
    engine: &mut Engine,
    model: Arc<AwqModel>,
    input_tokens: &[u32],
) -> Result<String> {
    let mut generated_tokens = Vec::new();

    // Prefill
    let mut input_embeddings = Vec::new();
    for &token in input_tokens {
        let embedding = model.embed_tokens.weight
            [token as usize * model.config.hidden_size..]
            .iter()
            .take(model.config.hidden_size)
            .cloned()
            .collect::<Vec<_>>();
        input_embeddings.extend(embedding);
    }

    let batch_handle = {
        let scheduler = engine.scheduler();
        scheduler.register_batch(1)
    };
    engine.prefill_batch(&batch_handle, &input_embeddings)?;

    let mut next_input = engine
        .cached_hidden(batch_handle.sequence_ids()[0])
        .expect("prefill should populate cache");

    // Decode loop
    for _ in 0..model.config.vocab_size.min(256) {
        let output = engine.decode_step(batch_handle.sequence_ids()[0], &next_input)?;
        let logits = &output.logits;

        // Apply LM head
        let mut output_logits = vec![0.0; model.config.vocab_size];
        for i in 0..model.config.vocab_size {
            let mut logit = 0.0;
            for j in 0..model.config.hidden_size {
                logit += logits[j] * model.lm_head.weight[i * model.config.hidden_size + j];
            }
            output_logits[i] = logit;
        }

        // Sample next token
        let next_token = sample_greedy(&output_logits);
        generated_tokens.push(next_token as u32);

        // Get embedding for next token
        next_input = model.embed_tokens.weight
            [next_token as usize * model.config.hidden_size..]
            .iter()
            .take(model.config.hidden_size)
            .cloned()
            .collect();

        if next_token == 151668 {
            // EOS token for Qwen3
            break;
        }
    }

    engine.retire_sequence(batch_handle.sequence_ids()[0]);

    let response = model
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow!("failed to decode generated tokens: {}", e))?;

    // Qwen3 uses a <think>...</think> block for reasoning.
    if let Some(start) = response.find("<think>") {
        if let Some(end) = response.find("</think>") {
            let thinking_content = &response[start + 7..end];
            let final_response = &response[end + 8..];
            return Ok(format!(
                "\n\n[Thinking]:\n{}\n\n[Response]:\n{}",
                thinking_content.trim(),
                final_response.trim()
            ));
        }
    }

    Ok(response)
}

fn sample_greedy(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
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
