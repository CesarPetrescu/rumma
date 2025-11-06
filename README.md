# rumma
Yet another LLM Inference in Rust

## CLI usage

The `rumma-cli` crate offers a simple demo driver that can either build a
random quantized model or load 4-bit AWQ checkpoints.

### Random demo weights

```bash
cargo run -p rumma-cli -- --hidden-size 4096 --layers 4 --group-size 128
```

### Local AWQ safetensors

```bash
cargo run -p rumma-cli -- --model /path/to/model.safetensors
```

### Download from Hugging Face

```bash
cargo run -p rumma-cli -- --hf-repo my-org/my-model --hf-file model.safetensors --revision main
```

Use `--hf-token` if the repository is private and `--cache-dir` to override the
default cache location.
