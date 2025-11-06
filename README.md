# rumma
Yet another LLM Inference in Rust

## Features

- **CPU & GPU Support**: Multi-threaded CPU execution with optional CUDA GPU acceleration
- **AWQ Quantization**: Efficient 4-bit quantized inference
- **HuggingFace Integration**: Download and run models directly from HuggingFace Hub
- **Automatic Backend Selection**: Seamlessly switches between GPU and CPU based on availability

## CUDA GPU Acceleration

Rumma supports NVIDIA CUDA for GPU-accelerated inference. See [CUDA_SUPPORT.md](CUDA_SUPPORT.md) for details.

**Quick start with CUDA:**
```bash
# For CUDA 12.x
cargo run --release --features cuda-12 -p rumma-cli -- \
    --model https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ

# For CUDA 11.x
cargo run --release --features cuda-11 -p rumma-cli -- \
    --model https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ
```

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
default cache location. Pass `--hf-download-repo` to eagerly fetch the entire
repository before loading the checkpoint.
