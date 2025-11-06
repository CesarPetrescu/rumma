# CUDA GPU Acceleration

Rumma supports CUDA GPU acceleration for significantly faster inference performance.

## Features

- **Automatic Backend Selection**: Rumma automatically detects CUDA availability and uses GPU when possible
- **CPU Fallback**: If CUDA is not available, Rumma seamlessly falls back to multi-threaded CPU execution
- **cuBLAS Integration**: Uses highly optimized cuBLAS for matrix operations
- **Minimal Code Changes**: The same code works with both CPU and GPU backends

## Prerequisites

To use CUDA acceleration, you need:

1. **NVIDIA GPU** with compute capability 3.5 or higher
2. **CUDA Toolkit** installed (version 11.8 or 12.x)
   - Download from: https://developer.nvidia.com/cuda-downloads
3. **CUDA libraries** in your system path

### Verifying CUDA Installation

Check if CUDA is properly installed:

```bash
nvcc --version  # Should show CUDA compiler version
nvidia-smi      # Should show GPU information
```

## Building with CUDA Support

### For CUDA 12.x

```bash
cargo build --release --features cuda-12
```

### For CUDA 11.x

```bash
cargo build --release --features cuda-11
```

### Running with CUDA

Once built with CUDA support, simply run as normal:

```bash
cargo run --release --features cuda-12 -p rumma-cli -- \
    --model https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ
```

You should see:
```
🚀 Using CUDA GPU acceleration
```

If CUDA initialization fails, you'll see:
```
⚠️  CUDA initialization failed: <error message>
   Falling back to CPU
🖥️  Using CPU (multi-threaded)
```

## Default Behavior (CPU-only)

By default, Rumma builds without CUDA to ensure compatibility on all systems:

```bash
cargo build --release
```

This creates a CPU-only binary that works everywhere without requiring CUDA.

## Performance Comparison

Expected performance improvements with CUDA (varies by model and GPU):

| Operation | CPU (32 cores) | GPU (RTX 4090) | Speedup |
|-----------|----------------|----------------|---------|
| Prefill   | 100 ms         | 10 ms          | 10x     |
| Decode    | 50 ms          | 5 ms           | 10x     |

*Note: Actual performance depends on model size, batch size, and hardware.*

## Troubleshooting

### "CUDA initialization failed"

1. **Check CUDA installation**: Run `nvidia-smi` and `nvcc --version`
2. **Set CUDA_ROOT**: `export CUDA_ROOT=/usr/local/cuda`
3. **Check library path**: Ensure CUDA libraries are in `LD_LIBRARY_PATH`

### "unsupported GPU architecture"

Your GPU is too old. CUDA requires compute capability 3.5+.

### Build errors

1. **Wrong CUDA version**: Make sure you're using the correct feature flag (cuda-11 or cuda-12)
2. **Missing CUDA**: Install CUDA Toolkit from NVIDIA
3. **Version mismatch**: The feature flag must match your installed CUDA version

## Advanced: Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPU to use (default: 0)
  ```bash
  CUDA_VISIBLE_DEVICES=1 cargo run --features cuda-12 ...
  ```

- `CUDA_FORCE_CPU`: Force CPU backend even with CUDA support
  ```bash
  CUDA_FORCE_CPU=1 cargo run --features cuda-12 ...
  ```

## Architecture

The CUDA backend uses:

- **cuBLAS**: For matrix multiplications (GEMM/GEMV)
- **CUDA Unified Memory**: For efficient host-device transfers
- **Lazy Initialization**: CUDA device is only initialized when needed

Future optimizations planned:
- [ ] On-GPU dequantization
- [ ] Kernel fusion (GEMM + ReLU)
- [ ] Multi-GPU support
- [ ] FP16 inference
