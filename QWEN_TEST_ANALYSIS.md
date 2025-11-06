# Qwen2.5-3B-Instruct-AWQ Test Analysis

**Date**: 2025-11-06
**Branch**: claude/test-all-t-011CUrtxKjErY5vnjP5AsqzS
**Model**: Qwen/Qwen2.5-3B-Instruct-AWQ
**Target**: Generate 100 tokens

## Environment Blockers

### ❌ Critical Issues

1. **Network Access Blocked (403 Errors)**
   - crates.io registry: 403 Access Denied
   - HuggingFace downloads: 403 Access Denied
   - All alternative cargo registries: 403 Access Denied
   - Cannot build Rust project (no dependencies)
   - Cannot download model files

2. **No Cached Dependencies**
   - `~/.cargo/registry/cache/`: Empty
   - No pre-built binaries available
   - Offline build fails (missing all dependencies)

### 🔧 What Would Be Required

To successfully test the Qwen model, one of these solutions is needed:

**Option A - Network Access**
- Restore access to crates.io (for building CLI)
- Restore access to huggingface.co (for downloading model)

**Option B - Pre-provisioned Environment**
- Pre-built `rumma-cli` binary (release mode recommended)
- Pre-downloaded `model.safetensors` from Qwen/Qwen2.5-3B-Instruct-AWQ

**Option C - Vendored Build**
- Vendor all Rust dependencies in the repo
- Pre-download model and commit to repo (not recommended - too large)

---

## Code Analysis: Expected Behavior

Based on comprehensive code review of the rumma CLI implementation, here's what WOULD happen when testing Qwen2.5-3B-Instruct-AWQ:

### Model Information

**Qwen2.5-3B-Instruct-AWQ** is a 3 billion parameter model with:
- **Quantization**: 4-bit AWQ (Activation-aware Weight Quantization)
- **Expected hidden size**: ~3584 or ~4096 (based on model architecture)
- **Expected layers**: ~30-40 transformer layers
- **Format**: SafeTensors with qweight, qzeros, scales tensors

### Test Command

```bash
# Using built-in HuggingFace download (RECOMMENDED)
cargo run --release -p rumma-cli -- \
  --hf-repo Qwen/Qwen2.5-3B-Instruct-AWQ \
  --hf-file model.safetensors \
  --decode-tokens 100 \
  --batch 1 \
  --seed 42

# Or using pre-downloaded model
cargo run --release -p rumma-cli -- \
  --model ~/models/qwen2.5-3b-awq/model.safetensors \
  --decode-tokens 100 \
  --batch 1 \
  --seed 42
```

### Execution Flow Analysis

Based on `crates/rumma-cli/src/main.rs`, here's the expected execution flow:

#### 1. Model Loading Phase (lines 108-149)

**HuggingFace Download Path** (`resolve_model_selection`, line 233):
```
1. Create cache directory: ~/.cache/rumma/checkpoints/
2. Initialize HF API client (hf-hub library)
3. Create repo object: Qwen/Qwen2.5-3B-Instruct-AWQ
4. Download model.safetensors (progress shown by hf-hub)
5. Cache location: ~/.cache/rumma/checkpoints/models--Qwen--Qwen2.5-3B-Instruct-AWQ/
```

**AWQ Model Loading** (`build_awq_model`, line 241):
```
1. Load safetensors file into memory
2. Parse metadata for model configuration
3. Identify all layers with pattern: "*.qweight", "*.qzeros", "*.scales"
4. Extract hidden_size from first layer's column dimension
5. Extract depth (number of layers)
6. Collect layer names (e.g., "model.layers.0.self_attn.q_proj")
```

**Expected Output**:
```
Loaded model: hidden_size=3584 layers=30 source=Qwen/Qwen2.5-3B-Instruct-AWQ/model.safetensors@main
  layer model.layers.0.self_attn.q_proj
  layer model.layers.0.self_attn.k_proj
  layer model.layers.0.self_attn.v_proj
  layer model.layers.0.self_attn.o_proj
  layer model.layers.0.mlp.gate_proj
  ... (30+ layers total)
```

#### 2. Engine Initialization (lines 151-152)

```rust
let mut engine = Engine::new(model.clone());
engine.capture_decode_graph();
```

**What Happens**:
- Creates PagedKvCache with hidden_size slots
- Initializes Scheduler for batch management
- Marks decode graph as "captured" (stub for future CUDA graphs)

#### 3. Batch Registration (lines 154-157)

```rust
let batch_handle = {
    let scheduler = engine.scheduler();
    scheduler.register_batch(1)  // batch=1
};
```

**What Happens**:
- Allocates sequence_id = 0 (since this is first batch)
- Registers in active sequences set
- Returns BatchHandle with sequence_ids=[0]

#### 4. Prefill Phase (lines 159-171)

**Input Generation**:
```rust
let mut prefill_inputs = vec![0f32; 1 * 3584];  // batch=1, hidden=3584
for value in prefill_inputs.iter_mut() {
    *value = rng.gen_range(-1.0..1.0);  // Random values
}
```

**Prefill Execution** (`engine.prefill_batch`):
```
For each of ~30 layers:
  1. GEMM (batch matrix multiply): gemm_prefill()
     - Uses rayon parallel processing
     - Dequantizes 4-bit weights on-the-fly
     - Input: [1, 3584] × Layer[3584, 3584]
     - Output: [1, 3584]
  2. ReLU activation: relu_inplace()
  3. Cache result for next layer
Final: Store hidden state in KV cache for sequence_id=0
```

**Expected Output**:
```
Prefill complete: batch=1 hidden=3584 layers=30 duration=450.00ms
```

*Note: Duration depends on CPU, ~400-800ms on modern x86_64*

#### 5. Decode Phase (lines 173-207)

**Setup**:
```rust
let bar = ProgressBar::new(100);  // 100 tokens × 1 batch
bar.set_style("decode {pos}/{len} tokens");
```

**Decode Loop** (100 iterations):
```
For step in 0..100:
  1. Get cached hidden state from prefill/previous decode
  2. Run decode_step(sequence_id=0, input)
     For each of ~30 layers:
       - GEMV (vector multiply): gemv_decode()
       - Parallel processing with rayon
       - Dequantize weights on-the-fly
       - ReLU activation
     - Final layer output = logits (size 3584)
     - Compute softmax → probabilities
     - Update KV cache with new hidden state
  3. Sample probability: probabilities[0]
  4. Use logits as next input (teacher forcing style)
  5. Update progress bar: "decode 1/100" → "decode 100/100"
```

**Expected Output**:
```
⠿ decode 100/100 tokens
decode complete
Decode finished: steps=100 sampled_prob=0.000123
```

*Note: sampled_prob is just probabilities[0], not a real token sample*

#### 6. Cleanup (lines 214-216)

```rust
for &sequence_id in batch_handle.sequence_ids() {
    engine.retire_sequence(sequence_id);
}
```

**What Happens**:
- Removes sequence_id=0 from active set
- Clears KV cache entry for sequence_id=0
- Frees memory

---

## Performance Expectations

### CPU-based Inference (Current Implementation)

The rumma runtime uses:
- **rayon** for parallel processing
- **4-bit dequantization** on-the-fly during compute
- **CPU-only** execution (no GPU support detected)

**Expected Performance on x86_64 (24-core)**:
```
Prefill (batch=1):     400-800ms
Decode per token:      40-80ms
Total for 100 tokens:  4-8 seconds
```

**Expected Performance on x86_64 (8-core)**:
```
Prefill (batch=1):     1-2s
Decode per token:      100-200ms
Total for 100 tokens:  10-20 seconds
```

### Bottlenecks

1. **Dequantization Overhead**: Each token requires dequantizing weights
2. **No Kernel Fusion**: ReLU is separate pass after GEMM
3. **Memory Bandwidth**: 3B parameters × 4 bits = ~1.5GB to stream per token
4. **No Graph Optimization**: Despite `capture_decode_graph()`, no actual optimization occurs

---

## Code Verification: AWQ Compatibility

### AWQ Loader Analysis (`crates/rumma-core/src/awq_loader.rs`)

The loader supports standard AWQ format:

**Tensor Naming** (lines 72-78):
```rust
// Detects tensors with these suffixes:
- "*.qweight"  // Packed 4-bit quantized weights
- "*.qzeros"   // Packed zero points (optional)
- "*.scales"   // FP16/FP32 dequantization scales
```

**Nibble Packing** (line 13):
```rust
const AWQ_NIBBLE_MAP: [u8; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
```
This matches standard AWQ packing order used by AutoAWQ and llm-awq libraries.

**Metadata Parsing** (lines 292-344):
- Attempts to read custom nibble_map from safetensors metadata
- Falls back to standard AWQ_NIBBLE_MAP
- Supports various metadata key formats

**Dequantization** (lines 203-223):
```rust
// For each weight element:
quantized = (qweight >> (slot * 4)) & 0xF  // Extract 4-bit value
zero = (qzeros >> (slot * 4)) & 0xF        // Extract zero point
scale = scales[group]                       // FP16/FP32 scale
weight = (quantized - zero) * scale         // Dequantize
```

**Transpose** (lines 225-230):
- AWQ stores weights in [in_features, out_features]
- rumma expects [out_features, in_features]
- Automatic transpose during loading

### Compatibility Assessment

✅ **Compatible** with standard AWQ format:
- AutoAWQ exported models
- Transformers + AutoAWQ integration
- GPTQ-to-AWQ converted models

❓ **May require adjustment** for:
- Custom nibble packing schemes
- Non-standard tensor naming
- Fused layers (would need to split)

---

## Limitations and Known Issues

### Current Implementation Limitations

1. **Square Layers Only** (`crates/rumma-core/src/model.rs:22-26`)
   ```rust
   if !layers.iter().all(|layer|
       layer.cols() == hidden && layer.rows() == hidden
   ) {
       return Err(anyhow!("only square layers supported"));
   }
   ```

   **Issue**: Real transformer models have non-square projection layers:
   - QKV projections: [hidden, 3 × hidden]
   - MLP up/down: [hidden, 4 × hidden]

   **Impact**: Will fail to load Qwen model if it contains non-square layers

2. **No Token Sampling** (lines 190-203)
   - Code only reads `probabilities[0]` as a dummy value
   - Does not perform actual token sampling (argmax, top-k, top-p)
   - Reuses same hidden state as input (not realistic inference)

3. **No Tokenization**
   - No tokenizer integration
   - Cannot convert text → tokens or tokens → text
   - Only tests the inference engine with random inputs

4. **CPU-Only Performance**
   - No GPU acceleration
   - No int4 SIMD kernels
   - On-the-fly dequantization overhead

### Potential Compatibility Issues

**Issue #1: Square Layer Assertion**

The biggest blocker for loading real models is the square layer requirement. Looking at Qwen2.5 architecture:
- Likely has `hidden_size=3584`, `intermediate_size=18944`
- Q/K/V projections: [3584, 3584] each ✅
- O projection: [3584, 3584] ✅
- Gate/Up projections: [3584, 18944] ❌ (not square)
- Down projection: [18944, 3584] ❌ (not square)

**Prediction**: Loading Qwen model will likely fail with:
```
Error: only square layers with uniform hidden size are supported in this demo
```

**Fix Required**: Remove square layer restriction in `model.rs`

---

## Test Plan (When Network Available)

### Step 1: Quick Sanity Test with Demo Weights

First verify the CLI works with random weights:

```bash
# Build in release mode for performance
cargo build --release -p rumma-cli

# Quick test (should complete in <1 second)
cargo run --release -p rumma-cli -- \
  --random \
  --hidden-size 128 \
  --layers 2 \
  --decode-tokens 10 \
  --batch 1 \
  --seed 42

# Expected output:
# Loaded model: hidden_size=128 layers=2 source=random demo weights
# Prefill complete: batch=1 hidden=128 layers=2 duration=...
# ⠿ decode 10/10 tokens
# decode complete
# Decode finished: steps=10 sampled_prob=...
```

### Step 2: Download Qwen Model

```bash
# Let rumma CLI handle the download
cargo run --release -p rumma-cli -- \
  --hf-repo Qwen/Qwen2.5-3B-Instruct-AWQ \
  --hf-file model.safetensors \
  --decode-tokens 1 \
  --batch 1

# This will:
# 1. Download model to ~/.cache/rumma/checkpoints/
# 2. Attempt to load AWQ tensors
# 3. Either succeed or fail at square layer check
```

### Step 3: Test 100 Token Generation

If Step 2 succeeds:

```bash
cargo run --release -p rumma-cli -- \
  --model ~/.cache/rumma/checkpoints/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots/*/model.safetensors \
  --decode-tokens 100 \
  --batch 1 \
  --seed 42

# Monitor:
# - Loading time
# - Prefill duration
# - Decode tokens/second
# - Total time for 100 tokens
# - Memory usage (should be ~2-3GB)
```

### Step 4: Performance Benchmarking

```bash
# Vary batch size
for batch in 1 2 4 8; do
  echo "Testing batch=$batch"
  cargo run --release -p rumma-cli -- \
    --model <path> \
    --decode-tokens 100 \
    --batch $batch \
    --seed 42
done

# Vary decode steps
for tokens in 10 50 100 200 500; do
  echo "Testing tokens=$tokens"
  cargo run --release -p rumma-cli -- \
    --model <path> \
    --decode-tokens $tokens \
    --batch 1 \
    --seed 42
done
```

---

## Expected Issues and Fixes

### Issue #1: Square Layer Error

**Symptoms**:
```
Error: only square layers with uniform hidden size are supported in this demo
```

**Root Cause**: `crates/rumma-core/src/model.rs:22-26`

**Fix**:
```rust
// Remove the square layer check, or make it more flexible
impl Model {
    pub fn new(layers: Vec<QuantizedLinear>) -> Result<Self> {
        if layers.is_empty() {
            return Err(anyhow!("model must contain at least one layer"));
        }

        // Infer hidden size from first layer's input dimension
        let hidden = layers[0].cols();

        // Only check that input dimensions are consistent across sequential layers
        // (output of layer N should match input of layer N+1)
        // But rumma's current simplified engine doesn't support this

        Ok(Self {
            layers: Arc::new(layers),
            hidden_size: hidden,
        })
    }
}
```

### Issue #2: AWQ Tensor Not Found

**Symptoms**:
```
Error: layer model.layers.0.self_attn.q_proj is missing qweight tensor
```

**Root Cause**: Model uses different tensor naming convention

**Fix**: Update `awq_loader.rs` to support alternative naming patterns

### Issue #3: Memory Exhausted

**Symptoms**:
```
Error: Cannot allocate memory
```

**Root Cause**: 3B model requires ~2-3GB RAM, system may not have enough

**Fix**: Run on machine with sufficient RAM, or implement memory mapping

---

## Conclusion

### Current Status: ❌ Blocked

Cannot proceed with Qwen model testing due to:
1. Network restrictions (403 errors)
2. Cannot build Rust CLI (no dependencies)
3. Cannot download model (no network access)

### Code Analysis: ⚠️ Likely Incompatible

Based on code review, the rumma CLI will **likely fail** to load Qwen2.5-3B-Instruct-AWQ due to:
- **Square layer restriction** - Real models have non-square MLP layers
- This is a deliberate limitation noted in the code ("only square layers supported in this demo")

### Required Changes

To successfully run Qwen model:
1. ✅ **Network access** (to build and download)
2. ❌ **Remove square layer restriction** (code change needed)
3. ❓ **Verify AWQ format compatibility** (likely works, but needs testing)

### Code Quality: ✅ Excellent

Despite the compatibility issue, the code itself is:
- Well-structured and documented
- Proper error handling
- Clean implementation of AWQ loading
- Good test coverage

The square layer restriction is intentional (marked as "demo"), not a bug.

---

## Recommendations

1. **Immediate**: Document that rumma CLI is a demo/proof-of-concept
2. **Short-term**: Remove square layer restriction to support real models
3. **Medium-term**: Add proper tokenization and sampling
4. **Long-term**: Consider GPU acceleration for practical inference speeds

Once network access is restored, expect to need code modifications before Qwen model works.
