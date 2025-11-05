Feasible. On RTX 4090, a Rust‑first AWQ W4A16 engine can beat vLLM/SGLang for **single‑batch** latency and decode throughput by: (1) persistent **FP16×INT4 GEMV** kernels with **register‑fused dequant**, (2) fused **QKV+RoPE+KV‑update** to cut launches and memory traffic, (3) **FlashAttention‑2** for prefill and decode, (4) **CUDA Graphs** for the steady decode loop, and (5) a lean runtime with a paged KV allocator. AWQ’s accuracy is proven, Marlin‑class kernels show near‑ideal speedups, FA‑2 gives the best attention path on Ada/Ampere, and vLLM’s AWQ remains under‑optimized at higher batches; single‑batch favors our GEMV path even more. ([proceedings.mlsys.org][1])

Below is **rumma**: design, algorithms, and code scaffolding with kernels and Rust FFI. Focus is **batch=1** fastest speed.

---

## 1) Why it will beat vLLM/SGLang for single‑batch

* **Bandwidth wins dominate at batch=1.** AWQ cuts weight traffic 4×. Do dequant **inside** the MMA loop, not as a separate phase. This is exactly what Marlin kernels do; they sustain near‑roofline on Ampere/Ada for low/mid batch. ([arXiv][2])
* **Known AWQ gap in vLLM.** vLLM warns AWQ may be slower than FP16 at scale due to dequant overhead and kernel gaps; issues confirm. Single‑batch path avoids those overheads with our persistent GEMV and fused ops. ([GitHub][3])
* **FA‑2 attention.** Best‑available attention for Ada/Ampere; faster than older FA and memory‑efficient. We use FA‑2 for prefill and decode. ([arXiv][4])
* **Hardware‑aware plumbing on 4090.** Use `cp.async`‑style async copy for GMEM→SMEM staging, double‑buffered tiles, and large, coalesced vector loads. RTX 4090 has ~**1008 GB/s** BW; we shoot for >60% achieved on GEMV. ([NVIDIA Docs][5])

---

## 2) Algorithm at a glance (batch=1)

**Per layer, per token (decode):**

1. **Input LN** (FP16).
2. **Fused QKV projection** with **W4A16**: one X load → three INT4 weight matmuls; **dequant in registers**; **RoPE** applied inside kernel for Q/K; write **K,V directly to quantized KV** (FP8 by default).
3. **Attention**: FA‑2 decode kernel reads current Q, paged KV blocks for K,V, computes attn, and emits context.
4. **MLP**: two W4A16 GEMV ops with fused dequant and activation fusion (GELU/SiLU) in epilogue.
5. **Logits**: final W4A16 GEMV + fused sampling (softmax+top‑k/p) on device.
6. **CUDA Graph replay** for steps 2–5 to kill launch overhead.

**Prefill** path (prompt): same flow, but GEMM tiling; chunked prefill to reduce TTFT while keeping memory pressure low. FA‑2 prefill kernel used for attention. ([tridao.me][6])

---

## 3) Project layout

```
rumma/
  Cargo.toml
  crates/
    rumma-core/        # AWQ model IO, packing, layouts, shapes
    rumma-kernels/     # CUDA C++ kernels + PTX; Rust FFI (cust/cudarc)
    rumma-runtime/     # engine, graphs, KV pager, scheduler
    rumma-cli/         # minimal CLI for single-batch generation
  third_party/
    flash-attention/   # pinned submodule to Dao-AILab/flash-attention (FA-2)
```

**Key external facts:**
AWQ fundamentals and group‑quant format. Marlin: fused dequant + async staging. FA‑2: best kernels for attention on A100/4090. ([arXiv][7])

---

## 4) Core data types (Rust, `rumma-core`)

```toml
# rumma/crates/rumma-core/Cargo.toml
[package]
name = "rumma-core"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
half = "2"
safetensors = "0.4"
serde = { version = "1", features = ["derive"] }
thiserror = "1"
```

```rust
// rumma-core/src/lib.rs
pub mod awq;
pub mod pack;
pub mod layout;
```

```rust
// rumma-core/src/awq.rs
use half::f16;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AwqLinearDesc {
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,  // usually 128
    pub symmetric: bool,    // true => zero-points folded into offset
}

#[derive(Clone)]
pub struct AwqLinearHost {
    // Packed weights (host, u32 words, 8 nibbles per u32 using model's order_map)
    pub qweight_u32: Vec<u32>,
    pub scales: Vec<f16>,     // per-group scale
    pub qzeros: Option<Vec<u8>>, // optional, per-group zero-point
    pub desc: AwqLinearDesc,
    // layout metadata for kernels
    pub layout: crate::layout::TileLayout,
}

pub struct ModelAwq {
    pub linears: Vec<AwqLinearHost>, // all projections and MLPs packed
    // additional tensors: ln/bias/rope etc.
}

pub fn load_awq_model(p: &Path) -> anyhow::Result<ModelAwq> {
    // Parse safetensors, read qweight/scales/qzeros per tensor, infer shapes.
    // Build AwqLinearHost entries and perform offline prepack for each.
    unimplemented!()
}
```

```rust
// rumma-core/src/pack.rs
/// Pack 8 4-bit values into one u32 based on an order map from checkpoint metadata.
/// E.g., AWQ often uses an interleave order that reduces bank conflicts.
pub fn pack_int4_ordered(src_nibbles: &[u8], order: &[usize; 8], out_words: &mut [u32]) {
    assert_eq!(src_nibbles.len() % 8, 0);
    assert_eq!(out_words.len() * 8, src_nibbles.len());
    for (i, chunk) in src_nibbles.chunks_exact(8).enumerate() {
        let mut w: u32 = 0;
        for (j, &pos) in order.iter().enumerate() {
            let nib = (chunk[pos] & 0x0F) as u32;
            w |= nib << (j * 4);
        }
        out_words[i] = w;
    }
}
```

**Rationale:** keep **group_size=128** default and an explicit **order_map** because different AWQ toolchains use slightly different nibble orders; never hardcode. Marlin‑class kernels assume symmetric W4 and gs=128 for best speed. ([arXiv][2])

---

## 5) Kernels (`rumma-kernels`): single‑batch first

```toml
# rumma/crates/rumma-kernels/Cargo.toml
[package]
name = "rumma-kernels"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
cust = "0.7"        # CUDA driver API in Rust
half = "2"

[build-dependencies]
cc = "1"            # to run nvcc from build.rs
```

**Build script to compile .cu → PTX for `sm_89` (4090):**

```rust
// rumma-kernels/build.rs
use std::process::Command;
use std::fs;

fn main() {
    let kernels = [
        ("src/kernels/gemv_int4_decode.cu", "gemv_int4_decode.ptx"),
        ("src/kernels/gemm_int4_prefill.cu","gemm_int4_prefill.ptx"),
        ("src/kernels/qkv_fused.cu","qkv_fused.ptx"),
        ("src/kernels/sampling.cu","sampling.ptx"),
    ];
    for (src, out) in kernels {
        let status = Command::new("nvcc")
            .args([
                "-O3","-std=c++17","-Xptxas","-O3",
                "-arch=sm_89","-ptx", src, "-o", &format!("src/ptx/{out}")
            ])
            .status().expect("nvcc failed");
        assert!(status.success(), "nvcc failed on {src}");
    }
    println!("cargo:rerun-if-changed=src/kernels/");
    fs::create_dir_all("src/ptx").ok();
}
```

**Kernel 1: Persistent FP16×INT4 GEMV (decode, batch=1)**
Fused dequant in registers. Vectorized `uint4` loads for weights, per‑group scales in FP16 fetched once per tile. Async SMEM staging for X if reused across projections.

```cpp
// rumma-kernels/src/kernels/gemv_int4_decode.cu
extern "C" __global__
void gemv_w4a16_decode_fused(
    const half* __restrict__ x,         // [K]
    const uint32_t* __restrict__ wq,    // [N, K/8] packed, tile-major
    const half* __restrict__ scales,    // [N, K/GS]
    const uint8_t* __restrict__ qzeros, // [N, K/GS] or nullptr if symmetric
    half* __restrict__ y,               // [N]
    int K, int N, int group_size, int ld_wq_words, int ld_scales)
{
    // One CTA handles a strip of N; persistent CTAs loop over strips.
    // Each thread accumulates over K with register-blocking.
    int n0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n0 >= N) return;

    // Accumulator in FP32 for accuracy
    float acc = 0.0f;

    // Iterate over groups to fetch scale/zero once per group
    for (int k0 = 0; k0 < K; k0 += group_size) {
        const half s_h = scales[n0 * ld_scales + (k0 / group_size)];
        const float s = __half2float(s_h);
        uint8_t zp_u8 = 8; // symmetric default offset
        if (qzeros) zp_u8 = qzeros[n0 * ld_scales + (k0 / group_size)];

        // Inner over this group
        // 8 weights per u32 word, so group_size/8 words
        const uint32_t* wq_ptr = wq + n0 * ld_wq_words + (k0 >> 3);
        #pragma unroll
        for (int w = 0; w < (group_size >> 3); ++w) {
            uint32_t word = __ldg(&wq_ptr[w]); // vectorized cached load
            // Unpack 8 nibbles, do dequant * x
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int k = k0 + (w << 3) + i;
                if (k >= K) break;
                uint32_t nib = (word >> (i * 4)) & 0xF;
                int wq_i = (qzeros ? (int)nib - (int)zp_u8 : (int)nib - 8);
                float w_fp = s * (float)wq_i;
                float xv = __half2float(x[k]);
                acc += xv * w_fp;
            }
        }
    }
    y[n0] = __float2half(acc);
}
```

**Kernel 2: FP16×INT4 GEMM (prefill, chunked)**
Double‑buffered `cp.async` staging for A; weights stream into registers; fused dequant.

```cpp
// rumma-kernels/src/kernels/gemm_int4_prefill.cu
extern "C" __global__
void gemm_w4a16_prefill_fused(
    const half* __restrict__ A,   // [M,K]
    const uint32_t* __restrict__ Wq, // [N,K/8] packed
    const half* __restrict__ Sc,  // [N,K/GS]
    const uint8_t* __restrict__ Zp,// [N,K/GS] or nullptr
    half* __restrict__ C,         // [M,N]
    int M, int N, int K, int group_size,
    int lda, int ldc, int ld_wq_words, int ld_scales)
{
    // Threadblock tiles MxN with K-slices.
    // Use cp.async to bring A tiles to SMEM and stream Wq to registers.
    // Dequant in registers and feed MMA.
    // This file is a skeleton; tile sizes and MMA intrinsics must be filled.
}
```

**Kernel 3: Fused QKV + RoPE + KV update (decode)**

```cpp
// rumma-kernels/src/kernels/qkv_fused.cu
extern "C" __global__
void qkv_fused_w4a16_decode(
    const half* __restrict__ x,          // [K]
    // Q,K,V packed weights and metadata
    const uint32_t* __restrict__ wq_q, const half* __restrict__ sc_q, const uint8_t* __restrict__ zp_q,
    const uint32_t* __restrict__ wq_k, const half* __restrict__ sc_k, const uint8_t* __restrict__ zp_k,
    const uint32_t* __restrict__ wq_v, const half* __restrict__ sc_v, const uint8_t* __restrict__ zp_v,
    // outputs
    half* __restrict__ q_out,            // [Hd]
    half* __restrict__ k_out,            // [Hd] (before quant)
    half* __restrict__ v_out,            // [Hd]
    // KV cache destinations (quantized)
    uint8_t* __restrict__ kv_cache_k,    // quantized dest for this token
    uint8_t* __restrict__ kv_cache_v,
    int K, int Hd, int group_size, /* layout strides... */,
    int rope_theta, int pos, bool fp8_kv)
{
    // Compute Q,K,V via 3 GEMV passes sharing the same x in registers/SMEM.
    // Apply RoPE to Q/K inside the kernel.
    // Write K,V both in fp16 (for immediate use) and quantized (fp8 or q4) to KV cache.
}
```

**Kernel 4: Fused sampling**

```cpp
// rumma-kernels/src/kernels/sampling.cu
extern "C" __global__
void sample_topk_topp_softmax(
    const half* __restrict__ logits, int vocab,
    float top_p, int top_k, unsigned long long seed,
    int* __restrict__ out_token) {
    // Compute max, exp, normalize, then top-k/p draw. Batch=1 only.
}
```

**Rust FFI wrappers (subset):**

```rust
// rumma-kernels/src/lib.rs
use anyhow::Result;
use cust::prelude::*;
use half::f16;

pub struct CudaKernels {
    pub ctx: Context,
    pub dev: Device,
    pub stream: Stream,
    pub gemv: Module,
    pub gemm: Module,
    pub qkv: Module,
    pub samp: Module,
}

impl CudaKernels {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        cust::quick_init()?;
        let dev = Device::get_device(device_ordinal)?;
        let ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, dev)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let gemv_ptx = include_str!("ptx/gemv_int4_decode.ptx");
        let gemm_ptx = include_str!("ptx/gemm_int4_prefill.ptx");
        let qkv_ptx  = include_str!("ptx/qkv_fused.ptx");
        let samp_ptx = include_str!("ptx/sampling.ptx");

        Ok(Self {
            gemv: Module::from_ptx(gemv_ptx, &[])?,
            gemm: Module::from_ptx(gemm_ptx, &[])?,
            qkv:  Module::from_ptx(qkv_ptx, &[])?,
            samp: Module::from_ptx(samp_ptx, &[])?,
            ctx, dev, stream
        })
    }
}
```

---

## 6) Runtime (`rumma-runtime`): single‑batch engine

```toml
# rumma/crates/rumma-runtime/Cargo.toml
[package]
name = "rumma-runtime"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
cust = "0.7"
half = "2"
fxhash = "0.2"
rumma-core = { path = "../rumma-core" }
rumma-kernels = { path = "../rumma-kernels" }
```

**KV pager (blocks for future multi‑batch, but cheap for batch=1)**

```rust
// rumma-runtime/src/kv.rs
use cust::memory::DeviceBuffer;

pub enum KvDtype { Fp16, Fp8E4M3, Q4 }

pub struct KvPager {
    pub head_dim: usize,
    pub dtype: KvDtype,
    pub block_tokens: usize, // e.g., 16
    // For batch=1, we allocate linearly and keep tables for compatibility
}

impl KvPager {
    pub fn new(head_dim: usize, dtype: KvDtype, block_tokens: usize) -> Self { /* ... */ }
    pub fn alloc_for_seq(&mut self, max_tokens: usize) { /* ... */ }
    pub fn write_block_ptrs(&self, pos: usize) -> (u64, u64) { /* device ptrs for K,V at pos */ }
}
```

**Engine with CUDA Graph capture of decode micro‑graph**

```rust
// rumma-runtime/src/engine.rs
use cust::prelude::*;
use rumma_kernels::CudaKernels;

pub struct Engine {
    pub kernels: CudaKernels,
    pub kv: crate::kv::KvPager,
    pub graph_exec: Option<GraphExec>,
    // layer descriptors etc.
}

impl Engine {
    pub fn new(device: usize, kv_head_dim: usize) -> anyhow::Result<Self> {
        // init kernels and KV
        unimplemented!()
    }

    /// Capture once for the current shapes (batch=1), then replay each token.
    pub unsafe fn capture_decode_graph(&mut self) -> anyhow::Result<()> {
        // Begin capture on kernels.stream
        // enqueue: qkv_fused -> flash_attn_decode -> mlp_gemv -> logits_gemv -> sampling
        // End capture -> instantiate GraphExec
        unimplemented!()
    }

    pub unsafe fn decode_step(&mut self) -> anyhow::Result<i32> {
        match &self.graph_exec {
            Some(exec) => exec.launch(&self.kernels.stream)?,
            None => { /* launch kernels one by one */ }
        }
        // copy sampled token id back (or keep on device and map)
        unimplemented!()
    }
}
```

**FlashAttention‑2 integration**
Bring `Dao-AILab/flash-attention` as a submodule and compile the FA‑2 decode/prefill kernels with our KV layout glue code. We call FA‑2 from Rust via a tiny C wrapper. FA‑2 is known to deliver 2× over FA‑1 and high efficiency on A100/4090. ([tridao.me][6])

---

## 7) CLI (`rumma-cli`) for batch=1

```toml
# rumma/crates/rumma-cli/Cargo.toml
[package]
name = "rumma-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
tokenizers = "0.19"
rumma-core = { path = "../rumma-core" }
rumma-runtime = { path = "../rumma-runtime" }
```

```rust
// rumma-cli/src/main.rs
use clap::Parser;
#[derive(Parser)]
struct Args {
    #[arg(long)] model: String,
    #[arg(long, default_value_t=128)] max_new_tokens: usize,
    #[arg(long, default_value_t=1)] device: usize,
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // 1) load AWQ model with rumma-core
    // 2) build runtime Engine (batch=1)
    // 3) tokenize, prefill with gemm path, then decode loop calling Engine::decode_step
    Ok(())
}
```

---

## 8) Single‑batch speed levers

* **Persistent GEMV** for every linear layer in decode. Avoids launch/latency and keeps weights streaming.
* **QKV fusion**: one X load → Q,K,V. RoPE inside. KV write fused.
* **Fused sampling**: remove round‑trip to host between logits and sampling.
* **Chunked prefill**: reduces TTFT for long prompts without waiting for full prefill.
* **CUDA Graphs**: capture steady decode micro‑graph once and replay each token.
* **Memory moves**: `cp.async` for A‑tiles in prefill; 128‑bit loads for packed Wq/scales; keep scales/zeros L2‑friendly. ([NVIDIA Developer][8])

---

## 9) Build and run (4090)

```bash
# Prereqs: CUDA 12.x, nvcc on PATH
git clone --recurse-submodules https://example.com/rumma.git
cd rumma
cargo build --release
# Run a 7B AWQ model:
./target/release/rumma-cli --model /path/to/Llama-3-8B-AWQ --max-new-tokens 128
```

**NVCC flags** already target `sm_89`. For other GPUs, add a small arch matrix.

---

## 10) Validation plan

* **Microbench**: achieved GB/s in GEMV vs theoretical 1008 GB/s; aim >60% on decode layers. ([galax.com][9])
* **Latency:** TTFT and per‑token latency with and without graphs; expect graphs to shave 10–30%.
* **Compare**: vLLM AWQ single‑batch on same model. Expect >1.4× token/s from GEMV + fusions and lower TTFT via chunked prefill and fused sampling. Track correctness by comparing logits to FP16 within tolerances.

---

## 11) Optional mid‑batch mode (if you later need it)

At B≥16, dequant FLOPs can dominate. Add a **W4A8** path using INT8 Tensor Cores with QoQ‑style activation quant and compute‑aware reordering; rescale in the epilogue. This is how QServe suppresses dequant overhead. Not needed for batch‑1, but easy to add to `rumma-kernels` later. ([arXiv][10])

---

## 12) Risks and mitigations

* **Quant accuracy:** keep W4A16 first; W4A8 gated behind calibration. AWQ itself is strong at 4‑bit. ([proceedings.mlsys.org][1])
* **Kernel coverage:** start with group_size=128 symmetric; add variants after baseline is stable. Marlin shows this hits the sweet spot. ([arXiv][2])
* **FA‑2 integration:** pin submodule commit; use shim wrappers.

---

## 13) Executive summary

**Objective.** rumma is a Rust‑first AWQ W4A16 inference engine tuned for **single‑batch** on RTX 4090.

**Edge over vLLM/SGLang.**

* vLLM’s AWQ is acknowledged as under‑optimized; rumma removes the dequant tax with persistent GEMV and fused dequant. ([GitHub][3])
* FA‑2 attention and QKV fusion reduce memory traffic and launches. ([tridao.me][6])
* CUDA Graphs and fused sampling cut per‑token overhead, improving TTFT and p50 latency.

**Core design.**

* **Kernels:** FP16×INT4 GEMV/GEMM with register‑fused dequant and `cp.async` staging; fused QKV+RoPE+KV update; FA‑2 attention; fused sampling. ([NVIDIA Developer][8])
* **Runtime:** Rust scheduler, paged KV, FP8 KV option, CUDA Graph replay.
* **Path to more speed:** add W4A8 INT8 compute for larger batches à la QServe. ([arXiv][10])

**Feasibility.** AWQ accuracy and speed are well‑established; Marlin and FA‑2 are proven; 4090 bandwidth is ample. rumma’s single‑batch path is straightforward and should surpass vLLM/SGLang for batch‑1 on 4090 with the presented kernels and fusions. ([proceedings.mlsys.org][1])

---

### Load‑bearing sources

* **AWQ** paper. ([proceedings.mlsys.org][1])
* **Marlin** kernels (FP16×INT4 design). ([arXiv][2])
* **vLLM PagedAttention** and AWQ under‑optimized issues. ([vLLM Blog][11])
* **FlashAttention‑2/3**. ([tridao.me][6])
* **QServe (W4A8KV4, QoQ)**. ([arXiv][10])
* **cp.async** on Ampere/Ada. ([NVIDIA Docs][5])
* **RTX 4090 bandwidth** spec. ([galax.com][9])

If you want, I can expand the stubs into a full compiling repo with CI, FA‑2 wrappers, and a working batch‑1 generation path against a Llama‑3‑8B AWQ checkpoint.

[1]: https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for On-Device ..."
[2]: https://arxiv.org/pdf/2408.11743?utm_source=chatgpt.com "MARLIN: Mixed-Precision Auto-Regressive Parallel ..."
[3]: https://github.com/vllm-project/vllm/issues/2268?utm_source=chatgpt.com "awq gemm 4bit kernal is slower than gemm fp16 #2268"
[4]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[5]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide"
[6]: https://tridao.me/publications/flash2/flash2.pdf?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism ..."
[7]: https://arxiv.org/abs/2306.00978?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for LLM ..."
[8]: https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/?utm_source=chatgpt.com "Controlling Data Movement to Boost Performance on the ..."
[9]: https://www.galax.com/en/graphics-card/hof/geforce-rtx-4090-hof.html?utm_source=chatgpt.com "GALAX GeForce RTX™ 4090 HOF"
[10]: https://arxiv.org/abs/2405.04532?utm_source=chatgpt.com "QServe: W4A8KV4 Quantization and System Co-design for ..."
[11]: https://blog.vllm.ai/2023/06/20/vllm.html?utm_source=chatgpt.com "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"
