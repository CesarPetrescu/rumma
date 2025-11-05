Feasible. On **1× RTX 4090 (Ada, SM 8.9, 1008 GB/s HBM)** a Rust‑first AWQ W4A16 engine can beat vLLM/SGLang for **single‑batch** prefill and decode by eliminating the two main bottlenecks: (1) bandwidth for weight reads and KV cache reads, and (2) launch/dequant overhead. The design below uses **register‑fused INT4→FP16 dequant in the matmul inner loop** (Marlin‑style), **FlashAttention‑2** for attention, **fused QKV+RoPE+KV‑update**, a **persistent GEMV** for decode, and **CUDA Graphs** for the steady decode step. These techniques are standard and proven in the literature; nothing exotic is required, so feasibility is high. ([NVIDIA Images][1])

---

## 1) What “fastest single‑batch on 4090” demands

**Prefill (prompt processing, batch=1):** compute‑bound GEMM. Win by using a **FP16×INT4 GEMM** with **cp.async** double buffering for activations and **register‑fused dequant** for weights. Integrate **QKV fusion** and call **FlashAttention‑2** for attention. ([NVIDIA Docs][2])

**Decode (token generation, batch=1):** memory‑bound GEMV + attention. Win by using a **persistent GEMV** with **register‑fused dequant**, **fused QKV+RoPE+KV‑update**, **decode‑optimized FA‑2**, and **CUDA Graphs** to remove launch overhead; consider **FP8 or Q4 KV** only if dequant cost stays hidden. ([NVIDIA Developer][3])

**Why this beats vLLM/SGLang for batch‑1:**
Their throughput lead comes from batching and KV paging/reuse. Batch‑1 exposes their **AWQ dequant and first‑token overheads**, which the plan below removes: (i) **dequant inside the kernel mainloop**, (ii) **fused operators** to cut launches/memory traffic, and (iii) **CUDA Graph replay** for the per‑token micrograph. vLLM’s own issues and warnings document AWQ under‑optimization and first‑token penalties. ([GitHub][4])

---

## 2) Algorithm, end‑to‑end

### 2.1 Weight‑only quantization (AWQ)

* **Format:** W4A16 with **per‑group scales** and optional zero‑points; group size typically **128**. Keep gs=128 for kernel simplicity and speed. ([MLSys Proceedings][5])
* **Offline prepack (host):** reorder rows/cols to match WMMA tiles; pack 8 nibbles/`u32` using the checkpoint’s nibble order; interleave to avoid bank conflicts. Output **tile‑major Wq**, **Scales**, **Zeros** arrays aligned to 16 B. (AWQ reference + Marlin show why this matters.) ([GitHub][6])

### 2.2 Prefill (batch=1) kernel pipeline

**Goal:** maximize FLOP utilization while keeping DRAM traffic minimal.

1. **Fused QKV projection (GEMM)**

   * Threadblock tile `TB_M×TB_N×TB_K` (e.g., 128×128×64).
   * **`cp.async`** activation tiles A[TB_M×TB_K] → SMEM, double‑buffered; dequantize W4 in registers into **FP16 fragments** directly consumable by WMMA; accumulate FP16/FP32.
   * Emit **Q, K, V** in one pass; do **RoPE in‑kernel** for Q/K. ([NVIDIA Docs][2])
2. **FlashAttention‑2 (prefill)**

   * Standard FA‑2 prefill with tiling across sequence; write context. FA‑2 gives 2× over older FA and high FLOP utilization on A100/4090. ([arXiv][7])
3. **MLP block**

   * Two W4 GEMMs with fused dequant + fused activation (SiLU/GELU) in epilogue.
4. **Chunked prefill**

   * Process the prompt in **chunks** (e.g., 512–2048 tokens) to cut TTFT and keep working sets hot. Capture this loop with **CUDA Graphs** if the chunk shape is static. ([NVIDIA Developer][3])

**Why it scales:** cp.async hides GMEM latency and avoids L1 pollution; dequant never touches DRAM; QKV fusion removes 2 kernel launches and 2 global passes per layer; FA‑2 reduces attention I/O. ([NVIDIA Docs][2])

### 2.3 Decode (batch=1) kernel pipeline

**Goal:** maximize effective GB/s and kill launch overhead.

1. **Fused QKV+RoPE+KV‑update (GEMV)**

   * **Persistent kernel**: one CTA per SM; keep partials in registers; stream packed Wq in **128‑bit** loads; dequant (INT4→FP16) in registers; compute Q/K/V; apply RoPE to Q/K; write **K,V** both in FP16 for immediate use and quantized to KV cache if enabled.
2. **FlashAttention‑2 (decode)**

   * Decode‑specific FA‑2 path reading **paged KV** blocks; single sequence, so the kernel tiles over the length axis with high cache locality. ([arXiv][7])
3. **MLP (GEMV)**

   * Two persistent GEMV passes with fused dequant and activation in epilogue.
4. **Logits + sampling**

   * Final GEMV to vocab + **fused softmax+top‑k/p** sampling on device.
5. **CUDA Graphs**

   * Capture steps 1–4 once for batch‑1, fixed dims; replay per token. (This removes Python/host launch costs and driver latency.) ([NVIDIA Developer][3])

**KV cache dtype:**

* Default **FP16** for maximum speed on 4090; switch to **FP8** if you need 2× context capacity; **Q4 KV** only if you profile a net gain after factoring extra dequant. Evidence: FP8 KV is standard in vLLM; Q4 KV can retain quality but sometimes costs extra compute. Measure on your model. ([VLLM Docs][8])

---

## 3) Why the algorithm is feasible

* **AWQ** is stable and accurate at 4‑bit. MLSys’24 best paper; widely reproduced. ([MLSys Proceedings][5])
* **Marlin‑class FP16×INT4** kernels are public. They demonstrate near‑ideal W4 speedups up to batch 16–32 by **register‑fused dequant + pipelining**; we replicate the ideas on Ada. ([GitHub][9])
* **FlashAttention‑2** is open, well‑documented, and efficient on A100/4090. ([GitHub][10])
* **Paged KV** is standard (vLLM). For batch‑1 it’s not the primary lever, but the pager avoids fragmentation and simplifies future scaling. ([arXiv][11])
* **CUDA Graphs** reliably remove launch overhead in steady loops. ([NVIDIA Developer][3])
* **4090 bandwidth and cache** are known: 1008 GB/s HBM, 72 MB L2. Our plan is bandwidth‑first for decode and Tensor‑Core‑first for prefill. ([NVIDIA Images][1])

---

## 4) Where this beats vLLM/SGLang for batch‑1

* **Dequant overhead:** vLLM issues report AWQ INT4 slower than FP16 where dequant dominates. We move dequant **into the inner MMA loop** and keep data in registers. ([GitHub][4])
* **First‑token latency:** vLLM users report 2–5× slower sampling for first token in AWQ. We fuse sampling and use graphs to cut overhead. ([GitHub][12])
* **Single‑batch emphasis:** vLLM/SGLang are optimized for high concurrency (Paged/RadixAttention). For batch‑1, our persistent GEMV and fusion remove more overhead than their schedulers can. ([arXiv][13])

---

## 5) Micro‑architecture details that matter on 4090

* **cp.async double buffering** for A‑tiles in GEMM; bypass L1 when helpful; use async barriers. ([NVIDIA Docs][2])
* **128‑bit vector loads** for Wq/Scales/Zeros; align buffers; prefetch scales/zeros to keep them L2‑resident (see L2 “persisting” policies). ([NVIDIA Docs][14])
* **Register‑only unpack/dequant:** unpack 8 nibbles/`u32`, map to signed, apply scale/zero in registers; form `half2` before MMA. (Marlin shows the pattern.) ([GitHub][9])
* **Persistent kernels** for GEMV to avoid global rescheduling; target 50–75% occupancy but high ILP.
* **CUDA Graphs** for the decode micrograph (QKV→attn→MLP→logits→sample) to remove per‑token launches. ([NVIDIA Developer][3])

---

## 6) Optional path for very long prompts or mid‑batch use

If you later need large prefill matrices or batch ≥16, add **W4A8 (INT8 compute)** to suppress dequant FLOPs (QoQ from **QServe**) and keep FA‑2. Keep this behind a calibration flag; not required for batch‑1 decode, but it can help batch‑1 **prefill** if your prompt is huge. ([arXiv][15])

---

## 7) “rumma” design (Rust + CUDA)

**Workspace**

```
rumma/
  crates/
    rumma-core/      # AWQ IO + prepack (host)
    rumma-kernels/   # CUDA C++ kernels + PTX, Rust FFI (cust)
    rumma-runtime/   # engine, graphs, paged KV, scheduler
    rumma-cli/       # batch=1 demo runner
  third_party/flash-attention/   # FA-2 submodule
```

**Core data (rumma‑core)**

```rust
pub struct AwqLinearDesc { pub in_features: usize, pub out_features: usize, pub group_size: usize, pub symmetric: bool }
pub struct AwqLinearHost { pub qweight_u32: Vec<u32>, pub scales: Vec<half::f16>, pub qzeros: Option<Vec<u8>>, pub desc: AwqLinearDesc }
pub struct ModelAwq { pub linears: Vec<AwqLinearHost> /* ...layernorms, rope... */ }
```

**Key kernels (rumma‑kernels)**
Signatures only; each uses register‑fused dequant.

```cpp
// 1) Persistent GEMV for decode (W4A16, batch=1)
extern "C" __global__
void gemv_w4a16_decode_fused(const half* x, const uint32_t* wq, const half* sc, const uint8_t* zp,
                             half* y, int K, int N, int group, int ld_wq, int ld_sc);

// 2) GEMM for prefill (cp.async-staged A, Wq in registers)
extern "C" __global__
void gemm_w4a16_prefill_fused(const half* A, const uint32_t* Wq, const half* Sc, const uint8_t* Zp,
                              half* C, int M, int N, int K, int group,
                              int lda, int ldc, int ld_wq, int ld_sc);

// 3) Fused QKV + RoPE + KV write (decode)
extern "C" __global__
void qkv_fused_w4a16_decode(const half* x, /* Wq_q/sc/zp, Wq_k/..., Wq_v/... */,
                            half* q_out, half* k_tmp, half* v_tmp,
                            uint8_t* kv_k_q, uint8_t* kv_v_q,
                            int K, int Hd, int group, int pos, int rope_theta, bool fp8_kv);

// 4) Fused sampling (softmax + top-k/p)
extern "C" __global__
void sample_topk_topp_softmax(const half* logits, int vocab, float top_p, int top_k, unsigned long long seed, int* out_token);
```

**Runtime (rumma‑runtime)**

* **KV pager:** block size 16 tokens; supports `Fp16`, `Fp8`, `Q4` backends; for batch‑1 we just step a linear cursor but keep the same structure.
* **Decode CUDA Graph:** capture `{qkv_fused → fa2_decode → mlp_gemv → logits_gemv → sample}` once; replay per token. ([NVIDIA Developer][3])

**FA‑2 integration**
Pin `Dao‑AILab/flash‑attention` and call the **decode**/**prefill** entrypoints via a tiny C shim. ([GitHub][10])

---

## 8) Validation and guardrails

* **Correctness:** compare layer outputs against FP16 at small batch/length; AWQ error envelopes from the paper are acceptable. ([MLSys Proceedings][5])
* **Profiling targets:**

  * **Prefill:** reach FA‑2‑level attention efficiency; GEMM roofline within reasonable gap; confirm `cp.async` overlapping in Nsight. ([Tri Dao][16])
  * **Decode:** achieved bandwidth on GEMV ≥60% of 1008 GB/s on large layers; graphs reduce per‑token CPU time measurably. ([NVIDIA Images][1])
* **KV dtype A/B:** FP16 vs FP8 vs Q4 for your model/context; pick the fastest at equal quality on your workload. ([VLLM Docs][8])

---

## 9) Risks and mitigations

* **Dequant math cost creeping back:** keep dequant inside the **main MMA loop**, not a pre‑pass; use `half2` FMAs; prefetch scales/zeros to L2; unroll nibble unpack. (Marlin pattern.) ([GitHub][9])
* **Launch overhead:** rely on **CUDA Graphs**; keep capture shapes static; parameterize via graph node params when needed. ([NVIDIA Developer][17])
* **KV quantization trade‑off:** only enable FP8/Q4 after measuring net win; do not default to heavy dequant paths for batch‑1. ([VLLM Docs][8])

---

## 10) Executive algorithm summary (single‑batch on 4090)

* **Prefill:** FP16×INT4 **GEMM with cp.async** + **QKV fusion + RoPE inside** → **FA‑2 prefill** → **fused‑act MLP**.
* **Decode:** **Persistent FP16×INT4 GEMV** + **QKV+RoPE+KV‑update fused** → **FA‑2 decode** → **persistent MLP GEMV** → **fused sampling**; all **captured in a CUDA Graph**.
* **KV dtype:** start FP16; optionally FP8/Q4 if you need more context and it still wins in profiling.
* **Why faster than vLLM/SGLang for batch‑1:** dequant and launch overhead are eliminated where they hurt most; FA‑2 handles attention efficiently; fusion reduces traffic and calls. Evidence for each lever exists in AWQ, Marlin, FA‑2, PagedAttention literature and issues. ([MLSys Proceedings][5])

---

## 11) Core Rust stubs (compilable skeleton)

**`rumma-core/src/pack.rs`** — ordered INT4 packing

```rust
pub fn pack_int4_ordered(src_nibbles: &[u8], order: &[usize; 8], out_words: &mut [u32]) {
    assert_eq!(src_nibbles.len() % 8, 0);
    assert_eq!(out_words.len() * 8, src_nibbles.len());
    for (i, chunk) in src_nibbles.chunks_exact(8).enumerate() {
        let mut w: u32 = 0;
        for (j, &pos) in order.iter().enumerate() {
            w |= ((chunk[pos] & 0x0F) as u32) << (j * 4);
        }
        out_words[i] = w;
    }
}
```

**`rumma-kernels/build.rs`** — compile CUDA for `sm_89`

```rust
use std::process::Command;
fn main() {
    let kernels = [
        ("src/kernels/gemv_int4_decode.cu","src/ptx/gemv_int4_decode.ptx"),
        ("src/kernels/gemm_int4_prefill.cu","src/ptx/gemm_int4_prefill.ptx"),
        ("src/kernels/qkv_fused.cu","src/ptx/qkv_fused.ptx"),
        ("src/kernels/sampling.cu","src/ptx/sampling.ptx"),
    ];
    for (src, out) in kernels {
        assert!(Command::new("nvcc")
            .args(["-O3","-std=c++17","-Xptxas","-O3","-arch=sm_89","-ptx",src,"-o",out])
            .status().unwrap().success());
    }
    println!("cargo:rerun-if-changed=src/kernels/");
}
```

**`rumma-kernels/src/kernels/gemv_int4_decode.cu`** — persistent GEMV (simplified)

```cpp
extern "C" __global__
void gemv_w4a16_decode_fused(const half* __restrict__ x,
                             const uint32_t* __restrict__ wq,
                             const half* __restrict__ sc,
                             const uint8_t* __restrict__ zp,
                             half* __restrict__ y,
                             int K, int N, int group, int ld_wq, int ld_sc) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    #pragma unroll
    for (int k0 = 0; k0 < K; k0 += group) {
        const float s = __half2float(sc[n * ld_sc + (k0 / group)]);
        const uint8_t z = (zp ? zp[n * ld_sc + (k0 / group)] : 8); // 8 => symmetric offset
        const uint32_t* wptr = wq + n * ld_wq + (k0 >> 3);
        #pragma unroll
        for (int w = 0; w < (group >> 3); ++w) {
            uint32_t word = __ldg(&wptr[w]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int k = k0 + (w<<3) + i; if (k >= K) break;
                int8_t q = int((word >> (i*4)) & 0xF) - int(z);
                acc += __half2float(x[k]) * (s * float(q));
            }
        }
    }
    y[n] = __float2half(acc);
}
```

**`rumma-runtime/src/engine.rs`** — graph capture skeleton

```rust
pub unsafe fn capture_decode_graph(&mut self) -> anyhow::Result<()> {
    // begin capture on stream
    // enqueue: qkv_fused → fa2_decode → mlp_gemv → logits_gemv → sampling
    // end capture → instantiate GraphExec
    Ok(())
}
```

This is the minimal spine to start filling in; swap in tuned tiles, FA‑2 wrappers, and the pager.

---

## 12) What to implement first for **batch‑1 wins**

1. **Persistent GEMV** with register‑fused dequant for all linear layers in decode.
2. **Fused QKV+RoPE+KV‑update** kernel.
3. **FA‑2 decode** integration.
4. **Fused sampling** and **CUDA Graphs** for the decode micrograph.
5. **Prefill GEMM** with cp.async and QKV fusion; add **chunked prefill**.

These are the highest‑leverage moves for a single 4090. Each is standard and backed by prior art. ([GitHub][9])

---

### References

* **AWQ (MLSys’24 Best Paper):** algorithm, accuracy, group quant. ([MLSys Proceedings][5])
* **Marlin:** FP16×INT4 fused‑dequant kernels. ([GitHub][9])
* **FlashAttention‑2:** fastest exact attention on A100/4090. ([arXiv][7])
* **PagedAttention (vLLM):** KV paging, batching. ([arXiv][13])
* **vLLM AWQ gaps:** under‑optimized warnings, first‑token penalty, slow INT4 GEMM reports. ([GitHub][4])
* **CUDA Graphs:** reduce launch overhead. ([NVIDIA Developer][3])
* **4090 specs:** 1008 GB/s, 72 MB L2. ([NVIDIA Images][1])
* **KV quantization options:** FP8 in vLLM; Q4 KV in ExLlamaV2 community. ([VLLM Docs][8])

If you want me to expand these stubs into a compiling `rumma` repo with FA‑2 bindings and a working batch‑1 path for a Llama‑3‑8B‑AWQ checkpoint, say the model and max context you care about.

[1]: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf?utm_source=chatgpt.com "NVIDIA ADA GPU ARCHITECTURE"
[2]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html?utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide"
[3]: https://developer.nvidia.com/blog/cuda-graphs/?utm_source=chatgpt.com "Getting Started with CUDA Graphs | NVIDIA Technical Blog"
[4]: https://github.com/vllm-project/vllm/issues/2268?utm_source=chatgpt.com "awq gemm 4bit kernal is slower than gemm fp16 #2268"
[5]: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for On-Device ..."
[6]: https://github.com/mit-han-lab/llm-awq?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for LLM ..."
[7]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[8]: https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html?utm_source=chatgpt.com "Quantized KV Cache - vLLM"
[9]: https://github.com/IST-DASLab/marlin?utm_source=chatgpt.com "IST-DASLab/marlin: FP16xINT4 LLM inference kernel ..."
[10]: https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com "Dao-AILab/flash-attention: Fast and memory-efficient exact ..."
[11]: https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model ..."
[12]: https://github.com/vllm-project/vllm/issues/1545?utm_source=chatgpt.com "The AWQ model's sampling time cost of first generate token ..."
[13]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[14]: https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf?utm_source=chatgpt.com "CUDA C++ Programming Guide"
[15]: https://arxiv.org/abs/2405.04532?utm_source=chatgpt.com "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving"
[16]: https://tridao.me/publications/flash2/flash2.pdf?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism ..."
[17]: https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/?utm_source=chatgpt.com "Constructing CUDA Graphs with Dynamic Parameters"
