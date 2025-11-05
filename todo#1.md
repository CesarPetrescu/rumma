Build a Rust-first AWQ 4‑bit engine for RTX 4090 that beats today’s stacks by combining Marlin‑class FP16×INT4 kernels, full dequant‑in‑matmul fusion, FlashAttention‑2 for attention, quantized KV cache, continuous batching with paged KV, and aggressive Ampere/Ada CUDA plumbing (cp.async, double‑buffering, CUDA Graphs). The result closes the 2–4× gap you described and should outperform vLLM’s current AWQ path on 4090 at both batch‑1 and mid‑batch, with clear upgrade path to Hopper/Blackwell.

Below is a complete design and implementation plan with code skeletons and the executive summary at the end.

0) Ground truth and gaps to exploit

AWQ is state‑of‑the‑art 4‑bit weight‑only PTQ with strong accuracy; TinyChat shows 3–4× speedups on edge; AWQ is an MLSys’24 paper. 
proceedings.mlsys.org
+1

AutoAWQ is deprecated; users are being directed to newer stacks. vLLM integrates AWQ but marks it under‑optimized. 
GitHub
+1

vLLM PagedAttention cuts KV waste to ~4% and delivers large throughput wins via continuous batching; however AWQ paths often underperform FP16 at larger batches due to dequant overhead. 
vLLM Blog
+2
GitHub
+2

Marlin kernels keep near‑ideal ~4× speedup up to batch 16–32 by moving dequant into register‑blocked Tensor Core loops and carefully staging memory. 
GitHub
+1

QServe proves the dequantization tax can be cut to <5% with W4A8KV4 and progressive quantization, beating TRT‑LLM on L40S/A100 at larger batches. This informs our activation‑quant path. 
arXiv

INT‑FlashAttention and related work show large wins from integer/quantized attention; production stacks largely still run FP16 attention. 
arXiv

TensorRT‑LLM on Hopper uses W4A8 (INT4 weights, FP8/INT8 activations) and custom plugins for big throughput, but that relies on Hopper FP8 features, not on 4090. 
NVIDIA GitHub
+1

Ada (RTX 4090) reality: 1008 GB/s GDDR6X bandwidth, large 72 MB L2; no Hopper TMA/WGMMA; but we do have Ampere/Ada cp.async for GMEM→SMEM overlap. Optimize for SM 8.9. 
NVIDIA Images
+1

Implication: For 4090, the fastest path is FP16×INT4 GEMM/GEMV with register‑fused dequant, Marlin‑style tiling plus FlashAttention‑2, FP8 or INT4/INT8 KV cache, a paged KV allocator, and CUDA Graphs. Add an optional W4A8 (INT8 compute) path for mid‑to‑large batches to avoid INT4→FP16 up‑convert cost, guided by QServe’s algorithmic ideas, but implemented for Ada’s INT8 Tensor Cores. 
arXiv
+1

1) Target outcomes on RTX 4090

Perf goals vs well‑tuned vLLM AWQ on 4090, same model and prompts:

Decode (batch‑1 to 4): +1.4–2.0× throughput with lower p50 TTFT from chunked prefill + CUDA Graphs.

Mid‑batch (8–32): +1.3–1.8× by Marlin‑class kernels, activation‑quant path (W4A8‑INT8 compute), and fused QKV.

Memory: 3–4× weight reduction (AWQ), ~2× KV capacity with FP8 KV or ~4× with Q4 KV for long contexts. 
Vllm Docs
+1

These are engineering targets, not guarantees. They are consistent with published kernel gains (Marlin, INT‑FA) and with dequant‑overhead removal reported by QServe. 
arXiv
+2
arXiv
+2

2) System architecture (Rust + CUDA)

Workspace layout

awq-rs/
  crates/
    awq-core/        # model loader, AWQ metadata, packer, graph plans
    awq-kernels/     # CUDA C++ kernels + PTX, built via nvcc; Rust FFI via cudarc/cust
    awq-runtime/     # graph capture, scheduler, paged KV allocator, memory pools
    awq-server/      # CLI + HTTP server


Major components

Model ingest + prepack (awq-core):

Parse AWQ safetensors: qweight, scales, qzeros, group_size (typically 128).

Offline prepack to a Marlin‑compatible layout: reorder rows/columns to match WMMA tiles, interleave groups to avoid SMEM bank conflicts, and pack nibbles into u32 blocks with the same order map used by AWQ’s reference (do not hardcode; read from metadata to stay format‑compatible across tools). 
proceedings.mlsys.org
+1

Kernels (awq-kernels):

FP16×INT4 GEMM (prefill) with fused dequant in registers and cp.async double‑buffering for A (activations) and weight streaming; Tensor Cores output FP16/BF16.

FP16×INT4 GEMV (decode) persistent kernel with L2‑favoring loads and register blocking.

Optional W4A8 path: on 4090, INT8×INT8→INT32 MMA using activation quantization à la QoQ; rescale to FP16 at epilogue. 
arXiv

Fused QKV projection kernel: one X load → three matmuls, shared staging of X in SMEM; emit Q,K,V; fuse RoPE for K/Q.

Attention: FlashAttention‑2 implementation for Ada/Ampere. For decode, stream‑optimized tiles. 
arXiv

Fused sampling: softmax + top‑k/p + RNG on device for first‑token latency.

Runtime (awq-runtime):

Paged KV allocator (block‑based) with block tables per sequence.

Quantized KV: FP8 (E4M3/E5M2) toggle; pluggable Q4 KV (per‑head/channel‑wise scales) for longer contexts. 
Vllm Docs
+1

Continuous batching with chunked prefill. CUDA Graphs for steady decode loops.

Scheduler tuned for 4090: switch GEMV/GEMM by active batch; speculative decoding hook.

Why Rust? Safety for host code, zero‑cost abstractions, and clean FFI to CUDA. Critical math runs in CUDA C++.

3) The 4090 kernel blueprint
3.1 Weight format and prepack

AWQ stores packed 4‑bit weights + per‑group scales/zeros; group_size=128 is the common sweet spot. We keep group 128 to align with Marlin kernels and minimize metadata traffic. 
Hugging Face
+1

Nibble order: AWQ uses a specific interleave to minimize conflicts; the reference shows order_map=[0,2,4,6,1,3,5,7] for one path. Treat it as an input parameter read from the checkpoint rather than guessed. This avoids format drift across toolchains. 
GitHub

Offline transforms:

Reorder weight tiles to match MMA fragment shapes so that dequant emits FP16 fragments directly usable by WMMA.

Apply QUICK‑style interleaving to avoid shared‑memory bank conflicts during activation staging. 
arXiv

3.2 FP16×INT4 GEMM (prefill)

Threadblock tile: e.g., TB_M×TB_N×TB_K = 128×128×64, split into warp tiles matching WMMA m16n16k16.

Pipelines:

Stage X (activations) in SMEM via cp.async, double‑buffered with async barriers.

Weights stream from GMEM into registers; unpack 8 nibbles per u32, apply scale/zero in registers, assemble to FP16 pairs (f16x2) and immediately feed MMA.

Keep scales/zeros in L2‑friendly layout; vectorized 128‑bit loads.

Epilogue: combine with bias if present; store FP16 to global.

Why it wins: Removes extra GMEM traffic, hides GMEM latency with cp.async, and avoids SMEM bank conflicts. This is the Marlin playbook adapted to SM 8.9. 
GitHub
+1

3.3 FP16×INT4 GEMV (decode)

Persistent CTAs: 1 CTA per SM, loop over columns; maintain register‑resident partials and fetch next weight tile while computing the current; use large L2 and evict‑first cache hints for one‑shot weight loads.

Vectorized bit‑unpack + FMA in registers; accumulate to FP16/FP32 and downcast at store.

This addresses your batch‑1 path where AWQ should shine by bandwidth reduction.

3.4 Optional W4A8 on Ada (INT8 compute)

Motivation: At batch ≥16, dequant arithmetic can dominate; move to INT8 MMA reduces that overhead. QServe’s QoQ suggests progressive quantization and reordering to keep overhead <5%. We adopt the algorithmic idea but implement with Ada INT8 Tensor Cores. Output rescales to FP16. 
arXiv

3.5 Fused QKV + RoPE + KV‑update

Single kernel to compute Q,K,V from the same X load, apply RoPE inside, and write K,V to quantized KV cache.

Reduces 2–3 kernel launches and GMEM round‑trips per layer.

3.6 Attention

FlashAttention‑2 kernels for Ada. Tiled softmax in SMEM with blockwise matmuls; we expose head‑dim/sequence tiling knobs for decode vs prefill. 
arXiv

Hook for INT‑FlashAttention variants on Ampere/Ada when ready to push fully integer Q/K/V for longer contexts. 
arXiv

3.7 KV cache formats

FP8 KV (E4M3/E5M2) toggle for 2× capacity. It’s already used in vLLM; we’ll quantize per‑tensor or per‑head. 
Vllm Docs

Q4 KV mode for even larger contexts on 24 GB; ExLlamaV2 shows quality can remain near FP16. 
GitHub

4) Runtime: scheduling, memory, and graphs

Paged KV allocator: 16‑token blocks, block table per sequence, compaction when sequences finish. This mirrors PagedAttention’s low‑waste behavior; near‑zero fragmentation enables large effective batches. 
vLLM Blog

Continuous batching + chunked prefill: split long prompts and interleave with decode to cut TTFT; avoid the known first‑token lag in current AWQ stacks. 
GitHub

CUDA Graphs: capture the steady‑state decode micrograph (proj→attn→mlp→proj→sample) to reduce launch overhead and CPU scheduling stalls.

Heuristic batch switch: GEMV for B≤4, GEMM for B≥8, consider W4A8 for B≥16 on 4090.

5) Baseline vs competitors (why this will be faster)

vLLM AWQ: documented as under‑optimized; community issues show slowdowns vs FP16 at higher batch due to dequant cost. Our kernels remove most dequant overhead and maintain 4× weight bandwidth win at mid‑batch. 
Vllm Docs
+1

TensorRT‑LLM: strongest on Hopper via FP8/W4A8 plugins; on 4090 the FP8 path isn’t available. We target Ada with INT4/INT8 approaches that close the gap. 
NVIDIA GitHub

Marlin: we adopt the proven design (register‑fused dequant, striped partitioning, async streams) and add QKV fusion + quantized KV, which Marlin itself doesn’t ship end‑to‑end. 
GitHub

INT‑FlashAttention: integrate integer attention when stable to lift attention’s share; today we start with FA‑2. 
arXiv

6) Security, reliability, correctness

Rust host prevents common lifetime and concurrency bugs;

Kernels: out‑of‑bounds guards, alignment assertions; nsys and ncu profiles in CI to check achieved occupancy and bandwidth; golden‑output unit tests vs FP16 baselines; deterministic sampling option.

7) Implementation plan (deliverable set)
Phase A — Minimal viable fast path (4090, AWQ W4A16)

Loader + prepack to Marlin layout; expose order_map and group_size. 
GitHub

GEMV decode kernel (FP16×INT4) persistent + fused dequant.

GEMM prefill kernel (FP16×INT4) with cp.async double buffer. 
NVIDIA Developer

FlashAttention‑2 kernels for prefill/decode. 
arXiv

Paged KV allocator + FP8 KV option. 
Vllm Docs

CUDA Graphs for decode loop; chunked prefill.

Expected: Clear wins at batch‑1 to 16 vs vLLM AWQ.

Phase B — Mid‑batch booster

W4A8 on Ada using INT8 MMA; activation quant (QoQ‑style) with register‑level parallel dequant; compute‑aware weight reorder. 
arXiv

Fused QKV + RoPE + KV update.

Fused sampling kernel.

Phase C — Attention and long‑context

Q4 KV path and INT‑FlashAttention option for Ampere/Ada. 
arXiv

Phase D — Hopper/Blackwell path (not for 4090, but ready)

W4A8 FP8 Tensor Cores and wgmma/TMA ports; Upgrade when H100/H200 available. 
PyTorch

8) Rust + CUDA code skeletons (key pieces)

Note: concise, production‑oriented scaffolding. Replace ... with model‑specific shapes.

Cargo workspace

# awq-rs/Cargo.toml
[workspace]
members = ["crates/awq-core", "crates/awq-kernels", "crates/awq-runtime", "crates/awq-server"]
resolver = "2"


AWQ packer (safe Rust, parametric order map)

// crates/awq-core/src/pack.rs
pub fn pack_int4_to_u32(
    src: &[u8],            // un-packed 0..15 per nibble
    order: &[usize; 8],    // e.g., [0,2,4,6,1,3,5,7] from checkpoint
    out_u32: &mut [u32],
) {
    assert_eq!(src.len() % 8, 0);
    assert_eq!(out_u32.len() * 8, src.len());
    for (chunk_idx, chunk) in src.chunks_exact(8).enumerate() {
        let mut word: u32 = 0;
        for (i, &pos) in order.iter().enumerate() {
            let nib = (chunk[pos] & 0x0F) as u32;
            word |= nib << (i * 4);
        }
        out_u32[chunk_idx] = word;
    }
}


Kernel launch wrapper (Rust FFI; using cudarc)

// crates/awq-kernels/src/lib.rs
use cudarc::driver::*;
pub struct GemmInt4 {
    func: Function,
    module: Module,
}
impl GemmInt4 {
    pub fn new(dev: &CudaDevice, ptx: &str) -> Result<Self, DriverError> {
        let module = dev.load_ptx(ptx.into(), "awq_int4", &["gemm_int4_fused"])?;
        let func = module.get_func("gemm_int4_fused")?;
        Ok(Self { func, module })
    }
    pub unsafe fn launch(
        &self,
        cfg: LaunchConfig,
        a: &DevicePtr<f16>,     // activations
        w_q: &DevicePtr<u32>,   // packed int4 weights
        scales: &DevicePtr<f16>,
        zeros: &DevicePtr<f16>,
        c: &mut DevicePtr<f16>,
        m: i32, n: i32, k: i32, group: i32) -> Result<(), DriverError> {
        self.func.launch(cfg, (a, w_q, scales, zeros, c, m, n, k, group))
    }
}


FP16×INT4 fused‑dequant GEMM kernel (CUDA C++; simplified)

// crates/awq-kernels/kernels/gemm_int4.cu
extern "C" __global__
void gemm_int4_fused(const half* __restrict__ A,  // [M,K]
                     const uint32_t* __restrict__ Wq, // packed int4
                     const half* __restrict__ Scales, // per-group
                     const half* __restrict__ Zeros,  // per-group
                     half* __restrict__ C,           // [M,N]
                     int M, int N, int K, int group) {
    // Tile shapes
    // cp.async GMEM->SMEM for A tiles; double-buffered
    // Stream Wq from GMEM directly into registers

    // Pseudocode for inner loop:
    // 1) load A_tile via cp.async into smem_a[buf]
    // 2) for each K-slice:
    //      - prefetch next A_tile
    //      - load packed Wq (vectorized 128b)
    //      - unpack eight 4-bit weights from u32 into half2
    //      - dequant: (w - zero) * scale in registers
    //      - wmma::mma_sync(acc, a_frag, w_frag, acc)
    //      - swap buffers; continue
    // 3) store acc to C
}


Fused QKV projection kernel call (Rust side)

pub fn launch_qkv_fused(/* X, Wq_q, Wq_k, Wq_v, scales/zeros... */) { /* ... */ }


FlashAttention‑2 integration

Implement FA‑2 kernels (or bring in open FA‑2 and integrate with our KV layout). Keep decode‑optimized tiling. 
arXiv

Paged KV allocator (Rust)

// fixed-size blocks, e.g., 16 tokens per block
struct BlockTable { /* seq_id -> Vec<BlockId> */ }
struct KvAllocator {
  free_list: Vec<BlockId>,
  table: HashMap<SeqId, Vec<BlockId>>,
  // fp16/fp8/q4 storage backends
}

9) Tuning playbook for RTX 4090

cp.async everywhere for A tiles; use async barriers; size SMEM so both A‑buffers fit without spills. 
NVIDIA Developer

Weights in registers, not SMEM; prefetch next strip; use .cs cache hint for one‑time reads.

128‑bit loads/stores; align Wq/scales/zeros to 16 B.

Occupancy vs registers: target ~50–75% occupancy with high ILP; tune CTA tile sizes to keep Tensor Cores saturated.

L2 residency for scales/zeros; batch reordering to improve spatial locality.

CUDA Graphs for steady decode loop to cut CPU stalls.

Batch‑aware policy: B≤4 → GEMV; 8≤B≤32 → GEMM; B≥16 consider W4A8 (INT8 MMA) to dodge dequant FLOPs. 
arXiv

10) Benchmark and validation

Microbench linear layers (QKV, MLP) across batch 1,2,4,8,16,32; report achieved GB/s vs 1008 GB/s theoretical. 
NVIDIA Images

End‑to‑end: TTFT vs prompt length (512→8k), decode t/s at concurrency ladder (1→32).

Compare against: vLLM AWQ (same model), FP16 FA‑2 baseline, ExLlamaV2 (Q4 KV) where relevant. 
GitHub

11) Risks and mitigations

Quant accuracy (W4A8 path): follow QoQ calibration; provide fall‑back to W4A16 if quality dips. 
arXiv

Kernel coverage: start with group_size=128, per‑channel symmetric; add variants after baseline stability. 
arXiv

Attention edge cases: keep FA‑2 as default on 4090; gate INT‑FA behind flag until stable. 
arXiv

12) What you run today on 4090

Recommended defaults

Quant: AWQ W4A16, group 128; offline prepack to Marlin layout. 
arXiv

KV: fp8_e4m3 for 2× capacity; switch to Q4 KV if you target very long contexts. 
Vllm Docs
+1

Attention: FA‑2; enable chunked prefill; capture decode loop with CUDA Graphs. 
arXiv

Batch policy: dynamic GEMV/GEMM; try activation‑quant (W4A8 INT8 MMA) for B≥16 if quality holds. 
arXiv

13) Why this is better than “what’s out there” on 4090

Closes the AWQ dequant gap that makes vLLM sometimes slower than FP16 at high batch. Our kernels perform dequant inside the MMA pipeline and overlap everything with cp.async. 
GitHub

Mid‑batch scaling via Marlin‑style register blocking and activation quant path, which open stacks do not fully deploy on Ada. 
GitHub

KV memory pressure solved with FP8/Q4 KV, keeping throughput high at long context; Paged KV prevents fragmentation. 
Vllm Docs
+1

First‑token latency addressed with chunked prefill, fused sampling, and graphs; a known weak spot in AWQ stacks. 
GitHub

14) Executive summary

What: A Rust‑based AWQ 4‑bit inference engine optimized for RTX 4090. Core: FP16×INT4 kernels with fused dequant, FlashAttention‑2, paged + quantized KV, continuous batching, and CUDA Graphs.

Why now: AWQ accuracy is excellent, but current GPU implementations underperform at medium/large batches due to dequant costs and missed hardware features. vLLM flags AWQ as under‑optimized; Marlin/INT‑FA/QServe show the path to big wins. 
arXiv
+3
Vllm Docs
+3
GitHub
+3

How it’s faster on 4090:

Dequant in‑register fused into MMA, cp.async double‑buffered loads.

GEMV/GEMM dual path with persistent decode kernels.

Optional W4A8 (INT8 compute) mode for B≥16 inspired by QServe’s QoQ.

FP8 or Q4 KV cache to double or quadruple context capacity without tanking quality.

Paged KV + chunked prefill + CUDA Graphs for lower TTFT and higher steady‑state t/s. 
vLLM Blog
+1

Business impact:

Higher throughput per 4090 → lower $/token.

Longer contexts on 24 GB cards → fewer GPUs per workload.

Rust host + modular kernels → maintainable, auditable, safe.

Upgrade path: On H100/H200 add W4A8 FP8 compute and wgmma/TMA; on Blackwell, move to FP4 native paths. 
PyTorch

References (selected load‑bearing)

AWQ MLSys’24 and TinyChat; Marlin kernels; vLLM PagedAttention and AWQ status; INT‑FlashAttention; QServe; cp.async on Ampere/Ada; FP8/Q4 KV cache:

AWQ paper and repo. 
proceedings.mlsys.org
+1

Marlin paper and repo. 
arXiv
+1

vLLM PagedAttention and AWQ under‑optimized note. 
vLLM Blog
+1

INT‑FlashAttention. 
arXiv

QServe (W4A8KV4, QoQ, SmoothAttention). 
arXiv

Ampere/Ada cp.async. 
NVIDIA Developer

FP8 KV in vLLM; Q4 KV evidence. 
Vllm Docs
+1

4090 bandwidth and L2. 
NVIDIA Images

If you want, I’ll generate the repo scaffold, the prepack tool, and a first working GEMV/GEMM kernel + FA‑2 path next, ready to run on a 4090 with a Llama‑3‑8B‑AWQ checkpoint.