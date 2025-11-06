use anyhow::Result;
use rumma_core::QuantizedLinear;

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

/// Compute backend for kernel operations
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaBackend),
}

impl Backend {
    /// Create the best available backend (CUDA if available, otherwise CPU)
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            match cuda::CudaBackend::new() {
                Ok(backend) => {
                    eprintln!("🚀 Using CUDA GPU acceleration");
                    return Backend::Cuda(backend);
                }
                Err(e) => {
                    eprintln!("⚠️  CUDA initialization failed: {}", e);
                    eprintln!("   Falling back to CPU");
                }
            }
        }

        eprintln!("🖥️  Using CPU (multi-threaded)");
        Backend::Cpu
    }

    /// Create a CPU-only backend
    pub fn cpu() -> Self {
        Backend::Cpu
    }

    #[cfg(feature = "cuda")]
    /// Create a CUDA backend (returns error if CUDA is not available)
    pub fn cuda() -> Result<Self> {
        Ok(Backend::Cuda(cuda::CudaBackend::new()?))
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::new()
    }
}

pub fn gemm_prefill(layer: &QuantizedLinear, input: &[f32], batch: usize) -> Result<Vec<f32>> {
    gemm_prefill_with_backend(&Backend::new(), layer, input, batch)
}

pub fn gemm_prefill_with_backend(
    backend: &Backend,
    layer: &QuantizedLinear,
    input: &[f32],
    batch: usize,
) -> Result<Vec<f32>> {
    match backend {
        Backend::Cpu => cpu::gemm_prefill(layer, input, batch),
        #[cfg(feature = "cuda")]
        Backend::Cuda(cuda_backend) => cuda_backend.gemm_prefill(layer, input, batch),
    }
}

pub fn gemv_decode(layer: &QuantizedLinear, input: &[f32]) -> Result<Vec<f32>> {
    gemv_decode_with_backend(&Backend::new(), layer, input)
}

pub fn gemv_decode_with_backend(
    backend: &Backend,
    layer: &QuantizedLinear,
    input: &[f32],
) -> Result<Vec<f32>> {
    match backend {
        Backend::Cpu => cpu::gemv_decode(layer, input),
        #[cfg(feature = "cuda")]
        Backend::Cuda(cuda_backend) => cuda_backend.gemv_decode(layer, input),
    }
}

pub fn relu_inplace(tensor: &mut [f32]) {
    cpu::relu_inplace(tensor)
}

pub fn softmax(logits: &mut [f32]) {
    cpu::softmax(logits)
}
