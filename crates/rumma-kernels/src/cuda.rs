use anyhow::{anyhow, Result};
use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use rumma_core::QuantizedLinear;
use std::sync::Arc;

pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        // Initialize CUDA device (use device 0 by default)
        let device = CudaDevice::new(0)
            .map_err(|e| anyhow!("Failed to initialize CUDA device: {:?}", e))?;

        // Initialize cuBLAS for matrix operations
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| anyhow!("Failed to initialize cuBLAS: {:?}", e))?;

        Ok(Self { device, blas })
    }

    /// GEMM prefill: batch matrix multiplication
    /// Performs: output = layer * input
    /// Where input is [batch, cols] and output is [batch, rows]
    pub fn gemm_prefill(
        &self,
        layer: &QuantizedLinear,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        if input.len() != layer.cols() * batch {
            return Err(anyhow!(
                "prefill input has {} elements, expected {}",
                input.len(),
                layer.cols() * batch
            ));
        }

        let rows = layer.rows();
        let cols = layer.cols();

        // Dequantize the layer weights to f32 on CPU
        // TODO: Optimize by doing dequantization on GPU
        let weights = self.dequantize_layer(layer)?;

        // Upload data to GPU
        let d_input = self.device.htod_sync_copy(input)
            .map_err(|e| anyhow!("Failed to copy input to GPU: {:?}", e))?;
        let d_weights = self.device.htod_sync_copy(&weights)
            .map_err(|e| anyhow!("Failed to copy weights to GPU: {:?}", e))?;
        let mut d_output = self.device.alloc_zeros::<f32>(rows * batch)
            .map_err(|e| anyhow!("Failed to allocate output on GPU: {:?}", e))?;

        // Perform GEMM: output = weights * input^T
        // cuBLAS uses column-major, so we need to think carefully about dimensions
        // We want: output[batch, rows] = weights[rows, cols] @ input[batch, cols]^T
        // In column-major: C = alpha * A * B + beta * C
        // Where A is [rows, cols], B is [cols, batch], C is [rows, batch]
        unsafe {
            self.blas.gemm(
                // m: number of rows of A and C
                rows as i32,
                // n: number of columns of B and C
                batch as i32,
                // k: number of columns of A and rows of B
                cols as i32,
                // alpha
                1.0,
                // A: weights [rows, cols]
                &d_weights,
                // lda: leading dimension of A
                rows as i32,
                // B: input [cols, batch] (transposed)
                &d_input,
                // ldb: leading dimension of B
                cols as i32,
                // beta
                0.0,
                // C: output [rows, batch]
                &mut d_output,
                // ldc: leading dimension of C
                rows as i32,
            ).map_err(|e| anyhow!("cuBLAS GEMM failed: {:?}", e))?;
        }

        // Copy result back to CPU
        let mut output = vec![0f32; rows * batch];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| anyhow!("Failed to copy output from GPU: {:?}", e))?;

        Ok(output)
    }

    /// GEMV decode: matrix-vector multiplication
    /// Performs: output = layer * input
    /// Where input is [cols] and output is [rows]
    pub fn gemv_decode(&self, layer: &QuantizedLinear, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != layer.cols() {
            return Err(anyhow!(
                "decode input has {} elements, expected {}",
                input.len(),
                layer.cols()
            ));
        }

        let rows = layer.rows();
        let cols = layer.cols();

        // Dequantize the layer weights to f32 on CPU
        let weights = self.dequantize_layer(layer)?;

        // Upload data to GPU
        let d_input = self.device.htod_sync_copy(input)
            .map_err(|e| anyhow!("Failed to copy input to GPU: {:?}", e))?;
        let d_weights = self.device.htod_sync_copy(&weights)
            .map_err(|e| anyhow!("Failed to copy weights to GPU: {:?}", e))?;
        let mut d_output = self.device.alloc_zeros::<f32>(rows)
            .map_err(|e| anyhow!("Failed to allocate output on GPU: {:?}", e))?;

        // Perform GEMV: output = weights * input
        // We can use GEMM with batch=1 for simplicity
        unsafe {
            self.blas.gemm(
                rows as i32,
                1,
                cols as i32,
                1.0,
                &d_weights,
                rows as i32,
                &d_input,
                cols as i32,
                0.0,
                &mut d_output,
                rows as i32,
            ).map_err(|e| anyhow!("cuBLAS GEMV failed: {:?}", e))?;
        }

        // Copy result back to CPU
        let mut output = vec![0f32; rows];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| anyhow!("Failed to copy output from GPU: {:?}", e))?;

        Ok(output)
    }

    /// Dequantize layer weights from 4-bit to f32
    /// TODO: Move this to GPU for better performance
    fn dequantize_layer(&self, layer: &QuantizedLinear) -> Result<Vec<f32>> {
        let rows = layer.rows();
        let cols = layer.cols();
        let mut weights = vec![0f32; rows * cols];

        // Dequantize using the layer's built-in method
        for row in 0..rows {
            for col in 0..cols {
                // Extract weight at [row, col]
                // Create a unit vector with 1.0 at position col
                let mut unit = vec![0f32; cols];
                unit[col] = 1.0;

                // Compute dot product to get the weight
                let val = layer.dot_row(row, &unit);
                weights[row + col * rows] = val; // column-major layout for cuBLAS
            }
        }

        Ok(weights)
    }
}
