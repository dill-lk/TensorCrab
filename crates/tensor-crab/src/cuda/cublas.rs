//! cuBLAS handle and GEMM wrapper.
//!
//! Provides a safe RAII wrapper around a cuBLAS context and a single-precision
//! general matrix-matrix multiplication (`sgemm`) helper for row-major matrices.

use std::sync::Arc;

use super::device::CudaDevice;
use super::error::{CudaError, CudaResult};
use super::ffi;
use super::stream::CudaStream;

/// RAII wrapper around a cuBLAS library context.
///
/// Creates a cuBLAS handle on construction and destroys it on drop.  A handle
/// is associated with a specific CUDA device.
pub struct CublasHandle {
    raw: ffi::cublasHandle_t,
    _device: Arc<CudaDevice>,
}

// SAFETY: cublasHandle_t is safe to move across threads (cuBLAS docs §3.1).
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Creates a new cuBLAS handle for `device`.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] if `cublasCreate` fails.
    pub fn new(device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let mut raw: ffi::cublasHandle_t = std::ptr::null_mut();
        let status = unsafe { ffi::cublasCreate_v2(&mut raw) };
        if status != 0 {
            return Err(CudaError::Internal(format!(
                "cublasCreate failed: status {status}"
            )));
        }
        Ok(Self {
            raw,
            _device: Arc::clone(device),
        })
    }

    /// Binds `stream` to this handle so all subsequent cuBLAS calls are
    /// submitted asynchronously on that stream.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] if `cublasSetStream` fails.
    pub fn set_stream(&self, stream: &CudaStream) -> CudaResult<()> {
        // SAFETY: stream pointer is valid for the duration of this call.
        let raw_stream = unsafe { stream.raw() };
        let status = unsafe { ffi::cublasSetStream_v2(self.raw, raw_stream) };
        if status != 0 {
            return Err(CudaError::Internal(format!(
                "cublasSetStream failed: status {status}"
            )));
        }
        Ok(())
    }

    /// Computes `C = alpha * A * B + beta * C` for row-major matrices.
    ///
    /// - `A` is `[m × k]`, `B` is `[k × n]`, `C` is `[m × n]` (all row-major).
    ///
    /// cuBLAS uses column-major convention internally.  To compute the
    /// row-major product `C = A * B` we exploit the identity
    /// `(A * B)^T = B^T * A^T` and call `cublasSgemm` with swapped arguments:
    ///
    /// ```text
    /// cublasSgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
    /// ```
    ///
    /// This yields the correct row-major result without any transpositions.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] if `cublasSgemm` fails.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        a: ffi::CUdeviceptr,
        b: ffi::CUdeviceptr,
        c: ffi::CUdeviceptr,
    ) -> CudaResult<()> {
        use std::os::raw::c_int;
        let status = unsafe {
            ffi::cublasSgemm_v2(
                self.raw,
                ffi::cublasOperation_t::CUBLAS_OP_N,
                ffi::cublasOperation_t::CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                b,
                n as c_int,
                a,
                k as c_int,
                &beta,
                c,
                n as c_int,
            )
        };
        if status != 0 {
            return Err(CudaError::Internal(format!(
                "cublasSgemm failed: status {status}"
            )));
        }
        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Ignore errors: nothing useful can be done in drop.
            unsafe {
                ffi::cublasDestroy_v2(self.raw);
            }
        }
    }
}

impl std::fmt::Debug for CublasHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CublasHandle")
            .field("raw", &(self.raw as usize))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;

    #[test]
    fn test_cublas_handle_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CublasHandle>();
    }

    #[test]
    fn test_cublas_create_requires_gpu() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).expect("GPU present"));
        let handle = CublasHandle::new(&dev);
        assert!(handle.is_ok());
    }
}
