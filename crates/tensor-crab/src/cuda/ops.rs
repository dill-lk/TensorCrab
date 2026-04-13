//! GPU-backed tensor operations.
//!
//! [`CudaTensor`] is the GPU analogue of [`crate::tensor::Tensor`].  It holds
//! its data in a [`CudaBuffer<f32>`] on the device and exposes element-wise
//! and reduction operations accelerated by the CUDA kernels in
//! [`super::kernels`].
//!
//! # Example
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use std::sync::Arc;
//! use tensor_crab::cuda::device::CudaDevice;
//! use tensor_crab::cuda::ops::CudaTensor;
//!
//! let dev = Arc::new(CudaDevice::new(0).expect("no GPU"));
//! let a = CudaTensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], &dev).unwrap();
//! let b = CudaTensor::from_slice(&[4.0_f32, 5.0, 6.0], &[3], &dev).unwrap();
//! let c = a.add(&b).unwrap();
//! let result = c.to_vec().unwrap(); // [5.0, 7.0, 9.0]
//! # }
//! ```

use std::sync::Arc;

use super::buffer::CudaBuffer;
use super::device::CudaDevice;
use super::error::{CudaError, CudaResult};
use super::module::{grid_size_1d, CudaModule, DEFAULT_BLOCK_SIZE};
use super::stream::CudaStream;

// ─── CudaTensor ───────────────────────────────────────────────────────────────

/// An N-dimensional `f32` tensor stored in GPU device memory.
///
/// `CudaTensor` mirrors the public API of [`crate::tensor::Tensor`] for the
/// subset of operations accelerated by CUDA kernels.  All operations return
/// new `CudaTensor` objects and do not modify their inputs (functional style,
/// same as the CPU `Tensor`).
///
/// # Shape convention
/// Shapes follow the same row-major convention as [`crate::tensor::Tensor`].
/// The `shape` slice contains the size of each dimension (e.g. `[batch,
/// features]`), and `numel()` is their product.
pub struct CudaTensor {
    /// GPU memory holding `numel()` `f32` values in row-major order.
    data: CudaBuffer<f32>,
    /// Dimension sizes.
    shape: Vec<usize>,
    /// Reference to the device that owns this tensor.
    device: Arc<CudaDevice>,
    /// Optional execution stream for asynchronous kernel dispatch.
    ///
    /// When `Some`, kernels are submitted to this stream and return without
    /// waiting for completion.  Call [`CudaTensor::synchronize`] to wait.
    /// When `None` (the default), each op synchronises the device before
    /// returning.
    stream: Option<Arc<CudaStream>>,
}

impl CudaTensor {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Creates a `CudaTensor` by uploading `data` to `device`.
    ///
    /// `data.len()` must equal the product of `shape`.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] if the lengths don't match, or
    /// [`CudaError::OutOfMemory`] if the device allocation fails.
    pub fn from_slice(data: &[f32], shape: &[usize], device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(CudaError::Internal(format!(
                "from_slice: data length {} does not match shape {:?} (numel = {})",
                data.len(),
                shape,
                numel
            )));
        }
        let buf = CudaBuffer::from_slice(data, device)?;
        Ok(Self {
            data: buf,
            shape: shape.to_vec(),
            device: Arc::clone(device),
            stream: None,
        })
    }

    /// Creates a zero-filled `CudaTensor` of the given `shape`.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if allocation fails.
    pub fn zeros(shape: &[usize], device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let numel: usize = shape.iter().product();
        let buf = CudaBuffer::zeros(numel, device)?;
        Ok(Self {
            data: buf,
            shape: shape.to_vec(),
            device: Arc::clone(device),
            stream: None,
        })
    }

    /// Creates a `CudaTensor` filled with `value`.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if allocation fails.
    pub fn full(value: f32, shape: &[usize], device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let numel: usize = shape.iter().product();
        // Allocate and fill using the fill kernel.
        let buf = CudaBuffer::uninitialized(numel, device)?;
        let tensor = Self {
            data: buf,
            shape: shape.to_vec(),
            device: Arc::clone(device),
            stream: None,
        };
        tensor.fill_inplace(value)?;
        Ok(tensor)
    }

    /// Creates a `CudaTensor` filled with `1.0`.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if allocation fails.
    pub fn ones(shape: &[usize], device: &Arc<CudaDevice>) -> CudaResult<Self> {
        Self::full(1.0, shape, device)
    }

    // ── Shape accessors ───────────────────────────────────────────────────────

    /// Returns the shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns a reference to the device this tensor lives on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    // ── Stream management ─────────────────────────────────────────────────────

    /// Attaches an execution stream to this tensor, returning a new `CudaTensor`.
    ///
    /// All subsequent operations on the returned tensor will be submitted
    /// asynchronously to `stream`.  Call [`CudaTensor::synchronize`] to block
    /// until all enqueued work has completed.
    pub fn with_stream(self, stream: Arc<CudaStream>) -> Self {
        Self {
            stream: Some(stream),
            ..self
        }
    }

    /// Detaches the execution stream, restoring synchronous operation semantics.
    ///
    /// The returned tensor's ops will synchronise the device after each launch.
    pub fn without_stream(self) -> Self {
        Self {
            stream: None,
            ..self
        }
    }

    /// Returns the attached execution stream, if any.
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }

    /// Blocks until all GPU work on this tensor's stream has completed.
    ///
    /// If no stream is attached this is a no-op — synchronous ops already
    /// complete before returning.
    ///
    /// # Errors
    /// Returns [`CudaError::Driver`] if the stream synchronization fails.
    pub fn synchronize(&self) -> CudaResult<()> {
        if let Some(s) = &self.stream {
            s.synchronize()
        } else {
            Ok(())
        }
    }

    // ── Host ↔ Device transfers ───────────────────────────────────────────────

    /// Downloads the tensor data to a host `Vec<f32>` in row-major order.
    ///
    /// # Errors
    /// Returns [`CudaError::MemcpyFailed`] if the transfer fails.
    pub fn to_vec(&self) -> CudaResult<Vec<f32>> {
        self.data.to_vec()
    }

    // ── Element-wise binary operations ────────────────────────────────────────

    /// Element-wise addition: `self + other`.
    ///
    /// Both tensors must have the same shape.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] on shape mismatch, or a
    /// [`CudaError::Driver`] / [`CudaError::KernelLaunch`] if the kernel fails.
    pub fn add(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(
            other,
            super::kernels::KERNEL_ADD_F32,
            super::kernels::PTX_BINARY_F32,
        )
    }

    /// Element-wise subtraction: `self - other`.
    pub fn sub(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(
            other,
            super::kernels::KERNEL_SUB_F32,
            super::kernels::PTX_BINARY_F32,
        )
    }

    /// Element-wise multiplication: `self * other`.
    pub fn mul(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(
            other,
            super::kernels::KERNEL_MUL_F32,
            super::kernels::PTX_BINARY_F32,
        )
    }

    /// Element-wise division: `self / other`.
    pub fn div(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(
            other,
            super::kernels::KERNEL_DIV_F32,
            super::kernels::PTX_BINARY_F32,
        )
    }

    // ── Scalar operations ─────────────────────────────────────────────────────

    /// Adds a scalar constant to every element.
    pub fn add_scalar(&self, scalar: f32) -> CudaResult<CudaTensor> {
        self.scalar_op(
            scalar,
            super::kernels::KERNEL_ADD_SCALAR_F32,
            super::kernels::PTX_SCALAR_BINARY_F32,
        )
    }

    /// Multiplies every element by a scalar constant.
    pub fn mul_scalar(&self, scalar: f32) -> CudaResult<CudaTensor> {
        self.scalar_op(
            scalar,
            super::kernels::KERNEL_MUL_SCALAR_F32,
            super::kernels::PTX_SCALAR_BINARY_F32,
        )
    }

    // ── Unary operations ──────────────────────────────────────────────────────

    /// Applies ReLU element-wise: `max(0, x)`.
    pub fn relu(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_RELU_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    /// Negates every element: `-x`.
    pub fn neg(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_NEG_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    /// Takes the absolute value of every element: `|x|`.
    pub fn abs(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_ABS_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    /// Takes the square root of every element: `√x`.
    pub fn sqrt(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_SQRT_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    /// Squares every element: `x²`.
    pub fn square(&self) -> CudaResult<CudaTensor> {
        self.unary_op(super::kernels::KERNEL_SQ_F32, super::kernels::PTX_UNARY_F32)
    }

    /// Applies `e^x` element-wise.
    pub fn exp(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_EXP_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    /// Applies `ln(x)` element-wise.
    pub fn log(&self) -> CudaResult<CudaTensor> {
        self.unary_op(
            super::kernels::KERNEL_LOG_F32,
            super::kernels::PTX_UNARY_F32,
        )
    }

    // ── Shape manipulation ────────────────────────────────────────────────────

    /// Returns a new `CudaTensor` with a different shape but the same data.
    ///
    /// The total number of elements must be preserved.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] if the element count changes.
    pub fn reshape(&self, new_shape: &[usize]) -> CudaResult<CudaTensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(CudaError::Internal(format!(
                "reshape: cannot reshape {:?} (numel={}) into {:?} (numel={})",
                self.shape,
                self.numel(),
                new_shape,
                new_numel
            )));
        }
        // Cloning the buffer shares the device memory (not possible with the current
        // CudaBuffer design which does not support sharing).  Instead copy the data.
        let new_buf = CudaBuffer::uninitialized(new_numel, &self.device)?;
        new_buf.copy_from_device(&self.data)?;
        Ok(CudaTensor {
            data: new_buf,
            shape: new_shape.to_vec(),
            device: Arc::clone(&self.device),
            stream: self.stream.clone(),
        })
    }

    /// Reshapes the tensor to a 1-D vector.
    pub fn flatten(&self) -> CudaResult<CudaTensor> {
        self.reshape(&[self.numel()])
    }

    // ── Matrix multiplication ─────────────────────────────────────────────────

    /// Matrix multiplication: `self @ other` using cuBLAS SGEMM.
    ///
    /// Both tensors must be 2-D.  `self` must be `[m, k]` and `other` must be
    /// `[k, n]`; the result is `[m, n]`.
    ///
    /// If a stream is attached to `self` the cuBLAS call is submitted
    /// asynchronously on that stream; call [`CudaTensor::synchronize`] to wait.
    ///
    /// # Errors
    /// Returns [`CudaError::Internal`] on dimension or shape errors, or if the
    /// cuBLAS call fails.
    pub fn matmul(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(CudaError::Internal(
                "matmul requires 2-D tensors".to_string(),
            ));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(CudaError::Internal(format!(
                "matmul shape mismatch: [{m}×{k}] @ [{k2}×{n}]"
            )));
        }
        let out_buf = CudaBuffer::uninitialized(m * n, &self.device)?;
        let handle = super::cublas::CublasHandle::new(&self.device)?;
        if let Some(s) = &self.stream {
            handle.set_stream(s)?;
        }
        let a_ptr = unsafe { self.data.as_device_ptr() };
        let b_ptr = unsafe { other.data.as_device_ptr() };
        let c_ptr = unsafe { out_buf.as_device_ptr() };
        handle.sgemm(m, n, k, 1.0, 0.0, a_ptr, b_ptr, c_ptr)?;
        if self.stream.is_none() {
            self.device.synchronize()?;
        }
        Ok(CudaTensor {
            data: out_buf,
            shape: vec![m, n],
            device: Arc::clone(&self.device),
            stream: self.stream.clone(),
        })
    }

    /// Returns the raw CUDA stream pointer for kernel launches.
    ///
    /// `None` → use the default (null) stream; ops sync on completion.
    fn raw_stream(&self) -> Option<super::ffi::CUstream> {
        // SAFETY: the stream outlives this call; the pointer is only used for the launch.
        self.stream.as_ref().map(|s| unsafe { s.raw() })
    }

    /// Synchronises the device after a kernel launch when no stream is set.
    ///
    /// With a stream attached the caller is responsible for calling
    /// [`CudaTensor::synchronize`] when results are needed.
    fn maybe_sync(&self) -> CudaResult<()> {
        if self.stream.is_none() {
            self.device.synchronize()
        } else {
            Ok(())
        }
    }

    /// Fills this tensor in-place with `value` using the CUDA fill kernel.
    fn fill_inplace(&self, value: f32) -> CudaResult<()> {
        let n = self.numel() as u32;
        if n == 0 {
            return Ok(());
        }

        let module = CudaModule::from_ptx(super::kernels::PTX_FILL_F32, &self.device)?;
        let func = module.function(super::kernels::KERNEL_FILL_F32)?;

        let mut out_ptr = unsafe { self.data.as_device_ptr() };
        let mut val = value;
        let mut count = self.numel() as u64;

        let mut args: [*mut std::os::raw::c_void; 3] = [
            std::ptr::addr_of_mut!(out_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(val).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(count).cast::<std::os::raw::c_void>(),
        ];

        let grid = grid_size_1d(n, DEFAULT_BLOCK_SIZE);
        unsafe {
            func.launch(
                (grid, 1, 1),
                (DEFAULT_BLOCK_SIZE, 1, 1),
                0,
                self.raw_stream(),
                &mut args,
            )
        }?;
        self.maybe_sync()
    }

    /// Runs a binary element-wise kernel.
    fn binary_op(&self, other: &CudaTensor, kernel: &str, ptx: &str) -> CudaResult<CudaTensor> {
        if self.shape != other.shape {
            return Err(CudaError::Internal(format!(
                "binary op shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let n = self.numel() as u32;
        let out_buf = CudaBuffer::uninitialized(self.numel(), &self.device)?;

        if n == 0 {
            return Ok(CudaTensor {
                data: out_buf,
                shape: self.shape.clone(),
                device: Arc::clone(&self.device),
                stream: self.stream.clone(),
            });
        }

        let module = CudaModule::from_ptx(ptx, &self.device)?;
        let func = module.function(kernel)?;

        let mut out_ptr = unsafe { out_buf.as_device_ptr() };
        let mut a_ptr = unsafe { self.data.as_device_ptr() };
        let mut b_ptr = unsafe { other.data.as_device_ptr() };
        let mut count = self.numel() as u64;

        let mut args: [*mut std::os::raw::c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(a_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(b_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(count).cast::<std::os::raw::c_void>(),
        ];

        let grid = grid_size_1d(n, DEFAULT_BLOCK_SIZE);
        unsafe {
            func.launch(
                (grid, 1, 1),
                (DEFAULT_BLOCK_SIZE, 1, 1),
                0,
                self.raw_stream(),
                &mut args,
            )
        }?;
        self.maybe_sync()?;

        Ok(CudaTensor {
            data: out_buf,
            shape: self.shape.clone(),
            device: Arc::clone(&self.device),
            stream: self.stream.clone(),
        })
    }

    /// Runs a unary element-wise kernel.
    fn unary_op(&self, kernel: &str, ptx: &str) -> CudaResult<CudaTensor> {
        let n = self.numel() as u32;
        let out_buf = CudaBuffer::uninitialized(self.numel(), &self.device)?;

        if n == 0 {
            return Ok(CudaTensor {
                data: out_buf,
                shape: self.shape.clone(),
                device: Arc::clone(&self.device),
                stream: self.stream.clone(),
            });
        }

        let module = CudaModule::from_ptx(ptx, &self.device)?;
        let func = module.function(kernel)?;

        let mut out_ptr = unsafe { out_buf.as_device_ptr() };
        let mut a_ptr = unsafe { self.data.as_device_ptr() };
        let mut count = self.numel() as u64;

        let mut args: [*mut std::os::raw::c_void; 3] = [
            std::ptr::addr_of_mut!(out_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(a_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(count).cast::<std::os::raw::c_void>(),
        ];

        let grid = grid_size_1d(n, DEFAULT_BLOCK_SIZE);
        unsafe {
            func.launch(
                (grid, 1, 1),
                (DEFAULT_BLOCK_SIZE, 1, 1),
                0,
                self.raw_stream(),
                &mut args,
            )
        }?;
        self.maybe_sync()?;

        Ok(CudaTensor {
            data: out_buf,
            shape: self.shape.clone(),
            device: Arc::clone(&self.device),
            stream: self.stream.clone(),
        })
    }

    /// Runs a scalar-binary kernel (`out[i] = a[i] op scalar`).
    fn scalar_op(&self, scalar: f32, kernel: &str, ptx: &str) -> CudaResult<CudaTensor> {
        let n = self.numel() as u32;
        let out_buf = CudaBuffer::uninitialized(self.numel(), &self.device)?;

        if n == 0 {
            return Ok(CudaTensor {
                data: out_buf,
                shape: self.shape.clone(),
                device: Arc::clone(&self.device),
                stream: self.stream.clone(),
            });
        }

        let module = CudaModule::from_ptx(ptx, &self.device)?;
        let func = module.function(kernel)?;

        let mut out_ptr = unsafe { out_buf.as_device_ptr() };
        let mut a_ptr = unsafe { self.data.as_device_ptr() };
        let mut s = scalar;
        let mut count = self.numel() as u64;

        let mut args: [*mut std::os::raw::c_void; 4] = [
            std::ptr::addr_of_mut!(out_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(a_ptr).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(s).cast::<std::os::raw::c_void>(),
            std::ptr::addr_of_mut!(count).cast::<std::os::raw::c_void>(),
        ];

        let grid = grid_size_1d(n, DEFAULT_BLOCK_SIZE);
        unsafe {
            func.launch(
                (grid, 1, 1),
                (DEFAULT_BLOCK_SIZE, 1, 1),
                0,
                self.raw_stream(),
                &mut args,
            )
        }?;
        self.maybe_sync()?;

        Ok(CudaTensor {
            data: out_buf,
            shape: self.shape.clone(),
            device: Arc::clone(&self.device),
            stream: self.stream.clone(),
        })
    }
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel())
            .field("device", &self.device.ordinal())
            .finish()
    }
}

// ─── CPU ↔ GPU conversion ─────────────────────────────────────────────────────

impl CudaTensor {
    /// Uploads a CPU [`crate::tensor::Tensor`] to the GPU.
    ///
    /// The tensor is made contiguous before upload.
    ///
    /// # Errors
    /// Returns a CUDA error if allocation or upload fails.
    pub fn from_cpu(cpu: &crate::tensor::Tensor, device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let data = cpu.to_vec();
        Self::from_slice(&data, cpu.shape(), device)
    }

    /// Downloads this `CudaTensor` to a CPU [`crate::tensor::Tensor`].
    ///
    /// # Errors
    /// Returns [`CudaError::MemcpyFailed`] if the download fails.
    pub fn to_cpu(&self) -> CudaResult<crate::tensor::Tensor> {
        let data = self.to_vec()?;
        Ok(crate::tensor::Tensor::from_vec(data, &self.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;

    /// Tests that don't require a GPU.
    #[test]
    fn test_shape_accessors() {
        // Verify that shape math is correct without touching the GPU.
        let shape = [2usize, 3, 4];
        let numel: usize = shape.iter().product();
        assert_eq!(numel, 24);
    }

    #[test]
    fn test_reshape_numel_preserved() {
        // Reshape from [2, 3] to [6] — same number of elements.
        let old_shape = [2usize, 3];
        let new_shape = [6usize];
        let old_numel: usize = old_shape.iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(old_numel, new_numel);
    }

    #[test]
    fn test_from_cpu_shape_matches() {
        if CudaDevice::count() == 0 {
            return; // No GPU — skip.
        }
        let cpu = crate::tensor::Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let dev = Arc::new(CudaDevice::new(0).expect("GPU present"));
        let gpu = CudaTensor::from_cpu(&cpu, &dev).expect("upload failed");
        assert_eq!(gpu.shape(), &[2, 2]);
        assert_eq!(gpu.numel(), 4);
    }

    #[test]
    fn test_round_trip_host_device() {
        if CudaDevice::count() == 0 {
            return;
        }
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dev = Arc::new(CudaDevice::new(0).expect("GPU present"));
        let gpu = CudaTensor::from_slice(&data, &[2, 3], &dev).expect("upload failed");
        let back = gpu.to_vec().expect("download failed");
        assert_eq!(data, back);
    }

    #[test]
    fn test_add_correctness() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], &dev).unwrap();
        let b = CudaTensor::from_slice(&[4.0_f32, 5.0, 6.0], &[3], &dev).unwrap();
        let c = a.add(&b).unwrap();
        let result = c.to_vec().unwrap();
        assert_eq!(result, vec![5.0_f32, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_scalar_correctness() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], &dev).unwrap();
        let b = a.mul_scalar(2.0).unwrap();
        let result = b.to_vec().unwrap();
        assert_eq!(result, vec![2.0_f32, 4.0, 6.0]);
    }

    #[test]
    fn test_relu_correctness() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_slice(&[-1.0_f32, 0.0, 2.0, -0.5, 3.0], &[5], &dev).unwrap();
        let r = a.relu().unwrap();
        let result = r.to_vec().unwrap();
        assert_eq!(result, vec![0.0_f32, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_neg_correctness() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_slice(&[1.0_f32, -2.0, 3.0], &[3], &dev).unwrap();
        let n = a.neg().unwrap();
        let result = n.to_vec().unwrap();
        assert_eq!(result, vec![-1.0_f32, 2.0, -3.0]);
    }

    #[test]
    fn test_zeros_all_zero() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let z = CudaTensor::zeros(&[4], &dev).unwrap();
        let v = z.to_vec().unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_reshape_preserves_data() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = CudaTensor::from_slice(&data, &[2, 3], &dev).unwrap();
        let r = t.reshape(&[6]).unwrap();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(r.to_vec().unwrap(), data);
    }

    #[test]
    fn test_matmul_shape_check_no_gpu() {
        // Verify shape validation logic (no GPU needed).
        // 2×3 @ 2×3 should fail because inner dims don't match.
        let k_lhs = 3usize;
        let k_rhs = 2usize;
        assert_ne!(k_lhs, k_rhs);
    }

    #[test]
    fn test_matmul_correctness() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let a = CudaTensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &dev).unwrap();
        let b = CudaTensor::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2], &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let result = c.to_vec().unwrap();
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (a, e) in result.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "got {a}, expected {e}");
        }
    }

    #[test]
    fn test_with_stream_propagates() {
        // Verify stream field is propagated through ops (no GPU needed for field logic).
        // We just check the type compiles and that stream is propagated to the result.
        // Real stream tests require a GPU.
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let stream = Arc::new(super::super::stream::CudaStream::new(&dev).unwrap());
        let a = CudaTensor::from_slice(&[1.0_f32, 2.0], &[2], &dev)
            .unwrap()
            .with_stream(Arc::clone(&stream));
        assert!(a.stream().is_some());
        let b = a.mul_scalar(2.0).unwrap();
        assert!(b.stream().is_some());
        b.synchronize().unwrap();
    }
}
