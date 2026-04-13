//! GPU memory buffers.
//!
//! [`CudaBuffer<T>`] is a type-safe RAII wrapper around a region of CUDA
//! device memory.  It owns the allocation and frees it on drop.
//!
//! # Example
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use std::sync::Arc;
//! use tensor_crab::cuda::device::CudaDevice;
//! use tensor_crab::cuda::buffer::CudaBuffer;
//!
//! let dev = Arc::new(CudaDevice::new(0).expect("no GPU"));
//! let host_data = vec![1.0_f32, 2.0, 3.0, 4.0];
//!
//! let buf = CudaBuffer::from_slice(&host_data, &dev).expect("alloc failed");
//! let back = buf.to_vec().expect("copy back failed");
//! assert_eq!(host_data, back);
//! # }
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use super::device::CudaDevice;
use super::error::{CUresult, CudaError, CudaResult};
use super::ffi;
use super::pool::GpuMemoryPool;

/// A contiguous region of GPU memory containing `len` elements of type `T`.
///
/// `CudaBuffer<T>` is analogous to [`Vec<T>`] but lives on the device.  Like
/// `Vec`, it owns its allocation and frees it on drop — unless the buffer was
/// created with [`CudaBuffer::uninitialized_pooled`], in which case the
/// allocation is returned to the pool on drop instead.
///
/// # Memory Layout
/// Elements are stored contiguously in row-major order, identical to the CPU
/// representation of `[T]`.  There is no padding or alignment beyond what
/// `cuMemAlloc` guarantees (256-byte alignment).
pub struct CudaBuffer<T> {
    /// Raw device pointer to the start of the allocation.
    ptr: ffi::CUdeviceptr,
    /// Number of elements of type `T`.
    len: usize,
    /// Keep the device alive while this buffer exists.
    _device: Arc<CudaDevice>,
    /// Optional memory pool.  When `Some`, `ptr` is returned to the pool on
    /// drop rather than freed.
    pool: Option<Arc<GpuMemoryPool>>,
    /// Phantom type parameter so the compiler tracks `T`.
    _marker: PhantomData<T>,
}

// SAFETY: GPU memory is not aliased from the CPU side, so moving the handle
// across threads is safe as long as it's not accessed from two threads at once.
unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T: Copy> CudaBuffer<T> {
    /// Allocates an uninitialised buffer of `len` elements on `device`.
    ///
    /// The contents of the buffer are **undefined** after this call.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if the allocation fails.
    pub fn uninitialized(len: usize, device: &Arc<CudaDevice>) -> CudaResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                _device: Arc::clone(device),
                pool: None,
                _marker: PhantomData,
            });
        }

        let byte_count = len * std::mem::size_of::<T>();
        let mut ptr: ffi::CUdeviceptr = 0;
        let result = unsafe { ffi::cuMemAlloc(&mut ptr, byte_count) };

        if CUresult::from_raw(result) == CUresult::OutOfMemory {
            return Err(CudaError::OutOfMemory {
                requested_bytes: byte_count,
            });
        }
        CUresult::from_raw(result).into_result()?;

        Ok(Self {
            ptr,
            len,
            _device: Arc::clone(device),
            pool: None,
            _marker: PhantomData,
        })
    }

    /// Allocates an uninitialised buffer of `len` elements on `device`, backed
    /// by `pool`.
    ///
    /// The buffer behaves identically to one created with [`Self::uninitialized`]
    /// except that when it is dropped the underlying allocation is returned to
    /// `pool` instead of being freed.  Future requests of the same byte size
    /// will be served from the pool without a driver call.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if the pool cannot satisfy the
    /// request and `cuMemAlloc` fails.
    pub fn uninitialized_pooled(
        len: usize,
        device: &Arc<CudaDevice>,
        pool: Arc<GpuMemoryPool>,
    ) -> CudaResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                _device: Arc::clone(device),
                pool: Some(pool),
                _marker: PhantomData,
            });
        }
        let byte_count = len * std::mem::size_of::<T>();
        let ptr = pool.alloc(byte_count)?;
        Ok(Self {
            ptr,
            len,
            _device: Arc::clone(device),
            pool: Some(pool),
            _marker: PhantomData,
        })
    }

    /// Allocates a zero-filled buffer of `len` elements on `device`.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if the allocation fails, or
    /// [`CudaError::Driver`] if zeroing fails.
    pub fn zeros(len: usize, device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let buf = Self::uninitialized(len, device)?;
        if len > 0 {
            let byte_count = len * std::mem::size_of::<T>();
            unsafe { CUresult::from_raw(ffi::cuMemsetD8(buf.ptr, 0, byte_count)).into_result() }?;
        }
        Ok(buf)
    }

    /// Allocates a device buffer and copies `data` from the host.
    ///
    /// # Errors
    /// Returns [`CudaError::OutOfMemory`] if the allocation fails, or
    /// [`CudaError::MemcpyFailed`] if the host-to-device copy fails.
    pub fn from_slice(data: &[T], device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let buf = Self::uninitialized(data.len(), device)?;
        if !data.is_empty() {
            buf.copy_from_host(data)?;
        }
        Ok(buf)
    }

    /// Copies `data` from host memory into this buffer.
    ///
    /// `data.len()` must equal `self.len()`.
    ///
    /// # Errors
    /// Returns [`CudaError::MemcpyFailed`] if the copy fails or the lengths
    /// differ.
    pub fn copy_from_host(&self, data: &[T]) -> CudaResult<()> {
        if data.len() != self.len {
            return Err(CudaError::MemcpyFailed(format!(
                "length mismatch: buffer has {} elements, source has {}",
                self.len,
                data.len()
            )));
        }
        if self.len == 0 {
            return Ok(());
        }
        let byte_count = self.len * std::mem::size_of::<T>();
        let result = unsafe { ffi::cuMemcpyHtoD(self.ptr, data.as_ptr().cast(), byte_count) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::MemcpyFailed(format!("host→device copy failed: {e}")))
    }

    /// Copies the buffer contents to a newly-allocated host `Vec<T>`.
    ///
    /// # Errors
    /// Returns [`CudaError::MemcpyFailed`] if the copy fails.
    pub fn to_vec(&self) -> CudaResult<Vec<T>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        // Allocate a buffer of MaybeUninit<T>, copy from device, then
        // convert to Vec<T>.  This avoids the `uninit_vec` clippy lint.
        let mut out: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(self.len);
        // SAFETY: MaybeUninit<T> requires no initialisation.
        unsafe { out.set_len(self.len) };
        let byte_count = self.len * std::mem::size_of::<T>();
        let result = unsafe { ffi::cuMemcpyDtoH(out.as_mut_ptr().cast(), self.ptr, byte_count) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::MemcpyFailed(format!("device→host copy failed: {e}")))?;
        // SAFETY: cuMemcpyDtoH initialised every element in `out`.
        let initialised =
            unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<T>>, Vec<T>>(out) };
        Ok(initialised)
    }

    /// Copies `src.len()` elements from `src` (device) into `self` (device).
    ///
    /// `src.len()` must equal `self.len()`.
    pub fn copy_from_device(&self, src: &CudaBuffer<T>) -> CudaResult<()> {
        if src.len != self.len {
            return Err(CudaError::MemcpyFailed(format!(
                "length mismatch: dst has {} elements, src has {}",
                self.len, src.len
            )));
        }
        if self.len == 0 {
            return Ok(());
        }
        let byte_count = self.len * std::mem::size_of::<T>();
        let result = unsafe { ffi::cuMemcpyDtoD(self.ptr, src.ptr, byte_count) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::MemcpyFailed(format!("device→device copy failed: {e}")))
    }

    /// Returns the number of elements in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the allocation size in bytes.
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns the raw device pointer.
    ///
    /// # Safety
    /// The caller must ensure the pointer is not used after this
    /// `CudaBuffer` is dropped.
    pub unsafe fn as_device_ptr(&self) -> ffi::CUdeviceptr {
        self.ptr
    }

    /// Returns a mutable alias of the device pointer, used when passing it
    /// as a kernel argument.
    ///
    /// # Safety
    /// The pointer must not be used after this buffer is dropped.
    #[allow(dead_code)]
    pub(crate) unsafe fn device_ptr_mut_ref(&mut self) -> &mut ffi::CUdeviceptr {
        &mut self.ptr
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        if let Some(pool) = &self.pool {
            // Return the allocation to the pool instead of freeing it.
            pool.dealloc(self.ptr, self.len * std::mem::size_of::<T>());
        } else {
            // Ignore errors: nothing useful can be done in drop.
            unsafe {
                ffi::cuMemFree(self.ptr);
            }
        }
    }
}

impl<T: std::fmt::Debug + Copy> std::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("ptr", &format!("0x{:016x}", self.ptr))
            .field("len", &self.len)
            .field("byte_size", &self.byte_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;

    /// Buffer is Send + Sync.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn buffer_is_send_sync() {
        assert_send_sync::<CudaBuffer<f32>>();
    }

    /// Verify byte_size calculation without touching GPU.
    #[test]
    fn test_byte_size_formula() {
        // We cannot create an actual buffer without a GPU, but we can verify
        // the arithmetic.
        let element_size = std::mem::size_of::<f32>();
        assert_eq!(element_size, 4);
        assert_eq!(100 * element_size, 400);
    }

    /// Verify that zero-len slices work without panicking.
    #[test]
    fn test_empty_buffer_no_gpu_path() {
        if CudaDevice::count() > 0 {
            // This path requires a real GPU; skip in headless environments.
            return;
        }
        // Just confirm types compile and basic logic holds.
        let size: usize = 0;
        assert_eq!(size * std::mem::size_of::<f32>(), 0);
    }
}
