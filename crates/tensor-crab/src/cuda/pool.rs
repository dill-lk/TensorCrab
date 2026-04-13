//! GPU memory pool to reduce `cuMemAlloc` / `cuMemFree` overhead.
//!
//! Allocations returned to the pool are cached by byte-size and reused on the
//! next request of the same size, skipping the expensive driver round-trip.
//!
//! # Usage
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use std::sync::Arc;
//! use tensor_crab::cuda::device::CudaDevice;
//! use tensor_crab::cuda::buffer::CudaBuffer;
//! use tensor_crab::cuda::pool::GpuMemoryPool;
//!
//! let dev = Arc::new(CudaDevice::new(0).expect("no GPU"));
//! let pool = GpuMemoryPool::new(Arc::clone(&dev));
//!
//! // Allocate a pooled buffer.
//! let buf = CudaBuffer::<f32>::uninitialized_pooled(1024, &dev, Arc::clone(&pool))
//!     .expect("alloc failed");
//! // When `buf` is dropped the allocation returns to the pool.
//! drop(buf);
//! assert_eq!(pool.cached_count(), 1);
//!
//! // Next allocation of the same size is served from the cache (no cuMemAlloc).
//! let buf2 = CudaBuffer::<f32>::uninitialized_pooled(1024, &dev, Arc::clone(&pool))
//!     .expect("alloc failed");
//! assert_eq!(pool.cached_count(), 0);
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::device::CudaDevice;
use super::error::{CUresult, CudaResult};
use super::ffi;

// ─── Slot ────────────────────────────────────────────────────────────────────

/// A cached device allocation.
struct Slot {
    ptr: ffi::CUdeviceptr,
}

// ─── GpuMemoryPool ───────────────────────────────────────────────────────────

/// Thread-safe GPU memory pool for a single device.
///
/// Allocations returned to the pool are cached by their exact byte size.
/// The pool never automatically shrinks — call [`GpuMemoryPool::purge`] to
/// release all cached allocations back to the driver.
///
/// # Thread Safety
/// All methods are safe to call concurrently from multiple threads.
pub struct GpuMemoryPool {
    /// The device this pool is associated with.
    _device: Arc<CudaDevice>,
    /// Map from byte-size → list of free slots.
    slots: Mutex<HashMap<usize, Vec<Slot>>>,
}

impl GpuMemoryPool {
    /// Creates a new memory pool associated with `device`.
    pub fn new(device: Arc<CudaDevice>) -> Arc<Self> {
        Arc::new(Self {
            _device: device,
            slots: Mutex::new(HashMap::new()),
        })
    }

    /// Allocates `bytes` bytes from the pool.
    ///
    /// If a cached slot of the same size is available it is returned
    /// immediately (no driver call).  Otherwise a fresh allocation is made
    /// with [`ffi::cuMemAlloc`].
    ///
    /// # Errors
    /// Returns a CUDA error if `cuMemAlloc` fails.
    pub(crate) fn alloc(&self, bytes: usize) -> CudaResult<ffi::CUdeviceptr> {
        if bytes == 0 {
            return Ok(0);
        }
        {
            let mut map = self.slots.lock().expect("GpuMemoryPool mutex poisoned");
            if let Some(slots) = map.get_mut(&bytes) {
                if let Some(slot) = slots.pop() {
                    return Ok(slot.ptr);
                }
            }
        }
        // Nothing cached — allocate fresh.
        let mut ptr: ffi::CUdeviceptr = 0;
        let result = unsafe { ffi::cuMemAlloc(&mut ptr, bytes) };
        CUresult::from_raw(result).into_result()?;
        Ok(ptr)
    }

    /// Returns `ptr` (of size `bytes`) to the pool instead of freeing it.
    ///
    /// A null pointer or zero-byte region is silently ignored.
    pub(crate) fn dealloc(&self, ptr: ffi::CUdeviceptr, bytes: usize) {
        if ptr == 0 || bytes == 0 {
            return;
        }
        let mut map = self.slots.lock().expect("GpuMemoryPool mutex poisoned");
        map.entry(bytes).or_default().push(Slot { ptr });
    }

    /// Releases all cached allocations back to the GPU driver.
    ///
    /// After this call [`GpuMemoryPool::cached_count`] returns `0`.
    pub fn purge(&self) {
        let mut map = self.slots.lock().expect("GpuMemoryPool mutex poisoned");
        for slots in map.values() {
            for slot in slots {
                // SAFETY: ptr was obtained from cuMemAlloc and is not in use.
                unsafe {
                    ffi::cuMemFree(slot.ptr);
                }
            }
        }
        map.clear();
    }

    /// Returns the number of cached (free, pooled) allocations.
    pub fn cached_count(&self) -> usize {
        let map = self.slots.lock().expect("GpuMemoryPool mutex poisoned");
        map.values().map(|v| v.len()).sum()
    }

    /// Returns the total number of bytes currently held in the pool.
    pub fn cached_bytes(&self) -> usize {
        let map = self.slots.lock().expect("GpuMemoryPool mutex poisoned");
        map.iter().map(|(&bytes, v)| bytes * v.len()).sum()
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        self.purge();
    }
}

impl std::fmt::Debug for GpuMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryPool")
            .field("cached_count", &self.cached_count())
            .field("cached_bytes", &self.cached_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;

    #[test]
    fn test_pool_empty_on_creation() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let pool = GpuMemoryPool::new(Arc::clone(&dev));
        assert_eq!(pool.cached_count(), 0);
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn test_pool_dealloc_cached_count() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let pool = GpuMemoryPool::new(Arc::clone(&dev));

        // Allocate from pool (goes to driver).
        let ptr = pool.alloc(256).expect("alloc failed");
        assert_eq!(pool.cached_count(), 0);

        // Return to pool.
        pool.dealloc(ptr, 256);
        assert_eq!(pool.cached_count(), 1);
        assert_eq!(pool.cached_bytes(), 256);

        // Re-allocate — served from cache.
        let ptr2 = pool.alloc(256).expect("re-alloc failed");
        assert_eq!(pool.cached_count(), 0);
        assert_ne!(ptr2, 0);

        // Purge to clean up.
        pool.dealloc(ptr2, 256);
        pool.purge();
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn test_pool_different_sizes_not_mixed() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let pool = GpuMemoryPool::new(Arc::clone(&dev));

        let p1 = pool.alloc(128).unwrap();
        let p2 = pool.alloc(256).unwrap();
        pool.dealloc(p1, 128);
        pool.dealloc(p2, 256);
        assert_eq!(pool.cached_count(), 2);

        // Request a 128-byte slot — should only reduce the 128 bucket.
        let _ = pool.alloc(128).unwrap();
        assert_eq!(pool.cached_count(), 1);
    }

    #[test]
    fn test_pool_zero_size_is_noop() {
        if CudaDevice::count() == 0 {
            return;
        }
        let dev = Arc::new(CudaDevice::new(0).unwrap());
        let pool = GpuMemoryPool::new(Arc::clone(&dev));
        let ptr = pool.alloc(0).unwrap();
        assert_eq!(ptr, 0);
        pool.dealloc(0, 0);
        assert_eq!(pool.cached_count(), 0);
    }
}
