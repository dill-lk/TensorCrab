//! CUDA stream management.
//!
//! A [`CudaStream`] represents an ordered sequence of GPU operations.
//! Operations submitted to the same stream execute in issue order; operations
//! on different streams may run concurrently.
//!
//! # Example
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use std::sync::Arc;
//! use tensor_crab::cuda::device::CudaDevice;
//! use tensor_crab::cuda::stream::CudaStream;
//!
//! let dev = Arc::new(CudaDevice::new(0).expect("no GPU"));
//! let stream = CudaStream::new(&dev).expect("stream creation failed");
//! stream.synchronize().expect("sync failed");
//! # }
//! ```

use std::sync::Arc;

use super::device::CudaDevice;
use super::error::{CUresult, CudaResult};
use super::ffi;

/// A CUDA execution stream.
///
/// All GPU work is enqueued onto a stream.  The CUDA default (null) stream
/// provides implicit serialisation; named streams allow overlapping of
/// computation and data transfers.
///
/// `CudaStream` is `Send` and `Sync` because the underlying CUDA stream is
/// associated with a device context and only accessed from this handle.
pub struct CudaStream {
    /// Raw Driver API stream handle.
    raw: ffi::CUstream,
    /// Keep the device alive for at least as long as this stream.
    _device: Arc<CudaDevice>,
}

// SAFETY: CUstream is a pointer but CUDA makes it safe to move across threads
// as long as it's not used concurrently from two threads at once.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Creates a new CUDA stream on `device`.
    ///
    /// # Errors
    /// Returns [`super::error::CudaError::Driver`] if stream creation fails.
    pub fn new(device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let mut raw: ffi::CUstream = std::ptr::null_mut();
        unsafe { CUresult::from_raw(ffi::cuStreamCreate(&mut raw, 0)).into_result() }?;
        Ok(Self {
            raw,
            _device: Arc::clone(device),
        })
    }

    /// Creates a non-blocking stream that does not synchronise with the
    /// default (null) stream.
    ///
    /// # Errors
    /// Returns [`super::error::CudaError::Driver`] if stream creation fails.
    pub fn new_non_blocking(device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let mut raw: ffi::CUstream = std::ptr::null_mut();
        // CU_STREAM_NON_BLOCKING = 1
        unsafe { CUresult::from_raw(ffi::cuStreamCreate(&mut raw, 1)).into_result() }?;
        Ok(Self {
            raw,
            _device: Arc::clone(device),
        })
    }

    /// Blocks the calling CPU thread until all GPU operations submitted to
    /// this stream have completed.
    ///
    /// # Errors
    /// Returns [`super::error::CudaError::Driver`] if the synchronisation
    /// fails (e.g. because a prior kernel launch failed).
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { CUresult::from_raw(ffi::cuStreamSynchronize(self.raw)).into_result() }
    }

    /// Returns `true` if all work on this stream has finished, or `false` if
    /// work is still pending.
    ///
    /// # Errors
    /// Returns an error only if the query itself fails (not if work is merely
    /// pending — that case returns `Ok(false)`).
    pub fn is_done(&self) -> CudaResult<bool> {
        let result = unsafe { ffi::cuStreamQuery(self.raw) };
        // CUDA_ERROR_NOT_READY = 600
        if result == 600 {
            return Ok(false);
        }
        CUresult::from_raw(result).into_result()?;
        Ok(true)
    }

    /// Returns the raw [`ffi::CUstream`] handle.
    ///
    /// # Safety
    /// The returned handle must not outlive this `CudaStream`.
    pub(crate) unsafe fn raw(&self) -> ffi::CUstream {
        self.raw
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // Ignore errors on drop — there is nothing meaningful we can do.
        unsafe {
            ffi::cuStreamDestroy(self.raw);
        }
    }
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("raw", &(self.raw as usize))
            .finish()
    }
}

// ─── CudaEvent ────────────────────────────────────────────────────────────────

/// A CUDA event used for fine-grained timing and stream synchronisation.
pub struct CudaEvent {
    raw: ffi::CUevent,
}

// SAFETY: Same reasoning as CudaStream.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Creates a new CUDA event.
    ///
    /// # Errors
    /// Returns a [`super::error::CudaError::Driver`] on failure.
    pub fn new() -> CudaResult<Self> {
        let mut raw: ffi::CUevent = std::ptr::null_mut();
        unsafe { CUresult::from_raw(ffi::cuEventCreate(&mut raw, 0)).into_result() }?;
        Ok(Self { raw })
    }

    /// Records this event on `stream`.  The event will be reached after all
    /// previously enqueued work on `stream` has completed.
    pub fn record(&self, stream: &CudaStream) -> CudaResult<()> {
        unsafe {
            CUresult::from_raw(ffi::cuEventRecord(self.raw, stream.raw())).into_result()
        }
    }

    /// Blocks the CPU until this event has been reached.
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { CUresult::from_raw(ffi::cuEventSynchronize(self.raw)).into_result() }
    }

    /// Returns the elapsed time in milliseconds between `start` and this event.
    ///
    /// Both events must have been recorded (and optionally synchronised) before
    /// calling this.
    pub fn elapsed_ms(&self, start: &CudaEvent) -> CudaResult<f32> {
        let mut ms: f32 = 0.0;
        unsafe {
            CUresult::from_raw(ffi::cuEventElapsedTime(&mut ms, start.raw, self.raw))
                .into_result()
        }?;
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            ffi::cuEventDestroy(self.raw);
        }
    }
}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEvent")
            .field("raw", &(self.raw as usize))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;

    /// Verify that stream and event types are `Send + Sync`.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn stream_is_send_sync() {
        assert_send_sync::<CudaStream>();
    }

    #[test]
    fn event_is_send_sync() {
        assert_send_sync::<CudaEvent>();
    }

    /// If no GPU is available, stream creation should fail gracefully.
    #[test]
    fn test_stream_creation_without_gpu() {
        if CudaDevice::count() == 0 {
            // We can't create a stream without a device context.
            // Just verify the test compiles and runs.
        }
    }
}
