//! CUDA device discovery and management.
//!
//! [`CudaDevice`] is the primary entry point for all GPU operations.  It wraps
//! a CUDA context, ensuring that the context is properly destroyed when the
//! last `Arc<CudaDevice>` is dropped.
//!
//! # Example
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use tensor_crab::cuda::device::CudaDevice;
//!
//! // Initialise CUDA and open device 0.
//! let dev = CudaDevice::new(0).expect("no CUDA GPU found");
//! println!("GPU: {} ({} MiB)", dev.name(), dev.total_memory_mib());
//! # }
//! ```

use std::ffi::CStr;
use std::os::raw::c_char;

use super::error::{CUresult, CudaError, CudaResult};
use super::ffi;

// ─── DeviceProperties ─────────────────────────────────────────────────────────

/// Properties of a CUDA-capable device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Human-readable device name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Major compute capability (e.g. `8` for Ampere).
    pub compute_major: i32,
    /// Minor compute capability (e.g. `9` for RTX 4090).
    pub compute_minor: i32,
    /// Number of streaming multiprocessors.
    pub multiprocessor_count: i32,
    /// Maximum number of threads per block.
    pub max_threads_per_block: i32,
    /// Warp size (typically 32).
    pub warp_size: i32,
    /// Peak global memory bandwidth clock rate in kHz.
    pub memory_clock_rate_khz: i32,
    /// Global memory bus width in bits.
    pub memory_bus_width: i32,
    /// L2 cache size in bytes.
    pub l2_cache_size: i32,
}

impl DeviceProperties {
    /// Returns `"SM X.Y"` — the compute capability as a human-readable string.
    pub fn compute_capability(&self) -> String {
        format!("SM {}.{}", self.compute_major, self.compute_minor)
    }

    /// Returns total memory in mebibytes.
    pub fn total_memory_mib(&self) -> usize {
        self.total_memory / (1024 * 1024)
    }
}

// ─── CudaDevice ───────────────────────────────────────────────────────────────

/// A handle to a CUDA-capable GPU device.
///
/// `CudaDevice` encapsulates both the device ordinal and the CUDA Driver API
/// context created for that device.  The context is destroyed when this struct
/// is dropped.
///
/// All [`super::buffer::CudaBuffer`] and [`super::stream::CudaStream`] objects
/// must live at most as long as the `CudaDevice` they were created from.
pub struct CudaDevice {
    /// Device ordinal (0-based index).
    ordinal: i32,
    /// The underlying Driver API context.
    ctx: ffi::CUcontext,
    /// Cached device properties.
    props: DeviceProperties,
}

// SAFETY: CudaDevice owns the context and the context is only accessed through
// &self (single thread at a time, enforced by CUDA's implicit serialisation of
// operations on a context).
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Initialises the CUDA Driver API (if not already initialised) and opens
    /// the device with ordinal `ordinal`.
    ///
    /// # Errors
    /// - [`CudaError::NoDevice`] — no CUDA-capable GPU was found.
    /// - [`CudaError::InvalidDevice`] — `ordinal` is out of range.
    /// - [`CudaError::Driver`] — any other Driver API error.
    pub fn new(ordinal: i32) -> CudaResult<Self> {
        // Initialise the driver (no-op if already done).
        unsafe { CUresult::from_raw(ffi::cuInit(0)).into_result() }?;

        // Validate the ordinal.
        let mut count: i32 = 0;
        unsafe { CUresult::from_raw(ffi::cuDeviceGetCount(&mut count)).into_result() }?;

        if count == 0 {
            return Err(CudaError::NoDevice);
        }
        if ordinal < 0 || ordinal >= count {
            return Err(CudaError::InvalidDevice {
                requested: ordinal,
                available: count,
            });
        }

        // Obtain the device handle.
        let mut device: ffi::CUdevice = 0;
        unsafe { CUresult::from_raw(ffi::cuDeviceGet(&mut device, ordinal)).into_result() }?;

        // Query properties.
        let props = Self::query_properties(device)?;

        // Create a context on the device.
        let mut ctx: ffi::CUcontext = std::ptr::null_mut();
        unsafe { CUresult::from_raw(ffi::cuCtxCreate(&mut ctx, 0, device)).into_result() }?;

        Ok(Self {
            ordinal,
            ctx,
            props,
        })
    }

    /// Returns the number of CUDA-capable devices on this machine.
    ///
    /// Returns `0` if no CUDA-capable device is found, or if the CUDA driver
    /// is not installed.
    pub fn count() -> i32 {
        // Try to init; on machines with no CUDA this will fail gracefully.
        let init_result = unsafe { ffi::cuInit(0) };
        if CUresult::from_raw(init_result) != CUresult::Success {
            return 0;
        }
        let mut n: i32 = 0;
        let result = unsafe { ffi::cuDeviceGetCount(&mut n) };
        if CUresult::from_raw(result) == CUresult::Success {
            n
        } else {
            0
        }
    }

    /// Returns `true` if at least one CUDA-capable device is available.
    pub fn is_available() -> bool {
        Self::count() > 0
    }

    /// Returns the ordinal (0-based index) of this device.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Returns a reference to the cached [`DeviceProperties`] for this device.
    pub fn properties(&self) -> &DeviceProperties {
        &self.props
    }

    /// Returns the device name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub fn name(&self) -> &str {
        &self.props.name
    }

    /// Returns the total device memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.props.total_memory
    }

    /// Returns the total device memory in mebibytes.
    pub fn total_memory_mib(&self) -> usize {
        self.props.total_memory_mib()
    }

    /// Returns the compute capability as a string (e.g. `"SM 8.9"`).
    pub fn compute_capability(&self) -> String {
        self.props.compute_capability()
    }

    /// Returns the raw Driver API context handle.
    ///
    /// # Safety
    /// Callers must ensure the context remains alive for the duration of any
    /// Driver API call that uses it.
    #[allow(dead_code)]
    pub(crate) unsafe fn raw_ctx(&self) -> ffi::CUcontext {
        self.ctx
    }

    /// Synchronises the context, blocking until all prior GPU work is done.
    ///
    /// # Errors
    /// Returns a [`CudaError::Driver`] if the synchronisation fails.
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { CUresult::from_raw(ffi::cuCtxSynchronize()).into_result() }
    }

    /// Checks whether this device can directly access the memory of `other`.
    pub fn can_access_peer(&self, other: &CudaDevice) -> CudaResult<bool> {
        let mut can: i32 = 0;
        unsafe {
            CUresult::from_raw(ffi::cuDeviceCanAccessPeer(
                &mut can,
                self.ordinal,
                other.ordinal,
            ))
            .into_result()
        }?;
        Ok(can != 0)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Queries and returns device properties for `dev`.
    fn query_properties(dev: ffi::CUdevice) -> CudaResult<DeviceProperties> {
        // Device name.
        let mut name_buf = [0i8; 256];
        unsafe {
            CUresult::from_raw(ffi::cuDeviceGetName(name_buf.as_mut_ptr(), 256, dev)).into_result()
        }?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .into_owned();

        // Total memory.
        let mut total_memory: usize = 0;
        unsafe { CUresult::from_raw(ffi::cuDeviceTotalMem(&mut total_memory, dev)).into_result() }?;

        // Attributes.
        let compute_major = query_attr(dev, ffi::CUdevice_attribute::ComputeCapabilityMajor)?;
        let compute_minor = query_attr(dev, ffi::CUdevice_attribute::ComputeCapabilityMinor)?;
        let multiprocessor_count = query_attr(dev, ffi::CUdevice_attribute::MultiprocessorCount)?;
        let max_threads_per_block = query_attr(dev, ffi::CUdevice_attribute::MaxThreadsPerBlock)?;
        let warp_size = query_attr(dev, ffi::CUdevice_attribute::WarpSize)?;
        let memory_clock_rate_khz = query_attr(dev, ffi::CUdevice_attribute::MemoryClockRate)?;
        let memory_bus_width = query_attr(dev, ffi::CUdevice_attribute::GlobalMemoryBusWidth)?;
        let l2_cache_size = query_attr(dev, ffi::CUdevice_attribute::L2CacheSize)?;

        Ok(DeviceProperties {
            name,
            total_memory,
            compute_major,
            compute_minor,
            multiprocessor_count,
            max_threads_per_block,
            warp_size,
            memory_clock_rate_khz,
            memory_bus_width,
            l2_cache_size,
        })
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // Destroy the context.  Ignore errors here; there is nothing
        // meaningful we can do during `drop`.
        unsafe {
            ffi::cuCtxDestroy(self.ctx);
        }
    }
}

impl std::fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudaDevice({}: {} [{}] {:.0} MiB)",
            self.ordinal,
            self.props.name,
            self.props.compute_capability(),
            self.props.total_memory_mib()
        )
    }
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("ordinal", &self.ordinal)
            .field("name", &self.props.name)
            .field("compute", &self.props.compute_capability())
            .field("memory_mib", &self.props.total_memory_mib())
            .finish()
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Queries a single integer device attribute.
fn query_attr(dev: ffi::CUdevice, attr: ffi::CUdevice_attribute) -> CudaResult<i32> {
    let mut val: i32 = 0;
    unsafe {
        CUresult::from_raw(ffi::cuDeviceGetAttribute(&mut val, attr as u32, dev)).into_result()
    }?;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// If no GPU is present these tests should pass trivially.
    #[test]
    fn test_device_count_non_negative() {
        let n = CudaDevice::count();
        assert!(n >= 0, "device count should be non-negative");
    }

    #[test]
    fn test_is_available_matches_count() {
        assert_eq!(CudaDevice::is_available(), CudaDevice::count() > 0);
    }

    /// Only run device-creation tests if a GPU is actually present.
    #[test]
    fn test_new_invalid_ordinal_without_gpu() {
        if CudaDevice::count() == 0 {
            // No device available — opening device 0 should fail gracefully.
            let result = CudaDevice::new(0);
            assert!(result.is_err(), "should fail with no GPU present");
        }
    }

    #[test]
    fn test_device_properties_compute_capability_format() {
        let props = DeviceProperties {
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
            compute_major: 8,
            compute_minor: 9,
            multiprocessor_count: 128,
            max_threads_per_block: 1024,
            warp_size: 32,
            memory_clock_rate_khz: 10_000,
            memory_bus_width: 256,
            l2_cache_size: 6 * 1024 * 1024,
        };
        assert_eq!(props.compute_capability(), "SM 8.9");
        assert_eq!(props.total_memory_mib(), 8192);
    }
}
