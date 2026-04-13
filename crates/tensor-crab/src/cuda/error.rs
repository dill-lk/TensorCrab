//! CUDA error types for TensorCrab.
//!
//! All fallible CUDA operations return `CudaResult<T>` which is an alias for
//! `Result<T, CudaError>`.

use std::fmt;

/// Convenience alias for a `Result` wrapping a [`CudaError`].
pub type CudaResult<T> = Result<T, CudaError>;

/// The CUDA Driver API result code.
///
/// Only the variants most commonly encountered in practice are listed
/// explicitly; the catch-all [`CUresult::Unknown`] covers the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CUresult {
    /// Operation completed successfully.
    Success = 0,
    /// One or more of the parameters passed to the API call are not within
    /// an acceptable range of values.
    InvalidValue = 1,
    /// The API call failed because it was unable to allocate enough memory
    /// to perform the requested operation.
    OutOfMemory = 2,
    /// The CUDA driver has not been initialised with `cuInit()`.
    NotInitialized = 3,
    /// The CUDA driver is in the process of shutting down.
    Deinitialized = 4,
    /// No CUDA-capable devices were detected by the installed CUDA driver.
    NoDevice = 100,
    /// The device ordinal supplied by the user does not correspond to a valid
    /// CUDA device.
    InvalidDevice = 101,
    /// The context creation failed.
    InvalidContext = 201,
    /// Out-of-memory on device during context creation.
    ContextAlreadyCurrent = 202,
    /// The PTX JIT compiler library was not found.
    NoBinaryForGpu = 209,
    /// A required kernel image is missing.
    FileNotFound = 301,
    /// The shared library could not be found.
    SharedObjectSymbolNotFound = 302,
    /// The PTX compilation failed.
    JitCompilerNotFound = 303,
    /// The device kernel launch exceeds resource limits.
    LaunchOutOfResources = 701,
    /// The GPU program failed to execute.
    LaunchFailed = 719,
    /// An unknown error code.
    Unknown = 999,
}

impl CUresult {
    /// Converts a raw `u32` driver API return value into a `CUresult`.
    pub fn from_raw(code: u32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            201 => Self::InvalidContext,
            202 => Self::ContextAlreadyCurrent,
            209 => Self::NoBinaryForGpu,
            301 => Self::FileNotFound,
            302 => Self::SharedObjectSymbolNotFound,
            303 => Self::JitCompilerNotFound,
            701 => Self::LaunchOutOfResources,
            719 => Self::LaunchFailed,
            _ => Self::Unknown,
        }
    }

    /// Returns `Ok(())` if this result is `Success`, otherwise wraps it in
    /// `Err(CudaError::Driver(...))`.
    pub fn into_result(self) -> CudaResult<()> {
        if self == Self::Success {
            Ok(())
        } else {
            Err(CudaError::Driver(self))
        }
    }
}

impl fmt::Display for CUresult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Success => "CUDA_SUCCESS",
            Self::InvalidValue => "CUDA_ERROR_INVALID_VALUE",
            Self::OutOfMemory => "CUDA_ERROR_OUT_OF_MEMORY",
            Self::NotInitialized => "CUDA_ERROR_NOT_INITIALIZED",
            Self::Deinitialized => "CUDA_ERROR_DEINITIALIZED",
            Self::NoDevice => "CUDA_ERROR_NO_DEVICE",
            Self::InvalidDevice => "CUDA_ERROR_INVALID_DEVICE",
            Self::InvalidContext => "CUDA_ERROR_INVALID_CONTEXT",
            Self::ContextAlreadyCurrent => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            Self::NoBinaryForGpu => "CUDA_ERROR_NO_BINARY_FOR_GPU",
            Self::FileNotFound => "CUDA_ERROR_FILE_NOT_FOUND",
            Self::SharedObjectSymbolNotFound => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            Self::JitCompilerNotFound => "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
            Self::LaunchOutOfResources => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            Self::LaunchFailed => "CUDA_ERROR_LAUNCH_FAILED",
            Self::Unknown => "CUDA_ERROR_UNKNOWN",
        };
        write!(f, "{msg}")
    }
}

/// Runtime error codes returned by the CUDA Runtime API (`cudart`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u32)]
pub enum CudartError {
    /// Operation completed successfully.
    Success = 0,
    /// The device function being invoked (indirectly) via a kernel launch is
    /// not supported on this device.
    InvalidDeviceFunction = 8,
    /// The CUDA runtime has not been initialised with `cudaInit()`.
    InitializationError = 3,
    /// Insufficient memory for the requested operation.
    MemoryAllocation = 2,
    /// One or more of the passed parameters is incorrect.
    InvalidValue = 11,
    /// An error occurred in the CUDA driver during kernel launch.
    LaunchFailure = 4,
    /// An unknown runtime error.
    Unknown = 999,
}

impl CudartError {
    /// Converts a raw `u32` runtime API return value into a `CudartError`.
    pub fn from_raw(code: u32) -> Self {
        match code {
            0 => Self::Success,
            2 => Self::MemoryAllocation,
            3 => Self::InitializationError,
            4 => Self::LaunchFailure,
            8 => Self::InvalidDeviceFunction,
            11 => Self::InvalidValue,
            _ => Self::Unknown,
        }
    }

    /// Returns `Ok(())` if success, otherwise `Err(CudaError::Runtime(...))`.
    pub fn into_result(self) -> CudaResult<()> {
        if self == Self::Success {
            Ok(())
        } else {
            Err(CudaError::Runtime(self))
        }
    }
}

impl fmt::Display for CudartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Success => "cudaSuccess",
            Self::InvalidDeviceFunction => "cudaErrorInvalidDeviceFunction",
            Self::InitializationError => "cudaErrorInitializationError",
            Self::MemoryAllocation => "cudaErrorMemoryAllocation",
            Self::InvalidValue => "cudaErrorInvalidValue",
            Self::LaunchFailure => "cudaErrorLaunchFailure",
            Self::Unknown => "cudaErrorUnknown",
        };
        write!(f, "{msg}")
    }
}

/// All errors that can arise from CUDA operations in TensorCrab.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// An error from the CUDA Driver API (`libcuda`).
    Driver(CUresult),

    /// An error from the CUDA Runtime API (`libcudart`).
    Runtime(CudartError),

    /// The requested device index does not exist.
    ///
    /// Valid devices are in the range `0 ..= device_count - 1`.
    InvalidDevice {
        /// The ordinal that was requested.
        requested: i32,
        /// The number of CUDA-capable devices available.
        available: i32,
    },

    /// No CUDA-capable GPU was found on this machine.
    NoDevice,

    /// PTX module JIT-compilation or loading failed.
    ModuleLoad(String),

    /// A kernel function name was not found in a loaded module.
    FunctionNotFound(String),

    /// A GPU memory allocation failed (device out of memory).
    OutOfMemory {
        /// Number of bytes that could not be allocated.
        requested_bytes: usize,
    },

    /// A host↔device memory copy operation failed.
    MemcpyFailed(String),

    /// A CUDA kernel launch failed.
    KernelLaunch(String),

    /// An unexpected internal error.
    Internal(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Runtime(e) => write!(f, "CUDA runtime error: {e}"),
            Self::InvalidDevice {
                requested,
                available,
            } => write!(
                f,
                "invalid CUDA device {requested}: only {available} device(s) available"
            ),
            Self::NoDevice => write!(f, "no CUDA-capable GPU found"),
            Self::ModuleLoad(msg) => write!(f, "failed to load PTX module: {msg}"),
            Self::FunctionNotFound(name) => {
                write!(f, "kernel function '{name}' not found in module")
            }
            Self::OutOfMemory { requested_bytes } => {
                write!(
                    f,
                    "GPU out of memory: failed to allocate {} bytes",
                    requested_bytes
                )
            }
            Self::MemcpyFailed(msg) => write!(f, "CUDA memcpy failed: {msg}"),
            Self::KernelLaunch(msg) => write!(f, "kernel launch failed: {msg}"),
            Self::Internal(msg) => write!(f, "internal CUDA error: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

// ─── Convenience macros ───────────────────────────────────────────────────────

/// Checks a raw CUDA Driver API return code.
///
/// Converts the `u32` to a [`CUresult`] and returns `Err(CudaError::Driver(…))`
/// if the result is not `Success`.
///
/// # Example
/// ```ignore
/// unsafe { check_cu!(ffi::cuInit(0)) }?;
/// ```
#[macro_export]
macro_rules! check_cu {
    ($expr:expr) => {{
        let code: u32 = $expr;
        $crate::cuda::error::CUresult::from_raw(code).into_result()
    }};
}

/// Checks a raw CUDA Runtime API return code.
///
/// Converts the `u32` to a [`CudartError`] and returns
/// `Err(CudaError::Runtime(…))` if the result is not `Success`.
#[macro_export]
macro_rules! check_cudart {
    ($expr:expr) => {{
        let code: u32 = $expr;
        $crate::cuda::error::CudartError::from_raw(code).into_result()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curesult_from_raw_success() {
        assert_eq!(CUresult::from_raw(0), CUresult::Success);
    }

    #[test]
    fn test_curesult_from_raw_oom() {
        assert_eq!(CUresult::from_raw(2), CUresult::OutOfMemory);
    }

    #[test]
    fn test_curesult_into_result_ok() {
        assert!(CUresult::Success.into_result().is_ok());
    }

    #[test]
    fn test_curesult_into_result_err() {
        assert!(CUresult::NoDevice.into_result().is_err());
    }

    #[test]
    fn test_cudart_from_raw_success() {
        assert_eq!(CudartError::from_raw(0), CudartError::Success);
    }

    #[test]
    fn test_cuda_error_display() {
        let e = CudaError::NoDevice;
        assert!(e.to_string().contains("no CUDA-capable"));
    }

    #[test]
    fn test_cuda_error_oom_display() {
        let e = CudaError::OutOfMemory {
            requested_bytes: 1024,
        };
        assert!(e.to_string().contains("1024"));
    }

    #[test]
    fn test_cuda_error_invalid_device_display() {
        let e = CudaError::InvalidDevice {
            requested: 3,
            available: 2,
        };
        assert!(e.to_string().contains("3"));
        assert!(e.to_string().contains("2"));
    }
}
