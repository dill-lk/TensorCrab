//! PTX module loading and CUDA kernel function management.
//!
//! A [`CudaModule`] wraps a loaded PTX program image.  Individual kernel
//! functions are retrieved as [`CudaFunction`] handles and invoked via
//! [`CudaFunction::launch`].

use std::ffi::CString;
use std::sync::Arc;

use super::device::CudaDevice;
use super::error::{CUresult, CudaError, CudaResult};
use super::ffi;

// ─── CudaModule ───────────────────────────────────────────────────────────────

/// A loaded CUDA module (compiled PTX or cubin image).
///
/// Modules are the unit of compiled GPU code in the CUDA Driver API.  A single
/// module may contain multiple kernel functions.
pub struct CudaModule {
    raw: ffi::CUmodule,
    /// Keep the device alive.
    _device: Arc<CudaDevice>,
}

// SAFETY: CUmodule is safe to move across threads.
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Loads a PTX module from the null-terminated string `ptx`.
    ///
    /// The PTX is JIT-compiled for the current device on first load.
    ///
    /// # Errors
    /// Returns [`CudaError::ModuleLoad`] if compilation fails.
    pub fn from_ptx(ptx: &str, device: &Arc<CudaDevice>) -> CudaResult<Self> {
        // CString ensures null termination.
        let ptx_cstr = CString::new(ptx).map_err(|_| {
            CudaError::ModuleLoad("PTX string contains interior NUL byte".to_string())
        })?;

        let mut raw: ffi::CUmodule = std::ptr::null_mut();
        let result = unsafe { ffi::cuModuleLoadData(&mut raw, ptx_cstr.as_ptr().cast()) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::ModuleLoad(format!("cuModuleLoadData failed: {e}")))?;

        Ok(Self {
            raw,
            _device: Arc::clone(device),
        })
    }

    /// Loads a PTX or cubin module from a file on disk.
    ///
    /// # Errors
    /// Returns [`CudaError::ModuleLoad`] if the file is not found or cannot
    /// be compiled.
    pub fn from_file(path: &str, device: &Arc<CudaDevice>) -> CudaResult<Self> {
        let path_cstr = CString::new(path)
            .map_err(|_| CudaError::ModuleLoad("path contains interior NUL byte".to_string()))?;

        let mut raw: ffi::CUmodule = std::ptr::null_mut();
        let result = unsafe { ffi::cuModuleLoad(&mut raw, path_cstr.as_ptr()) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::ModuleLoad(format!("cuModuleLoad('{path}') failed: {e}")))?;

        Ok(Self {
            raw,
            _device: Arc::clone(device),
        })
    }

    /// Retrieves a kernel function handle by `name` from this module.
    ///
    /// The returned [`CudaFunction`] borrows from this module — it becomes
    /// invalid if the module is dropped.
    ///
    /// # Errors
    /// Returns [`CudaError::FunctionNotFound`] if no function with that name
    /// exists in the module.
    pub fn function(&self, name: &str) -> CudaResult<CudaFunction<'_>> {
        let name_cstr =
            CString::new(name).map_err(|_| CudaError::FunctionNotFound(name.to_string()))?;

        let mut func: ffi::CUfunction = std::ptr::null_mut();
        let result = unsafe { ffi::cuModuleGetFunction(&mut func, self.raw, name_cstr.as_ptr()) };
        CUresult::from_raw(result)
            .into_result()
            .map_err(|_| CudaError::FunctionNotFound(name.to_string()))?;

        Ok(CudaFunction {
            raw: func,
            _module: self,
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            ffi::cuModuleUnload(self.raw);
        }
    }
}

impl std::fmt::Debug for CudaModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModule")
            .field("raw", &(self.raw as usize))
            .finish()
    }
}

// ─── CudaFunction ─────────────────────────────────────────────────────────────

/// A handle to a GPU kernel function within a [`CudaModule`].
///
/// Launch parameters (grid and block dimensions) are specified when calling
/// [`CudaFunction::launch`].
pub struct CudaFunction<'m> {
    raw: ffi::CUfunction,
    /// Borrow the module so it cannot be dropped while this function exists.
    _module: &'m CudaModule,
}

impl<'m> CudaFunction<'m> {
    /// Launches the kernel with the given configuration.
    ///
    /// # Parameters
    /// - `grid`  — `(x, y, z)` dimensions of the grid (in blocks).
    /// - `block` — `(x, y, z)` dimensions of each block (in threads).
    /// - `shared_mem` — dynamic shared memory per block in bytes.
    /// - `stream` — the stream to launch on (pass `None` for the default
    ///   stream).
    /// - `args`  — slice of pointers to kernel arguments.  Each element must
    ///   point to the actual argument value (not to a pointer to it, unless the
    ///   kernel argument is itself a pointer).
    ///
    /// # Safety
    /// The caller must ensure:
    /// - `args` contains exactly as many elements as the kernel expects.
    /// - Each pointer in `args` is valid for the duration of the launch.
    /// - The grid/block dimensions are within device limits.
    ///
    /// # Errors
    /// Returns [`CudaError::KernelLaunch`] if the launch fails.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: Option<ffi::CUstream>,
        args: &mut [*mut std::os::raw::c_void],
    ) -> CudaResult<()> {
        use super::ffi::cuLaunchKernel;

        let raw_stream = stream.unwrap_or(std::ptr::null_mut());

        let result = cuLaunchKernel(
            self.raw,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem,
            raw_stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        CUresult::from_raw(result)
            .into_result()
            .map_err(|e| CudaError::KernelLaunch(format!("cuLaunchKernel failed: {e}")))
    }
}

impl<'m> std::fmt::Debug for CudaFunction<'m> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaFunction")
            .field("raw", &(self.raw as usize))
            .finish()
    }
}

// ─── Grid / block helpers ─────────────────────────────────────────────────────

/// Computes a 1-D grid size sufficient to cover `n` elements with `block_size`
/// threads per block.
///
/// # Example
/// ```
/// # use tensor_crab::cuda::module::grid_size_1d;
/// assert_eq!(grid_size_1d(1024, 256), 4);
/// assert_eq!(grid_size_1d(1025, 256), 5);
/// ```
pub fn grid_size_1d(n: u32, block_size: u32) -> u32 {
    n.div_ceil(block_size)
}

/// Default block size for 1-D element-wise kernels.
///
/// 256 threads per block is a common heuristic that balances occupancy with
/// register pressure on most NVIDIA architectures.
pub const DEFAULT_BLOCK_SIZE: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size_exact_multiple() {
        assert_eq!(grid_size_1d(1024, 256), 4);
    }

    #[test]
    fn test_grid_size_partial_block() {
        assert_eq!(grid_size_1d(1025, 256), 5);
    }

    #[test]
    fn test_grid_size_less_than_one_block() {
        assert_eq!(grid_size_1d(100, 256), 1);
    }

    #[test]
    fn test_grid_size_single_element() {
        assert_eq!(grid_size_1d(1, 256), 1);
    }

    #[test]
    fn test_default_block_size() {
        // Compile-time constant sanity check.
        assert_eq!(DEFAULT_BLOCK_SIZE, 256);
    }
}
