//! Raw FFI bindings to the CUDA Driver API and CUDA Runtime API.
//!
//! All symbols in this module are `unsafe` to call.  Prefer the safe wrappers
//! in [`crate::cuda::device`], [`crate::cuda::stream`], and
//! [`crate::cuda::buffer`] when possible.
//!
//! ## Linking
//!
//! The linker flags for `libcuda` and `libcudart` are emitted by `build.rs`
//! when the `cuda` feature is enabled.  This module only declares the symbols;
//! it does **not** use `#[link(…)]` attributes directly so that the build
//! script can control link-search paths.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ─── Primitive CUDA types ─────────────────────────────────────────────────────

/// Device memory pointer.  Holds the address of GPU-resident data.
pub type CUdeviceptr = u64;

/// Integer handle identifying a CUDA-capable device (its ordinal index).
pub type CUdevice = c_int;

/// Raw CUDA Driver API result code (`u32`).
pub type CUresult = c_uint;

/// Raw CUDA Runtime API result code (`u32`).
pub type cudaError_t = c_uint;

// ─── Opaque handle types ──────────────────────────────────────────────────────

/// Opaque CUDA context handle.
///
/// A context is the primary object in the CUDA Driver API; all work (memory
/// allocations, kernel launches) takes place within a context.
#[repr(C)]
pub struct CUctx_st {
    _private: [u8; 0],
}

/// Pointer to an opaque [`CUctx_st`].
pub type CUcontext = *mut CUctx_st;

/// Opaque CUDA module handle.  Modules hold compiled PTX / cubin images.
#[repr(C)]
pub struct CUmod_st {
    _private: [u8; 0],
}

/// Pointer to an opaque [`CUmod_st`].
pub type CUmodule = *mut CUmod_st;

/// Opaque CUDA kernel function handle retrieved from a [`CUmodule`].
#[repr(C)]
pub struct CUfunc_st {
    _private: [u8; 0],
}

/// Pointer to an opaque [`CUfunc_st`].
pub type CUfunction = *mut CUfunc_st;

/// Opaque CUDA stream handle.
#[repr(C)]
pub struct CUstream_st {
    _private: [u8; 0],
}

/// Pointer to an opaque [`CUstream_st`].
pub type CUstream = *mut CUstream_st;

/// Opaque CUDA event handle.
#[repr(C)]
pub struct CUevent_st {
    _private: [u8; 0],
}

/// Pointer to an opaque [`CUevent_st`].
pub type CUevent = *mut CUevent_st;

// ─── Device attribute enum ────────────────────────────────────────────────────

/// CUDA device attributes that can be queried with
/// [`cuDeviceGetAttribute`].
#[repr(u32)]
#[allow(dead_code)]
pub enum CUdevice_attribute {
    /// Maximum number of threads per block.
    MaxThreadsPerBlock = 1,
    /// Maximum x-dimension of a block.
    MaxBlockDimX = 2,
    /// Maximum y-dimension of a block.
    MaxBlockDimY = 3,
    /// Maximum z-dimension of a block.
    MaxBlockDimZ = 4,
    /// Maximum x-dimension of a grid.
    MaxGridDimX = 5,
    /// Maximum y-dimension of a grid.
    MaxGridDimY = 6,
    /// Maximum z-dimension of a grid.
    MaxGridDimZ = 7,
    /// Total amount of shared memory available per block in bytes.
    MaxSharedMemoryPerBlock = 8,
    /// Total amount of constant memory available on the device in bytes.
    TotalConstantMemory = 9,
    /// Warp size in threads.
    WarpSize = 10,
    /// Maximum pitch in bytes allowed by the memory copy functions.
    MaxPitch = 11,
    /// Maximum number of 32-bit registers available per block.
    MaxRegistersPerBlock = 12,
    /// Peak clock frequency in kilohertz.
    ClockRate = 13,
    /// Alignment requirement for textures.
    TextureAlignment = 14,
    /// Device can possibly copy memory and execute a kernel concurrently.
    GpuOverlap = 15,
    /// Number of multiprocessors on the device.
    MultiprocessorCount = 16,
    /// Major compute capability version number.
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number.
    ComputeCapabilityMinor = 76,
    /// Device supports executing multiple kernels within the same context
    /// simultaneously.
    ConcurrentKernels = 31,
    /// Device has ECC support enabled.
    EccEnabled = 32,
    /// PCI bus ID of the device.
    PciBusId = 33,
    /// PCI device ID of the device.
    PciDeviceId = 34,
    /// Device is using a TCC driver model.
    TccDriver = 35,
    /// Peak memory clock frequency in kilohertz.
    MemoryClockRate = 36,
    /// Global memory bus width in bits.
    GlobalMemoryBusWidth = 37,
    /// Size of the L2 cache in bytes.
    L2CacheSize = 38,
    /// Maximum resident threads per multiprocessor.
    MaxThreadsPerMultiprocessor = 39,
    /// Number of asynchronous engines.
    AsyncEngineCount = 40,
    /// Device shares a unified address space with the host.
    UnifiedAddressing = 41,
}

// ─── Memory-copy direction ────────────────────────────────────────────────────

/// Direction flags for [`cudaMemcpy`] / [`cuMemcpyHtoD`] etc.
#[repr(u32)]
#[allow(dead_code)]
pub enum cudaMemcpyKind {
    /// Host → Host copy.
    HostToHost = 0,
    /// Host → Device copy.
    HostToDevice = 1,
    /// Device → Host copy.
    DeviceToHost = 2,
    /// Device → Device copy.
    DeviceToDevice = 3,
    /// Direction inferred from pointer attributes (unified memory).
    Default = 4,
}

// ─── CUDA Driver API extern declarations ─────────────────────────────────────

extern "C" {
    // ── Initialisation ────────────────────────────────────────────────────────

    /// Initialises the CUDA Driver API.
    ///
    /// Must be called before any other Driver API function.
    /// `flags` must be `0`.
    pub fn cuInit(flags: c_uint) -> CUresult;

    // ── Device management ─────────────────────────────────────────────────────

    /// Returns the number of CUDA-capable devices.
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;

    /// Returns a device handle for the device with ordinal `ordinal`.
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;

    /// Returns a human-readable name for the device (NUL-terminated).
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;

    /// Returns the total amount of memory on the device, in bytes.
    pub fn cuDeviceTotalMem(bytes: *mut usize, dev: CUdevice) -> CUresult;

    /// Returns information about a device attribute.
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_uint, dev: CUdevice) -> CUresult;

    // ── Context management ────────────────────────────────────────────────────

    /// Creates a CUDA context for a device.
    ///
    /// `flags` is usually `0`.  The newly created context is pushed onto the
    /// per-thread context stack and becomes the active context.
    pub fn cuCtxCreate(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;

    /// Destroys a previously created CUDA context.
    pub fn cuCtxDestroy(ctx: CUcontext) -> CUresult;

    /// Pushes a context onto the current CPU thread's context stack.
    pub fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult;

    /// Pops the current context from the CPU thread's context stack.
    pub fn cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult;

    /// Synchronises the context, blocking until all prior work is complete.
    pub fn cuCtxSynchronize() -> CUresult;

    // ── Memory management ─────────────────────────────────────────────────────

    /// Allocates `bytesize` bytes of device memory.
    ///
    /// On success `*dptr` is set to the device address of the allocation.
    pub fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

    /// Frees device memory at `dptr`.
    pub fn cuMemFree(dptr: CUdeviceptr) -> CUresult;

    /// Copies `byte_count` bytes from host memory at `src_host` to the
    /// device allocation at `dst_device`.
    pub fn cuMemcpyHtoD(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
    ) -> CUresult;

    /// Copies `byte_count` bytes from device memory at `src_device` to the
    /// host buffer at `dst_host`.
    pub fn cuMemcpyDtoH(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult;

    /// Copies `byte_count` bytes between two device memory locations.
    pub fn cuMemcpyDtoD(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult;

    /// Asynchronously copies `byte_count` bytes from host to device on `stream`.
    pub fn cuMemcpyHtoDAsync(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;

    /// Asynchronously copies `byte_count` bytes from device to host on `stream`.
    pub fn cuMemcpyDtoHAsync(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;

    /// Sets `byte_count` bytes of device memory starting at `dst_device` to
    /// `uc` (interpreted as `u8`).
    pub fn cuMemsetD8(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult;

    /// Sets `n` 32-bit words of device memory starting at `dst_device` to `ui`.
    pub fn cuMemsetD32(dst_device: CUdeviceptr, ui: c_uint, n: usize) -> CUresult;

    // ── Module and kernel management ──────────────────────────────────────────

    /// Loads a PTX or cubin module from the NUL-terminated string `image`.
    ///
    /// PTX strings must be null-terminated.
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;

    /// Loads a PTX or cubin module from a file on the filesystem.
    pub fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult;

    /// Unloads a module, freeing its resources.
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;

    /// Retrieves a kernel function handle from `module` by name.
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;

    // ── Kernel launch ─────────────────────────────────────────────────────────

    /// Launches a CUDA kernel.
    ///
    /// # Parameters
    /// - `f` — kernel function handle
    /// - `grid_{x,y,z}` — grid dimensions in blocks
    /// - `block_{x,y,z}` — block dimensions in threads
    /// - `shared_mem_bytes` — dynamic shared memory per block
    /// - `stream` — stream to launch on (null = default stream)
    /// - `kernel_params` — pointer to array of kernel argument pointers
    /// - `extra` — reserved; must be `null`
    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    // ── Stream management ─────────────────────────────────────────────────────

    /// Creates a new CUDA stream.  `flags` should be `0` or
    /// `CU_STREAM_NON_BLOCKING` (= 1).
    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;

    /// Destroys a CUDA stream.
    pub fn cuStreamDestroy(stream: CUstream) -> CUresult;

    /// Blocks the CPU until all work enqueued on `stream` has completed.
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    /// Returns `CUDA_SUCCESS` if all work on `stream` has completed, or
    /// `CUDA_ERROR_NOT_READY` if it has not.
    pub fn cuStreamQuery(stream: CUstream) -> CUresult;

    // ── Event management ──────────────────────────────────────────────────────

    /// Creates a CUDA event.  `flags` is usually `0`.
    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;

    /// Destroys a CUDA event.
    pub fn cuEventDestroy(event: CUevent) -> CUresult;

    /// Records `event` on `stream`.
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;

    /// Blocks the CPU until `event` has been reached.
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;

    /// Returns the elapsed time in milliseconds between `start` and `end`.
    pub fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    // ── Peer access ───────────────────────────────────────────────────────────

    /// Returns `1` if device `src` can directly access device `dst`'s memory.
    pub fn cuDeviceCanAccessPeer(
        can_access: *mut c_int,
        src_device: CUdevice,
        dst_device: CUdevice,
    ) -> CUresult;

    /// Enables peer access from the current context to the context `peer_context`.
    pub fn cuCtxEnablePeerAccess(peer_context: CUcontext, flags: c_uint) -> CUresult;
}

// ─── CUDA Runtime API extern declarations ─────────────────────────────────────

extern "C" {
    // ── Memory ────────────────────────────────────────────────────────────────

    /// Allocates `size` bytes of device memory.
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;

    /// Frees device memory.
    pub fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;

    /// Synchronous host↔device copy.
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
    ) -> cudaError_t;

    /// Asynchronous host↔device copy on a stream.
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
        stream: CUstream,
    ) -> cudaError_t;

    /// Sets `count` bytes at `dev_ptr` to `value`.
    pub fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> cudaError_t;

    // ── Device ────────────────────────────────────────────────────────────────

    /// Returns the number of CUDA-capable devices.
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

    /// Sets the active device for the current host thread.
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;

    /// Returns the current active device.
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;

    /// Blocks the host thread until all previously issued CUDA work has
    /// completed on the device.
    pub fn cudaDeviceSynchronize() -> cudaError_t;

    /// Resets the current device, releasing all resources.
    pub fn cudaDeviceReset() -> cudaError_t;

    // ── Error reporting ───────────────────────────────────────────────────────

    /// Returns a NUL-terminated string describing the CUDA error code.
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

    /// Returns the last error from a runtime call.
    pub fn cudaGetLastError() -> cudaError_t;

    /// Returns but does not clear the last runtime error.
    pub fn cudaPeekAtLastError() -> cudaError_t;
}

// ─── cuBLAS types and extern declarations ─────────────────────────────────────

/// Opaque cuBLAS context handle.
#[repr(C)]
pub struct cublasContext {
    _private: [u8; 0],
}

/// Pointer to an opaque [`cublasContext`].
pub type cublasHandle_t = *mut cublasContext;

/// Raw cuBLAS status code.
pub type cublasStatus_t = c_uint;

/// Matrix transposition option for cuBLAS operations.
#[repr(u32)]
#[allow(dead_code)]
pub enum cublasOperation_t {
    /// No transpose.
    CUBLAS_OP_N = 0,
    /// Transpose.
    CUBLAS_OP_T = 1,
    /// Conjugate transpose.
    CUBLAS_OP_C = 2,
}

extern "C" {
    /// Creates a cuBLAS library context handle.
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;

    /// Destroys a cuBLAS library context handle and frees all its resources.
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

    /// Sets the cuBLAS library stream for asynchronous execution.
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;

    /// Single-precision general matrix-matrix multiplication.
    ///
    /// Computes `C = alpha * op(A) * op(B) + beta * C`.
    #[allow(clippy::too_many_arguments)]
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: CUdeviceptr,
        lda: c_int,
        B: CUdeviceptr,
        ldb: c_int,
        beta: *const f32,
        C: CUdeviceptr,
        ldc: c_int,
    ) -> cublasStatus_t;
}
