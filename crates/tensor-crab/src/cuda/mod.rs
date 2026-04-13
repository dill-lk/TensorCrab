//! # CUDA GPU Acceleration
//!
//! This module provides NVIDIA CUDA GPU support for TensorCrab operations.
//! It is only compiled when the `cuda` feature is enabled.
//!
//! ## Feature Flag
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! tensor-crab = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! ## Requirements
//!
//! - NVIDIA GPU with CUDA Compute Capability ≥ 5.0
//! - CUDA Toolkit 11.x or 12.x installed
//! - `CUDA_PATH` or `CUDA_ROOT` environment variable set (if CUDA is not in
//!   `/usr/local/cuda`)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    User Code                                │
//! │             CudaTensor ops (add, relu, …)                   │
//! └────────────────────────┬────────────────────────────────────┘
//!                          │
//! ┌────────────────────────▼────────────────────────────────────┐
//! │               ops::CudaTensor                              │
//! │    GPU-backed tensor with element-wise & reduction ops      │
//! └──────┬──────────────────┬───────────────────┬──────────────┘
//!        │                  │                   │
//! ┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼──────────────┐
//! │  buffer     │   │   module     │   │   kernels            │
//! │ CudaBuffer  │   │ CudaModule   │   │ Embedded PTX sources │
//! │ (GPU memory)│   │ CudaFunction │   │ for f32 ops          │
//! └──────┬──────┘   └───────┬──────┘   └──────────────────────┘
//!        │                  │
//! ┌──────▼──────────────────▼──────────────────────────────────┐
//! │                   ffi                                       │
//! │          Raw unsafe CUDA Driver API bindings                │
//! │   (cuInit, cuMemAlloc, cuLaunchKernel, …)                  │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use std::sync::Arc;
//! use tensor_crab::cuda::{CudaDevice, CudaTensor};
//!
//! // 1. Initialise the device.
//! let dev = Arc::new(CudaDevice::new(0).expect("no CUDA GPU"));
//! println!("Using: {}", dev.name());
//!
//! // 2. Upload data to the GPU.
//! let a = CudaTensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &dev)
//!     .expect("upload failed");
//! let b = CudaTensor::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2], &dev)
//!     .expect("upload failed");
//!
//! // 3. Run GPU operations.
//! let c = a.add(&b).expect("add failed");
//! let r = c.relu().expect("relu failed");
//!
//! // 4. Download results.
//! let result = r.to_cpu().expect("download failed");
//! println!("{result}");
//! # }
//! ```

pub mod buffer;
pub mod device;
pub mod error;
pub mod ffi;
pub mod kernels;
pub mod module;
pub mod ops;
pub mod stream;

pub use buffer::CudaBuffer;
pub use device::{CudaDevice, DeviceProperties};
pub use error::{CUresult, CudaError, CudaResult, CudartError};
pub use module::{grid_size_1d, CudaFunction, CudaModule, DEFAULT_BLOCK_SIZE};
pub use ops::CudaTensor;
pub use stream::{CudaEvent, CudaStream};
