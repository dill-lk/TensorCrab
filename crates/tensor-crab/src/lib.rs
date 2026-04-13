//! # TensorCrab
//!
//! A Rust-native machine learning library — zero Python overhead, no GIL,
//! compile-time memory safety.
//!
//! ## Quick Start
//!
//! ```rust
//! use tensor_crab::prelude::*;
//!
//! let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
//! let c = a.matmul(&b).unwrap();
//! println!("{c}"); // [[19, 22], [43, 50]]
//! ```

pub mod error;
pub mod tensor;

pub use error::TensorError;
pub use tensor::Tensor;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use super::Tensor;
    pub use super::TensorError;
}
