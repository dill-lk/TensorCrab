//! # TensorCrab
//!
//! A Rust-native machine learning library — zero Python overhead, no GIL,
//! compile-time memory safety.
//!
//! ## Quick Start — Tensor Engine
//!
//! ```rust
//! use tensor_crab::prelude::*;
//!
//! let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
//! let c = a.matmul(&b).unwrap();
//! println!("{c}"); // [[19, 22], [43, 50]]
//! ```
//!
//! ## Quick Start — Autograd
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::{Variable, backward};
//!
//! let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
//! let z = x.var_mul(&x).var_sum(); // z = sum(x^2)
//! backward(&z);
//! // dz/dx = 2x = [4.0, 6.0]
//! let g = x.grad().unwrap();
//! assert!((g.to_vec()[0] - 4.0).abs() < 1e-5);
//! ```
//!
//! ## Quick Start — Neural Network
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::Variable;
//! use tensor_crab::nn::{Module, Sequential, Linear, ReLU};
//!
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(4, 8)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::new(8, 2)),
//! ]);
//! let x = Variable::new(Tensor::randn_seeded(&[3, 4], 0), false);
//! let y = model.forward(&x);
//! assert_eq!(y.data().shape(), &[3, 2]);
//! ```

//! ## Quick Start — Optimizers
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::{Variable, backward};
//! use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
//! use tensor_crab::optim::{Optimizer, Adam};
//!
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(4, 8)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::new(8, 1)),
//! ]);
//! let mut opt = Adam::new(model.parameters(), 0.001);
//!
//! let x = Variable::new(Tensor::randn_seeded(&[2, 4], 0), false);
//! let target = Variable::new(Tensor::zeros(&[2, 1]), false);
//! let pred = model.forward(&x);
//! let l = loss::mse_loss(&pred, &target);
//! backward(&l);
//! opt.step();
//! opt.zero_grad();
//! ```

pub mod autograd;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod error;
pub mod nn;
pub mod optim;
pub mod tensor;

pub use error::TensorError;
pub use tensor::Tensor;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use super::autograd::{backward, Variable};
    pub use super::nn::{Module, Sequential};
    pub use super::optim::{Adam, AdamW, DataLoader, Optimizer, StepLR, SGD};
    pub use super::Tensor;
    pub use super::TensorError;
}
