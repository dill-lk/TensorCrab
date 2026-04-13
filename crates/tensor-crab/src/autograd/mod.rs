//! # Autograd — Automatic Differentiation Engine
//!
//! This module provides Stage 2 of TensorCrab: automatic differentiation via
//! a dynamic computation graph.
//!
//! ## Core types
//!
//! - [`Variable`] — a [`crate::tensor::Tensor`] that tracks gradients.
//! - [`backward`] — walks the graph in reverse topological order, applying
//!   the chain rule.
//!
//! ## Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::{Variable, backward};
//!
//! // Create a leaf variable with requires_grad = true.
//! let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
//!
//! // Build a small computation graph:  z = sum(x * x)
//! let y = x.var_mul(&x);
//! let z = y.var_sum();
//!
//! // Run the backward pass.
//! backward(&z);
//!
//! // Inspect the gradient:  dz/dx = 2x  =>  [4.0, 6.0]
//! let g = x.grad().unwrap();
//! assert!((g.to_vec()[0] - 4.0_f32).abs() < 1e-5);
//! assert!((g.to_vec()[1] - 6.0_f32).abs() < 1e-5);
//! ```

pub mod engine;
pub mod graph;
pub mod ops;
pub mod variable;

pub use engine::backward;
pub use variable::Variable;
