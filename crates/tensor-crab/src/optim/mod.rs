//! # Optimizers (`optim`)
//!
//! Gradient-based optimizers for training neural network parameters.
//!
//! ## Optimizers
//!
//! | Type | Description |
//! |---|---|
//! | [`SGD`] | Stochastic gradient descent with optional momentum and weight decay |
//! | [`Adam`] | Adaptive moment estimation |
//! | [`AdamW`] | Adam with decoupled weight decay |
//!
//! ## Learning Rate Schedulers
//!
//! | Type | Description |
//! |---|---|
//! | [`StepLR`] | Multiplies learning rate by `gamma` every `step_size` epochs |
//! | [`CosineAnnealingLR`] | Cosine-annealed learning rate from `lr_max` to `lr_min` |
//!
//! ## DataLoader
//!
//! [`DataLoader`] wraps a dataset (pairs of input / target tensors) and yields
//! mini-batches with optional shuffling each epoch.
//!
//! ## Quick Start
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::{Variable, backward};
//! use tensor_crab::nn::{Module, Sequential, Linear, loss};
//! use tensor_crab::optim::{Optimizer, SGD};
//!
//! let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
//! let mut opt = SGD::new(model.parameters(), 0.01);
//!
//! // One training step
//! let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
//! let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
//! let pred = model.forward(&x);
//! let l = loss::mse_loss(&pred, &target);
//! backward(&l);
//! opt.step();
//! opt.zero_grad();
//! ```

pub mod adam;
pub mod adamw;
pub mod dataloader;
pub mod scheduler;
pub mod sgd;

pub use adam::Adam;
pub use adamw::AdamW;
pub use dataloader::DataLoader;
pub use scheduler::{CosineAnnealingLR, StepLR};
pub use sgd::SGD;

use std::sync::Arc;

use crate::autograd::Variable;

/// The common interface for all optimizers.
///
/// Call [`Optimizer::step`] after computing gradients (via `backward()`) to
/// update all registered parameters, then call [`Optimizer::zero_grad`] to
/// clear the gradients before the next forward pass.
pub trait Optimizer {
    /// Updates all registered parameters using their current gradients.
    ///
    /// Parameters with `requires_grad = false` or a `None` gradient are
    /// silently skipped.
    fn step(&mut self);

    /// Resets the gradients of all registered parameters to `None`.
    ///
    /// Must be called before the next forward pass to avoid accumulating
    /// gradients across steps.
    fn zero_grad(&self);
}

/// Returns all parameters that have `requires_grad = true`.
///
/// Convenience helper used by optimizer constructors.
pub(crate) fn filter_trainable(params: Vec<Arc<Variable>>) -> Vec<Arc<Variable>> {
    params.into_iter().filter(|p| p.requires_grad).collect()
}
