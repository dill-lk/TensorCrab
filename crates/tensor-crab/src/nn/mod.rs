//! # Neural Network Modules (`nn`)
//!
//! Composable building blocks for building and training neural networks on top
//! of the autograd engine.
//!
//! ## Core Trait
//!
//! Every layer implements [`Module`]:
//! - [`Module::forward`] — runs the layer's computation.
//! - [`Module::parameters`] — returns all trainable parameters.
//! - [`Module::zero_grad`] — resets all parameter gradients.
//!
//! ## Layers
//!
//! | Type | Description |
//! |---|---|
//! | [`Linear`] | Fully-connected layer: `x @ Wᵀ + b` |
//! | [`ReLU`] | ReLU activation |
//! | [`Sigmoid`] | Sigmoid activation |
//! | [`Tanh`] | Tanh activation |
//! | [`Softmax`] | Softmax over a chosen axis |
//! | [`Sequential`] | Ordered container of modules |
//! | [`BatchNorm1d`] | Batch normalisation for 2-D inputs |
//! | [`Dropout`] | Training-time stochastic zeroing |
//!
//! ## Loss Functions
//!
//! Free functions in [`loss`]:
//! - [`loss::mse_loss`] — mean squared error
//! - [`loss::bce_loss`] — binary cross-entropy
//! - [`loss::cross_entropy_loss`] — categorical cross-entropy
//!
//! ## Quick Start
//!
//! ```rust
//! use std::sync::Arc;
//! use tensor_crab::tensor::Tensor;
//! use tensor_crab::autograd::{Variable, backward};
//! use tensor_crab::nn::{Module, Sequential, Linear, ReLU};
//!
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(4, 8)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::new(8, 2)),
//! ]);
//!
//! let x = Variable::new(Tensor::randn_seeded(&[3, 4], 0), false);
//! let y = model.forward(&x);
//! assert_eq!(y.data.shape(), &[3, 2]);
//! ```

pub mod activations;
pub mod batchnorm;
pub mod dropout;
pub mod linear;
pub mod loss;
pub mod sequential;

pub use activations::{ReLU, Sigmoid, Softmax, Tanh};
pub use batchnorm::BatchNorm1d;
pub use dropout::Dropout;
pub use linear::Linear;
pub use sequential::Sequential;

use std::sync::Arc;

use crate::autograd::Variable;

/// A composable unit of a neural network.
///
/// Implement this trait to create custom layers.  All built-in layers
/// ([`Linear`], [`ReLU`], [`Sequential`], …) implement `Module`.
pub trait Module: Send + Sync {
    /// Runs the layer's computation on `input` and returns the output.
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable>;

    /// Returns all trainable parameters owned by this module (and its
    /// sub-modules, for containers like [`Sequential`]).
    fn parameters(&self) -> Vec<Arc<Variable>>;

    /// Resets all parameter gradients to `None`.
    ///
    /// Call this before each training step.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// Switches the module (and its children) to **training** mode.
    ///
    /// In training mode, [`Dropout`] applies random masking and
    /// [`BatchNorm1d`] uses batch statistics.
    ///
    /// The default implementation is a no-op for stateless modules.
    fn set_training(&self, _training: bool) {}
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::autograd::{backward, Variable};
    use crate::tensor::Tensor;

    use super::{
        loss, BatchNorm1d, Dropout, Linear, Module, ReLU, Sequential, Sigmoid, Softmax, Tanh,
    };

    // ── New tensor ops ────────────────────────────────────────────────────

    #[test]
    fn test_tanh_op() {
        let a = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]);
        let b = a.tanh();
        assert_abs_diff_eq!(b.to_vec()[0], 0.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(b.to_vec()[1], 1.0_f32.tanh(), epsilon = 1e-6);
        assert_abs_diff_eq!(b.to_vec()[2], (-1.0_f32).tanh(), epsilon = 1e-6);
    }

    #[test]
    fn test_sqrt_op() {
        let a = Tensor::from_vec(vec![4.0_f32, 9.0, 16.0], &[3]);
        let b = a.sqrt();
        assert_abs_diff_eq!(
            b.to_vec().as_slice(),
            [2.0_f32, 3.0, 4.0].as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_broadcast_to_op() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
        let b = a.broadcast_to(&[4, 3]);
        assert_eq!(b.shape(), &[4, 3]);
        // Every row should equal [1, 2, 3].
        let v = b.to_vec();
        for row in 0..4 {
            assert_abs_diff_eq!(v[row * 3], 1.0_f32, epsilon = 1e-6);
            assert_abs_diff_eq!(v[row * 3 + 1], 2.0_f32, epsilon = 1e-6);
            assert_abs_diff_eq!(v[row * 3 + 2], 3.0_f32, epsilon = 1e-6);
        }
    }

    // ── New autograd ops ─────────────────────────────────────────────────

    #[test]
    fn test_var_transpose_grad() {
        // z = sum(x^T)  →  dz/dx = ones(x.shape)
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]),
            true,
        );
        let z = x.var_transpose().var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        assert_eq!(g.shape(), &[2, 2]);
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [1.0_f32, 1.0, 1.0, 1.0].as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_var_tanh_grad() {
        // z = sum(tanh(x))  →  dz/dx = 1 - tanh(x)^2
        let x_data = Tensor::from_vec(vec![0.5_f32, -0.5, 1.0], &[3]);
        let x = Variable::new(x_data.clone(), true);
        let z = x.var_tanh().var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        let expected: Vec<f32> = x_data
            .to_vec()
            .iter()
            .map(|&v| 1.0 - v.tanh().powi(2))
            .collect();
        for (a, e) in g.to_vec().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_var_sqrt_grad() {
        // z = sum(sqrt(x))  →  dz/dx = 1 / (2 * sqrt(x))
        let x_data = Tensor::from_vec(vec![1.0_f32, 4.0, 9.0], &[3]);
        let x = Variable::new(x_data.clone(), true);
        let z = x.var_sqrt().var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        let expected: Vec<f32> = x_data
            .to_vec()
            .iter()
            .map(|&v| 1.0 / (2.0 * v.sqrt()))
            .collect();
        for (a, e) in g.to_vec().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_var_sum_keepdim_grad() {
        // z = sum_keepdim(x, axis=0)  has shape [1,3] for input [2,3]
        // sum(z) gives scalar; grad flows back to all [2,3] elements.
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            true,
        );
        let z = x.var_sum_keepdim(0).var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_var_mul_scalar_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), true);
        let z = x.var_mul_scalar(3.0).var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [3.0_f32, 3.0, 3.0].as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_var_add_scalar_grad() {
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
        let z = x.var_add_scalar(10.0).var_sum();
        backward(&z);
        let g = x.grad().unwrap();
        assert_abs_diff_eq!(
            g.to_vec().as_slice(),
            [1.0_f32, 1.0].as_slice(),
            epsilon = 1e-6
        );
    }

    // ── Linear ───────────────────────────────────────────────────────────

    #[test]
    fn test_linear_output_shape() {
        let linear = Linear::new(4, 8);
        let x = Variable::new(Tensor::randn_seeded(&[3, 4], 0), false);
        let y = linear.forward(&x);
        assert_eq!(y.data.shape(), &[3, 8]);
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::new(3, 5);
        let params = linear.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].data.shape(), &[5, 3]); // weight
        assert_eq!(params[1].data.shape(), &[5]); // bias
    }

    #[test]
    fn test_linear_grad_flows() {
        let linear = Linear::new(2, 3);
        let x = Variable::new(Tensor::randn_seeded(&[4, 2], 1), false);
        let y = linear.forward(&x).var_sum();
        backward(&y);
        assert!(
            linear.weight.grad().is_some(),
            "weight should have a gradient"
        );
        assert!(linear.bias.grad().is_some(), "bias should have a gradient");
    }

    #[test]
    fn test_linear_kaiming_shape() {
        let linear = Linear::new_kaiming(4, 6);
        assert_eq!(linear.weight.data.shape(), &[6, 4]);
        assert_eq!(linear.bias.data.shape(), &[6]);
    }

    // ── Activations ──────────────────────────────────────────────────────

    #[test]
    fn test_relu_activation() {
        let relu = ReLU::new();
        let x = Variable::new(Tensor::from_vec(vec![-1.0_f32, 0.0, 2.0], &[3]), false);
        let y = relu.forward(&x);
        assert_eq!(y.data.to_vec(), vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_activation() {
        let sig = Sigmoid::new();
        let x = Variable::new(Tensor::from_vec(vec![0.0_f32], &[1]), false);
        let y = sig.forward(&x);
        assert_abs_diff_eq!(y.data.to_vec()[0], 0.5_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_activation() {
        let tanh = Tanh::new();
        let x = Variable::new(Tensor::from_vec(vec![0.0_f32], &[1]), false);
        let y = tanh.forward(&x);
        assert_abs_diff_eq!(y.data.to_vec()[0], 0.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let sm = Softmax::new(0);
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
        let y = sm.forward(&x);
        let sum: f32 = y.data.to_vec().iter().sum();
        assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_batch_sums_to_one() {
        // Each row should sum to 1.
        let sm = Softmax::new(1);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            false,
        );
        let y = sm.forward(&x);
        let v = y.data.to_vec();
        let row0_sum: f32 = v[0..3].iter().sum();
        let row1_sum: f32 = v[3..6].iter().sum();
        assert_abs_diff_eq!(row0_sum, 1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(row1_sum, 1.0_f32, epsilon = 1e-5);
    }

    // ── Sequential ───────────────────────────────────────────────────────

    #[test]
    fn test_sequential_output_shape() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(8, 2)),
        ]);
        let x = Variable::new(Tensor::randn_seeded(&[5, 4], 2), false);
        let y = model.forward(&x);
        assert_eq!(y.data.shape(), &[5, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(4, 2)),
        ]);
        // 2 layers × (weight + bias) = 4 parameters
        assert_eq!(model.parameters().len(), 4);
    }

    #[test]
    fn test_sequential_zero_grad() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 3))]);
        let x = Variable::new(Tensor::randn_seeded(&[2, 2], 3), false);
        let loss = model.forward(&x).var_sum();
        backward(&loss);
        assert!(model.parameters()[0].grad().is_some());
        model.zero_grad();
        assert!(model.parameters()[0].grad().is_none());
    }

    #[test]
    fn test_sequential_grad_flows() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(8, 1)),
        ]);
        let x = Variable::new(Tensor::randn_seeded(&[3, 4], 5), false);
        let loss = model.forward(&x).var_sum();
        backward(&loss);
        for p in model.parameters() {
            assert!(p.grad().is_some(), "all parameters should have gradients");
        }
    }

    // ── Loss functions ───────────────────────────────────────────────────

    #[test]
    fn test_mse_loss_zero() {
        let pred = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
        let l = loss::mse_loss(&pred, &target);
        assert_abs_diff_eq!(l.data.to_vec()[0], 0.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_value() {
        // mse([0, 0], [1, 1]) = 1.0
        let pred = Variable::new(Tensor::zeros(&[2]), true);
        let target = Variable::new(Tensor::ones(&[2]), false);
        let l = loss::mse_loss(&pred, &target);
        assert_abs_diff_eq!(l.data.to_vec()[0], 1.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_grad() {
        // d(MSE)/d(pred) = 2*(pred - target)/n
        let pred = Variable::new(Tensor::from_vec(vec![2.0_f32, 4.0], &[2]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0_f32, 0.0], &[2]), false);
        let l = loss::mse_loss(&pred, &target);
        backward(&l);
        let g = pred.grad().unwrap();
        // n=2, grad = 2*(pred-target)/n = [2.0, 4.0]
        assert_abs_diff_eq!(g.to_vec()[0], 2.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g.to_vec()[1], 4.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_bce_loss_perfect_prediction() {
        let pred = Variable::new(Tensor::from_vec(vec![0.99_f32, 0.01], &[2]), true);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[2]), false);
        let l = loss::bce_loss(&pred, &target);
        // loss should be close to 0
        assert!(
            l.data.to_vec()[0] < 0.02,
            "BCE with near-perfect preds should be low"
        );
    }

    #[test]
    fn test_cross_entropy_loss_correct_class() {
        // Correct class (idx 1) has high logit → low loss.
        let logits = Variable::new(Tensor::from_vec(vec![0.1_f32, 3.0, 0.2], &[3]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0, 0.0], &[3]), false);
        let l = loss::cross_entropy_loss(&logits, &target);
        assert!(
            l.data.to_vec()[0] < 0.2,
            "CE loss should be low when correct class has high logit"
        );
    }

    #[test]
    fn test_cross_entropy_grad_flows() {
        let logits = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 0.5], &[3]), true);
        let target = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0, 0.0], &[3]), false);
        let l = loss::cross_entropy_loss(&logits, &target);
        backward(&l);
        assert!(logits.grad().is_some());
    }

    // ── BatchNorm1d ──────────────────────────────────────────────────────

    #[test]
    fn test_batchnorm_output_shape() {
        let bn = BatchNorm1d::new(4);
        let x = Variable::new(Tensor::randn_seeded(&[8, 4], 10), false);
        let y = bn.forward(&x);
        assert_eq!(y.data.shape(), &[8, 4]);
    }

    #[test]
    fn test_batchnorm_normalises() {
        // After normalisation + γ=1, β=0 the output should have ~0 mean and ~1 std.
        let bn = BatchNorm1d::new(1);
        // Simple all-ones gamma and zero bias already set by default.
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let x = Variable::new(Tensor::from_vec(data, &[8, 1]), false);
        let y = bn.forward(&x);
        let v = y.data.to_vec();
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert_abs_diff_eq!(mean, 0.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_batchnorm_parameters() {
        let bn = BatchNorm1d::new(6);
        let params = bn.parameters();
        assert_eq!(params.len(), 2); // gamma, beta
        assert_eq!(params[0].data.shape(), &[6]);
        assert_eq!(params[1].data.shape(), &[6]);
    }

    #[test]
    fn test_batchnorm_eval_mode() {
        // In eval mode, running stats are used and output should be stable
        // across batches.
        let bn = BatchNorm1d::new(2);
        // First warm up with a training batch to build running stats.
        let x_train = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]),
            false,
        );
        let _ = bn.forward(&x_train);

        // Switch to eval mode.
        bn.set_training(false);
        let x_eval = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]),
            false,
        );
        let y = bn.forward(&x_eval);
        assert_eq!(y.data.shape(), &[2, 2]);
    }

    // ── Dropout ──────────────────────────────────────────────────────────

    #[test]
    fn test_dropout_eval_identity() {
        let dropout = Dropout::new(0.5);
        dropout.set_training(false);
        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
        let y = dropout.forward(&x);
        assert_eq!(y.data.to_vec(), x.data.to_vec());
    }

    #[test]
    fn test_dropout_training_zeroes_some() {
        // With p=0.5 and a large input, some elements should be zeroed.
        let dropout = Dropout::with_seed(0.5, 42);
        let data: Vec<f32> = vec![1.0_f32; 100];
        let x = Variable::new(Tensor::from_vec(data, &[100]), false);
        let y = dropout.forward(&x);
        let zeroed = y.data.to_vec().iter().filter(|&&v| v == 0.0).count();
        assert!(zeroed > 10, "expected some zeroed elements, got {zeroed}");
        assert!(
            zeroed < 90,
            "expected some surviving elements, got {zeroed} zeroed"
        );
    }

    #[test]
    fn test_dropout_scale() {
        // The surviving elements should be scaled by 1/(1-p).
        let dropout = Dropout::with_seed(0.0, 0); // p=0, nothing dropped
        let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 2.0, 2.0], &[3]), false);
        let y = dropout.forward(&x);
        // With p=0, scale=1/(1-0)=1.0, so output == input.
        assert_abs_diff_eq!(
            y.data.to_vec().as_slice(),
            [2.0_f32, 2.0, 2.0].as_slice(),
            epsilon = 1e-6
        );
    }

    // ── Save / Load ───────────────────────────────────────────────────────

    #[test]
    fn test_save_load_weights() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4)),
            Box::new(Linear::new(4, 2)),
        ]);

        let tmp_path = "/tmp/test_nn_weights.bin";
        model.save_weights(tmp_path).expect("save_weights failed");

        let loaded = model.load_weights(tmp_path).expect("load_weights failed");
        let original_params = model.parameters();

        assert_eq!(loaded.len(), original_params.len());
        for (loaded_t, param) in loaded.iter().zip(original_params.iter()) {
            assert_eq!(loaded_t.shape(), param.data.shape());
            for (a, b) in loaded_t.to_vec().iter().zip(param.data.to_vec().iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
        }
    }

    // ── End-to-end training step ──────────────────────────────────────────

    #[test]
    fn test_e2e_training_step() {
        // One forward + backward + verify loss decreases.
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(4, 1)),
        ]);

        let x = Variable::new(
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], &[2, 2]),
            false,
        );
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32, 1.0], &[2, 1]), false);

        let pred = model.forward(&x);
        let l = loss::mse_loss(&pred, &target);
        let loss_val = l.data.to_vec()[0];
        backward(&l);

        // All parameters should have gradients after backward.
        for p in model.parameters() {
            assert!(p.grad().is_some());
        }

        // Loss should be finite.
        assert!(
            loss_val.is_finite(),
            "loss should be finite, got {loss_val}"
        );
    }
}
