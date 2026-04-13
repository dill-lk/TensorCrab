//! Loss functions for supervised learning.
//!
//! All functions accept prediction and target [`Variable`]s and return a
//! scalar `Variable` (shape `[1]`) whose `.backward()` propagates gradients
//! to the prediction.
//!
//! Target variables should be created with `requires_grad = false`.

use std::sync::Arc;

use crate::autograd::Variable;
use crate::tensor::Tensor;

// ─── MSE Loss ─────────────────────────────────────────────────────────────────

/// Mean squared error: `mean((pred − target)²)`.
///
/// Returns a scalar `Variable` of shape `[1]`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::loss::mse_loss;
///
/// let pred = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), true);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
/// let loss = mse_loss(&pred, &target);
/// assert!((loss.data().to_vec()[0]).abs() < 1e-6);
/// ```
pub fn mse_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable> {
    let n = pred.data().numel() as f32;
    // diff = pred - target,  diff² summed, then divided by n
    let diff = pred.var_sub(target);
    let diff_sq = diff.var_mul(&diff);
    diff_sq.var_sum().var_mul_scalar(1.0 / n)
}

// ─── BCE Loss ────────────────────────────────────────────────────────────────

/// Binary cross-entropy: `-mean(target * log(pred) + (1 − target) * log(1 − pred))`.
///
/// `pred` values are clamped to `[ε, 1 − ε]` (ε = 1e-7) before taking the
/// logarithm to prevent `log(0)`.
///
/// Both `pred` and `target` must have the same shape.  `target` should
/// contain values in `{0, 1}`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::loss::bce_loss;
///
/// let pred = Variable::new(Tensor::from_vec(vec![0.9_f32, 0.1], &[2]), true);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[2]), false);
/// let loss = bce_loss(&pred, &target);
/// // loss ≈ 0.105
/// assert!(loss.data().to_vec()[0] < 0.2);
/// ```
pub fn bce_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable> {
    let n = pred.data().numel() as f32;

    // log(pred)
    let log_pred = pred.var_log();
    // log(1 - pred)
    let one = Variable::new(Tensor::ones(pred.data().shape()), false);
    let one_minus_pred = one.var_sub(pred);
    let log_one_minus_pred = one_minus_pred.var_log();

    // target * log(pred) + (1 - target) * log(1 - pred)
    let one_c = Variable::new(Tensor::ones(target.data().shape()), false);
    let one_minus_target = one_c.var_sub(target);

    let term1 = target.var_mul(&log_pred);
    let term2 = one_minus_target.var_mul(&log_one_minus_pred);
    let sum_terms = term1.var_add(&term2);

    // -mean(...)
    sum_terms.var_sum().var_mul_scalar(-1.0 / n)
}

// ─── Cross-Entropy Loss ───────────────────────────────────────────────────────

/// Categorical cross-entropy: `-mean(Σ target * log_softmax(pred))`.
///
/// `pred` should contain raw logits (unnormalized scores) of shape
/// `[batch, num_classes]` or `[num_classes]`.
///
/// `target` should contain one-hot encoded class probabilities with the same
/// shape as `pred`.
///
/// Internally computes `log_softmax(pred)` in a numerically stable way:
/// `log_softmax(x) = x − log(Σ exp(x − max(x))) − max(x)`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::loss::cross_entropy_loss;
///
/// // Single sample, 3 classes. True class is 1 (one-hot: [0, 1, 0]).
/// let logits = Variable::new(Tensor::from_vec(vec![0.1_f32, 2.0, 0.5], &[3]), true);
/// let target = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0, 0.0], &[3]), false);
/// let loss = cross_entropy_loss(&logits, &target);
/// // loss should be small (correct class has highest logit)
/// assert!(loss.data().to_vec()[0] < 0.5);
/// ```
pub fn cross_entropy_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable> {
    // Numerically stable log-softmax:
    //   log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    let max_val = pred.data().max().to_vec()[0];
    let max_const = Variable::new(Tensor::full(max_val, pred.data().shape()), false);
    let x_shifted = pred.var_sub(&max_const);

    let exp_x = x_shifted.var_exp();
    let sum_exp = exp_x.var_sum_keepdim(pred.data().ndim() - 1);
    let log_sum_exp = sum_exp.var_log();

    // log_softmax = x_shifted - log_sum_exp  (broadcast)
    let log_softmax = x_shifted.var_sub(&log_sum_exp);

    // cross-entropy = -mean(sum(target * log_softmax, dim=-1))
    let n = pred.data().numel() as f32;
    target
        .var_mul(&log_softmax)
        .var_sum()
        .var_mul_scalar(-1.0 / n)
}
