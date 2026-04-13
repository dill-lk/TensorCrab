use std::sync::Arc;

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── ReLU ────────────────────────────────────────────────────────────────────

/// ReLU activation: `max(0, x)`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, ReLU};
///
/// let relu = ReLU::new();
/// let x = Variable::new(Tensor::from_vec(vec![-1.0_f32, 2.0], &[2]), false);
/// let y = relu.forward(&x);
/// assert_eq!(y.data().to_vec(), vec![0.0, 2.0]);
/// ```
pub struct ReLU;

impl ReLU {
    /// Creates a new `ReLU` activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        input.var_relu()
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

/// Sigmoid activation: `σ(x) = 1 / (1 + exp(−x))`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Sigmoid};
///
/// let sigmoid = Sigmoid::new();
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f32], &[1]), false);
/// let y = sigmoid.forward(&x);
/// assert!((y.data().to_vec()[0] - 0.5).abs() < 1e-6);
/// ```
pub struct Sigmoid;

impl Sigmoid {
    /// Creates a new `Sigmoid` activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        input.var_sigmoid()
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}

// ─── Tanh ────────────────────────────────────────────────────────────────────

/// Tanh activation: `tanh(x)`.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Tanh};
///
/// let tanh = Tanh::new();
/// let x = Variable::new(Tensor::from_vec(vec![0.0_f32], &[1]), false);
/// let y = tanh.forward(&x);
/// assert!((y.data().to_vec()[0]).abs() < 1e-6);
/// ```
pub struct Tanh;

impl Tanh {
    /// Creates a new `Tanh` activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        input.var_tanh()
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a chosen axis.
///
/// `softmax(x, dim)[i] = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))`
///
/// The subtraction of the per-sample max ensures numerical stability without
/// affecting the output (the constant cancels in numerator and denominator).
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Softmax};
///
/// let softmax = Softmax::new(0);
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
/// let y = softmax.forward(&x);
/// let probs: f32 = y.data().to_vec().iter().sum();
/// assert!((probs - 1.0).abs() < 1e-5);
/// ```
pub struct Softmax {
    /// The axis over which the softmax normalisation is applied.
    pub dim: usize,
}

impl Softmax {
    /// Creates a new `Softmax` module that normalises over `dim`.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        // Subtract the max along `dim` for numerical stability.
        // The max is treated as a constant (no gradient needed).
        let max_val = input.data().max().to_vec()[0];
        let max_tensor = Tensor::full(max_val, input.data().shape());
        let x_shifted = input.var_sub(&Variable::new(max_tensor, false));

        let exp_x = x_shifted.var_exp();
        // sum_exp: same shape as input but with dim = 1 (keepdim).
        let sum_exp = exp_x.var_sum_keepdim(self.dim);
        // Divide broadcasts sum_exp back to input shape.
        exp_x.var_div(&sum_exp)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }
}
