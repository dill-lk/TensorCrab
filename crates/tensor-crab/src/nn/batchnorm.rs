use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── BatchNorm1d ──────────────────────────────────────────────────────────────

/// Batch normalisation for 2-D inputs `[batch, num_features]`.
///
/// Normalises each feature to zero mean and unit variance over the batch,
/// then applies learnable scale (γ) and shift (β) parameters:
///
/// ```text
/// y = γ * (x − μ) / √(σ² + ε) + β
/// ```
///
/// - In **training mode** the batch mean (μ) and variance (σ²) are computed
///   from the current mini-batch, and running estimates are updated with
///   exponential moving average (momentum 0.1 by default).
/// - In **eval mode** the stored running mean/variance are used instead.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, BatchNorm1d};
///
/// let bn = BatchNorm1d::new(4);
/// let x = Variable::new(Tensor::randn_seeded(&[8, 4], 42), false);
/// let y = bn.forward(&x);
/// assert_eq!(y.data.shape(), &[8, 4]);
/// ```
pub struct BatchNorm1d {
    /// Learnable scale parameter γ, shape `[num_features]`.
    pub gamma: Arc<Variable>,
    /// Learnable shift parameter β, shape `[num_features]`.
    pub beta: Arc<Variable>,
    /// Running mean estimate, shape `[num_features]`.
    pub running_mean: Mutex<Tensor>,
    /// Running variance estimate, shape `[num_features]`.
    pub running_var: Mutex<Tensor>,
    /// Small value added to the denominator for numerical stability.
    pub eps: f32,
    /// Momentum for the exponential moving average of running stats.
    pub momentum: f32,
    /// Number of features.
    pub num_features: usize,
    training: AtomicBool,
}

impl BatchNorm1d {
    /// Creates a new `BatchNorm1d` with default ε = 1e-5 and momentum = 0.1.
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Variable::new(Tensor::ones(&[num_features]), true),
            beta: Variable::new(Tensor::zeros(&[num_features]), true),
            running_mean: Mutex::new(Tensor::zeros(&[num_features])),
            running_var: Mutex::new(Tensor::ones(&[num_features])),
            eps: 1e-5,
            momentum: 0.1,
            num_features,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        let shape = input.data.shape().to_vec();
        assert_eq!(
            shape.len(),
            2,
            "BatchNorm1d: expected 2-D input [batch, features], got {:?}",
            shape
        );
        let batch = shape[0];
        let features = shape[1];
        assert_eq!(
            features,
            self.num_features,
            "BatchNorm1d: expected {n} features, got {features}",
            n = self.num_features
        );

        let n = batch as f32;

        if self.training.load(Ordering::Relaxed) {
            // ── Training mode ─────────────────────────────────────────────
            // Compute batch mean and variance using Variable ops so that
            // gradients flow through to the input.
            let sum_x = input.var_sum_keepdim(0); // [1, features]
            let mean = sum_x.var_mul_scalar(1.0 / n); // [1, features]

            let diff = input.var_sub(&mean); // [batch, features]
            let diff_sq = diff.var_mul(&diff); // [batch, features]
            let var = diff_sq.var_sum_keepdim(0).var_mul_scalar(1.0 / n); // [1, features]

            let std = var.var_add_scalar(self.eps).var_sqrt(); // [1, features]
            let x_norm = diff.var_div(&std); // [batch, features]

            // Update running stats (outside the autograd graph).
            {
                let batch_mean_vec = mean.data.to_vec();
                let batch_var_vec = var.data.to_vec();
                let m = self.momentum;

                let mut rm = self
                    .running_mean
                    .lock()
                    .expect("BatchNorm1d: running_mean mutex poisoned");
                let mut rv = self
                    .running_var
                    .lock()
                    .expect("BatchNorm1d: running_var mutex poisoned");

                let new_mean: Vec<f32> = rm
                    .to_vec()
                    .iter()
                    .zip(batch_mean_vec.iter())
                    .map(|(&r, &b)| (1.0 - m) * r + m * b)
                    .collect();
                let new_var: Vec<f32> = rv
                    .to_vec()
                    .iter()
                    .zip(batch_var_vec.iter())
                    .map(|(&r, &b)| (1.0 - m) * r + m * b)
                    .collect();

                *rm = Tensor::from_vec(new_mean, &[features]);
                *rv = Tensor::from_vec(new_var, &[features]);
            }

            // Apply γ and β (both broadcast from [features] to [batch, features]).
            x_norm.var_mul(&self.gamma).var_add(&self.beta)
        } else {
            // ── Eval mode ─────────────────────────────────────────────────
            // Normalise with running stats (constants, no gradient needed).
            let rm = self
                .running_mean
                .lock()
                .expect("BatchNorm1d: running_mean mutex poisoned")
                .clone();
            let rv = self
                .running_var
                .lock()
                .expect("BatchNorm1d: running_var mutex poisoned")
                .clone();

            let std_data: Vec<f32> = rv.to_vec().iter().map(|&v| (v + self.eps).sqrt()).collect();
            let std_tensor = Tensor::from_vec(std_data, &[features]);

            let mean_var = Variable::new(rm, false);
            let std_var = Variable::new(std_tensor, false);

            let diff = input.var_sub(&mean_var);
            let x_norm = diff.var_div(&std_var);
            x_norm.var_mul(&self.gamma).var_add(&self.beta)
        }
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![Arc::clone(&self.gamma), Arc::clone(&self.beta)]
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
