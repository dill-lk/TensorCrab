use std::sync::{Arc, Mutex};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::{filter_trainable, Optimizer};

/// Adam optimizer — Adaptive Moment Estimation.
///
/// Maintains per-parameter first and second moment estimates for adaptive
/// learning rates.
///
/// ## Update rule
///
/// For each parameter with gradient `g` at step `t`:
/// ```text
/// m  ← β₁ * m + (1 − β₁) * g
/// v  ← β₂ * v + (1 − β₂) * g²
/// m̂  = m / (1 − β₁ᵗ)       (bias correction)
/// v̂  = v / (1 − β₂ᵗ)       (bias correction)
/// param ← param − lr * m̂ / (√v̂ + ε)
/// ```
///
/// Optional L2 regularisation (`weight_decay`) is added to the gradient
/// before the moment updates (coupled weight decay — use [`crate::optim::AdamW`] for the
/// decoupled variant).
///
/// Default hyper-parameters follow the original paper:
/// β₁ = 0.9, β₂ = 0.999, ε = 1e-8.
///
/// # Example
/// ```rust
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::{Module, Sequential, Linear, loss};
/// use tensor_crab::optim::{Optimizer, Adam};
///
/// let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
/// let mut opt = Adam::new(model.parameters(), 0.001);
///
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
/// let pred = model.forward(&x);
/// let l = loss::mse_loss(&pred, &target);
/// backward(&l);
/// opt.step();
/// opt.zero_grad();
/// ```
pub struct Adam {
    /// Trainable parameters to optimise.
    params: Vec<Arc<Variable>>,
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay rate for the first moment estimate.
    pub beta1: f32,
    /// Exponential decay rate for the second moment estimate.
    pub beta2: f32,
    /// Small constant for numerical stability.
    pub eps: f32,
    /// L2 regularisation coefficient (0 = no weight decay).
    pub weight_decay: f32,
    /// First moment estimates (mean of gradients), one per parameter.
    m: Vec<Mutex<Option<Tensor>>>,
    /// Second moment estimates (uncentered variance), one per parameter.
    v: Vec<Mutex<Option<Tensor>>>,
    /// Global step count (shared across all parameters in this optimizer).
    t: Mutex<usize>,
}

impl Adam {
    /// Creates a new `Adam` optimizer with default β₁ = 0.9, β₂ = 0.999, ε = 1e-8.
    pub fn new(params: Vec<Arc<Variable>>, lr: f32) -> Self {
        let params = filter_trainable(params);
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: (0..n).map(|_| Mutex::new(None)).collect(),
            v: (0..n).map(|_| Mutex::new(None)).collect(),
            t: Mutex::new(0),
        }
    }

    /// Sets β₁ and returns `self`.
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets β₂ and returns `self`.
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets ε and returns `self`.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the weight decay (coupled L2 regularisation) and returns `self`.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        // Increment global step.
        let t = {
            let mut guard = self.t.lock().expect("Adam: step counter mutex poisoned");
            *guard += 1;
            *guard
        };

        let bc1 = 1.0 - self.beta1.powi(t as i32);
        let bc2 = 1.0 - self.beta2.powi(t as i32);

        for ((param, m_cell), v_cell) in self.params.iter().zip(self.m.iter()).zip(self.v.iter()) {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            // Incorporate coupled weight decay into the gradient.
            let g = if self.weight_decay != 0.0 {
                let wd = param.data().mul_scalar(self.weight_decay);
                grad.add(&wd)
                    .expect("Adam: weight decay add failed — shape mismatch")
            } else {
                grad
            };

            // First moment: m = β₁ * m + (1 − β₁) * g
            let mut m_guard = m_cell.lock().expect("Adam: m mutex poisoned");
            let new_m = match m_guard.take() {
                None => g.mul_scalar(1.0 - self.beta1),
                Some(prev_m) => prev_m
                    .mul_scalar(self.beta1)
                    .add(&g.mul_scalar(1.0 - self.beta1))
                    .expect("Adam: first moment update failed"),
            };
            *m_guard = Some(new_m.clone());

            // Second moment: v = β₂ * v + (1 − β₂) * g²
            let g_sq = g.mul(&g).expect("Adam: g² failed");
            let mut v_guard = v_cell.lock().expect("Adam: v mutex poisoned");
            let new_v = match v_guard.take() {
                None => g_sq.mul_scalar(1.0 - self.beta2),
                Some(prev_v) => prev_v
                    .mul_scalar(self.beta2)
                    .add(&g_sq.mul_scalar(1.0 - self.beta2))
                    .expect("Adam: second moment update failed"),
            };
            *v_guard = Some(new_v.clone());

            // Bias-corrected estimates.
            let m_hat = new_m.mul_scalar(1.0 / bc1);
            let v_hat = new_v.mul_scalar(1.0 / bc2);

            // Parameter update: param ← param − lr * m̂ / (√v̂ + ε)
            let denom_data: Vec<f32> = v_hat
                .to_vec()
                .into_iter()
                .map(|v| v.sqrt() + self.eps)
                .collect();
            let denom = Tensor::from_vec(denom_data, v_hat.shape());
            let update = m_hat
                .div(&denom)
                .expect("Adam: m̂/denom failed")
                .mul_scalar(self.lr);
            let p = param.data().clone();
            let new_p = p
                .sub(&update)
                .expect("Adam: parameter update failed — shape mismatch");
            param.set_data(new_p);
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::nn::{loss, Linear, Module, Sequential};

    #[test]
    fn test_adam_decreases_loss() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

        let pred0 = model.forward(&x);
        let l0 = loss::mse_loss(&pred0, &target);
        let loss0 = l0.data().to_vec()[0];
        backward(&l0);
        opt.step();
        opt.zero_grad();

        let pred1 = model.forward(&x);
        let l1 = loss::mse_loss(&pred1, &target);
        let loss1 = l1.data().to_vec()[0];

        assert!(
            loss1 < loss0,
            "Adam step should decrease loss: {loss0} → {loss1}"
        );
    }

    #[test]
    fn test_adam_multi_step_convergence() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(crate::nn::ReLU::new()),
            Box::new(Linear::new(4, 1)),
        ]);
        let mut opt = Adam::new(model.parameters(), 0.01);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

        let pred0 = model.forward(&x);
        let loss_start = loss::mse_loss(&pred0, &target).data().to_vec()[0];
        backward(&loss::mse_loss(&pred0, &target));
        opt.step();
        opt.zero_grad();

        for _ in 1..30 {
            let pred = model.forward(&x);
            let l = loss::mse_loss(&pred, &target);
            backward(&l);
            opt.step();
            opt.zero_grad();
        }

        let pred_final = model.forward(&x);
        let loss_end = loss::mse_loss(&pred_final, &target).data().to_vec()[0];
        assert!(
            loss_end < loss_start,
            "Adam: loss should decrease after 30 steps: {loss_start} → {loss_end}"
        );
    }

    #[test]
    fn test_adam_step_counter_increments() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = Adam::new(model.parameters(), 0.001);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

        for _ in 0..5 {
            let pred = model.forward(&x);
            let l = loss::mse_loss(&pred, &target);
            backward(&l);
            opt.step();
            opt.zero_grad();
        }

        let t = *opt.t.lock().unwrap();
        assert_eq!(t, 5, "Step counter should be 5 after 5 steps");
    }
}
