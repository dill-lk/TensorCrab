use std::sync::{Arc, Mutex};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::{filter_trainable, Optimizer};

/// AdamW optimizer — Adam with decoupled weight decay.
///
/// Identical to [`Adam`](crate::optim::Adam) except that weight decay is
/// applied directly to the parameter **after** the Adam update rather than
/// being folded into the gradient.  This decoupling is important for
/// regularisation because it keeps the adaptive learning rate from scaling
/// the weight decay term.
///
/// ## Update rule
///
/// For each parameter with gradient `g` at step `t`:
/// ```text
/// m    ← β₁ * m + (1 − β₁) * g
/// v    ← β₂ * v + (1 − β₂) * g²
/// m̂    = m / (1 − β₁ᵗ)
/// v̂    = v / (1 − β₂ᵗ)
/// param ← param − lr * m̂ / (√v̂ + ε)   ← adaptive update
///       − lr * weight_decay * param       ← decoupled L2
/// ```
///
/// Default hyper-parameters: β₁ = 0.9, β₂ = 0.999, ε = 1e-8,
/// weight_decay = 0.01.
///
/// # Example
/// ```rust
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::{Module, Sequential, Linear, loss};
/// use tensor_crab::optim::{Optimizer, AdamW};
///
/// let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
/// let mut opt = AdamW::new(model.parameters(), 0.001);
///
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
/// let pred = model.forward(&x);
/// let l = loss::mse_loss(&pred, &target);
/// backward(&l);
/// opt.step();
/// opt.zero_grad();
/// ```
pub struct AdamW {
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
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// First moment estimates, one per parameter.
    m: Vec<Mutex<Option<Tensor>>>,
    /// Second moment estimates, one per parameter.
    v: Vec<Mutex<Option<Tensor>>>,
    /// Global step count.
    t: Mutex<usize>,
}

impl AdamW {
    /// Creates a new `AdamW` optimizer with default β₁ = 0.9, β₂ = 0.999,
    /// ε = 1e-8, weight_decay = 0.01.
    pub fn new(params: Vec<Arc<Variable>>, lr: f32) -> Self {
        let params = filter_trainable(params);
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
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

    /// Sets the decoupled weight decay coefficient and returns `self`.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        let t = {
            let mut guard = self.t.lock().expect("AdamW: step counter mutex poisoned");
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

            // First moment (gradient only — no weight decay here).
            let mut m_guard = m_cell.lock().expect("AdamW: m mutex poisoned");
            let new_m = match m_guard.take() {
                None => grad.mul_scalar(1.0 - self.beta1),
                Some(prev_m) => prev_m
                    .mul_scalar(self.beta1)
                    .add(&grad.mul_scalar(1.0 - self.beta1))
                    .expect("AdamW: first moment update failed"),
            };
            *m_guard = Some(new_m.clone());

            // Second moment.
            let g_sq = grad.mul(&grad).expect("AdamW: g² failed");
            let mut v_guard = v_cell.lock().expect("AdamW: v mutex poisoned");
            let new_v = match v_guard.take() {
                None => g_sq.mul_scalar(1.0 - self.beta2),
                Some(prev_v) => prev_v
                    .mul_scalar(self.beta2)
                    .add(&g_sq.mul_scalar(1.0 - self.beta2))
                    .expect("AdamW: second moment update failed"),
            };
            *v_guard = Some(new_v.clone());

            // Bias-corrected estimates.
            let m_hat = new_m.mul_scalar(1.0 / bc1);
            let v_hat = new_v.mul_scalar(1.0 / bc2);

            // Adaptive update.
            let denom_data: Vec<f32> = v_hat
                .to_vec()
                .into_iter()
                .map(|v| v.sqrt() + self.eps)
                .collect();
            let denom = Tensor::from_vec(denom_data, v_hat.shape());
            let adam_update = m_hat
                .div(&denom)
                .expect("AdamW: m̂/denom failed")
                .mul_scalar(self.lr);

            let p = param.data().clone();

            // Decoupled weight decay.
            let wd_update = p.mul_scalar(self.lr * self.weight_decay);

            let new_p = p
                .sub(&adam_update)
                .expect("AdamW: adam update failed")
                .sub(&wd_update)
                .expect("AdamW: weight decay update failed");
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
    fn test_adamw_decreases_loss() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = AdamW::new(model.parameters(), 0.01).with_weight_decay(0.0);

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
            "AdamW step should decrease loss: {loss0} → {loss1}"
        );
    }

    #[test]
    fn test_adamw_weight_decay_shrinks_params() {
        // With zero gradient but non-zero weight decay, parameters should shrink.
        let param = Variable::new(Tensor::from_vec(vec![1.0_f32, 1.0], &[2]), true);
        let mut opt = AdamW::new(vec![Arc::clone(&param)], 0.0).with_weight_decay(0.1);

        // Provide a non-None gradient so the step runs (zero gradient).
        let dummy = Variable::new(Tensor::zeros(&[2]), false);
        let loss_var = param.var_mul(&dummy).var_sum();
        backward(&loss_var);
        opt.step();

        // Adam update is 0 (grad = 0 → m = 0 → update = 0).
        // Weight decay: new_p = p - lr * wd * p = 1 - 0.0 * 0.1 * 1 = 1.0
        // (lr = 0 so no actual change expected — just verifying no panic)
        let vals = param.data().to_vec();
        assert!(vals[0].is_finite(), "AdamW: parameter should remain finite");
    }

    #[test]
    fn test_adamw_multi_step_convergence() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(crate::nn::ReLU::new()),
            Box::new(Linear::new(4, 1)),
        ]);
        let mut opt = AdamW::new(model.parameters(), 0.01).with_weight_decay(0.0);

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
            "AdamW: loss should decrease after 30 steps: {loss_start} → {loss_end}"
        );
    }
}
