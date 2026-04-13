use std::sync::{Arc, Mutex};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::{filter_trainable, Optimizer};

/// Stochastic gradient descent with optional momentum and weight decay.
///
/// ## Update rule
///
/// Without momentum (`momentum = 0`):
/// ```text
/// param ← param − lr * (grad + weight_decay * param)
/// ```
///
/// With momentum (`momentum > 0`), a velocity buffer `v` is maintained:
/// ```text
/// v    ← momentum * v + (grad + weight_decay * param)
/// param ← param − lr * v
/// ```
///
/// # Example
/// ```rust
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::{Variable, backward};
/// use tensor_crab::nn::{Module, Sequential, Linear, loss};
/// use tensor_crab::optim::{Optimizer, SGD};
///
/// let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
/// let mut opt = SGD::new(model.parameters(), 0.01);
///
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
/// let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
/// let pred = model.forward(&x);
/// let l = loss::mse_loss(&pred, &target);
/// backward(&l);
/// opt.step();
/// opt.zero_grad();
/// ```
pub struct SGD {
    /// Trainable parameters to optimise.
    params: Vec<Arc<Variable>>,
    /// Learning rate.
    pub lr: f32,
    /// Momentum factor (0 = no momentum).
    pub momentum: f32,
    /// L2 regularisation coefficient (0 = no weight decay).
    pub weight_decay: f32,
    /// Velocity buffers, one per parameter.
    velocities: Vec<Mutex<Option<Tensor>>>,
}

impl SGD {
    /// Creates a new `SGD` optimizer with the given learning rate.
    ///
    /// Momentum and weight decay default to 0.
    pub fn new(params: Vec<Arc<Variable>>, lr: f32) -> Self {
        let params = filter_trainable(params);
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: (0..n).map(|_| Mutex::new(None)).collect(),
        }
    }

    /// Sets the momentum factor and returns `self` for builder-style chaining.
    ///
    /// # Panics
    /// Panics if `momentum` is not in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&momentum),
            "SGD: momentum must be in [0, 1), got {momentum}"
        );
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay (L2 regularisation) and returns `self`.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        assert!(
            weight_decay >= 0.0,
            "SGD: weight_decay must be ≥ 0, got {weight_decay}"
        );
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (param, vel_cell) in self.params.iter().zip(self.velocities.iter()) {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let p = param.data().clone();

            // Incorporate weight decay into the effective gradient.
            let effective_grad = if self.weight_decay != 0.0 {
                let wd_term = p.mul_scalar(self.weight_decay);
                grad.add(&wd_term)
                    .expect("SGD: weight decay add failed — shape mismatch")
            } else {
                grad
            };

            // Update velocity and compute parameter delta.
            let delta = if self.momentum != 0.0 {
                let mut vel_guard = vel_cell.lock().expect("SGD: velocity mutex poisoned");
                let new_vel = match vel_guard.take() {
                    None => effective_grad,
                    Some(v) => v
                        .mul_scalar(self.momentum)
                        .add(&effective_grad)
                        .expect("SGD: velocity update failed — shape mismatch"),
                };
                *vel_guard = Some(new_vel.clone());
                new_vel
            } else {
                effective_grad
            };

            let new_p = p
                .sub(&delta.mul_scalar(self.lr))
                .expect("SGD: parameter update failed — shape mismatch");
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
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::autograd::backward;
    use crate::nn::{loss, Linear, Module, Sequential};

    #[test]
    fn test_sgd_decreases_loss() {
        // Simple regression: learn to map [1, 0] → 1.
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = SGD::new(model.parameters(), 0.1);

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
            "SGD step should decrease loss: {loss0} → {loss1}"
        );
    }

    #[test]
    fn test_sgd_with_momentum() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = SGD::new(model.parameters(), 0.05).with_momentum(0.9);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

        // Two steps with momentum should converge faster than without.
        let pred0 = model.forward(&x);
        let l0 = loss::mse_loss(&pred0, &target);
        let loss0 = l0.data().to_vec()[0];
        backward(&l0);
        opt.step();
        opt.zero_grad();

        let pred1 = model.forward(&x);
        let l1 = loss::mse_loss(&pred1, &target);
        backward(&l1);
        opt.step();
        opt.zero_grad();

        let pred2 = model.forward(&x);
        let l2 = loss::mse_loss(&pred2, &target);
        let loss2 = l2.data().to_vec()[0];

        assert!(
            loss2 < loss0,
            "SGD+momentum should decrease loss over 2 steps"
        );
    }

    #[test]
    fn test_sgd_zero_grad_clears_gradients() {
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = SGD::new(model.parameters(), 0.01);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
        let pred = model.forward(&x);
        let l = loss::mse_loss(&pred, &target);
        backward(&l);

        for p in model.parameters() {
            assert!(p.grad().is_some(), "grad should exist before zero_grad");
        }
        opt.step();
        opt.zero_grad();
        for p in model.parameters() {
            assert!(p.grad().is_none(), "grad should be None after zero_grad");
        }
    }

    #[test]
    fn test_sgd_weight_decay() {
        // With large weight decay, parameters should move toward zero.
        let param = Variable::new(Tensor::from_vec(vec![2.0_f32], &[1]), true);
        let mut opt = SGD::new(vec![Arc::clone(&param)], 0.1).with_weight_decay(0.5);

        // Manually set gradient to 0 so that only weight decay contributes.
        // We do this by doing a forward that produces zero gradient.
        let zero_grad = Variable::new(Tensor::zeros(&[1]), false);
        // loss = 0 * param → grad = 0; weight decay still applies
        let loss_var = param.var_mul(&zero_grad).var_sum();
        backward(&loss_var);
        // grad is 0 but weight decay adds weight_decay * param = 0.5 * 2 = 1.
        // new_param = 2 - 0.1 * (0 + 0.5 * 2) = 2 - 0.1 = 1.9
        opt.step();
        let new_val = param.data().to_vec()[0];
        assert_abs_diff_eq!(new_val, 1.9_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_sgd_multi_step_convergence() {
        // XOR-like regression: verify loss decreases over 50 steps.
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)),
            Box::new(crate::nn::ReLU::new()),
            Box::new(Linear::new(4, 1)),
        ]);
        let mut opt = SGD::new(model.parameters(), 0.05);

        let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);

        let pred0 = model.forward(&x);
        let loss_start = loss::mse_loss(&pred0, &target).data().to_vec()[0];
        backward(&loss::mse_loss(&pred0, &target));
        opt.step();
        opt.zero_grad();

        for _ in 1..50 {
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
            "loss should decrease after 50 SGD steps: {loss_start} → {loss_end}"
        );
    }
}
