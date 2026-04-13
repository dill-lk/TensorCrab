//! Learning rate schedulers.
//!
//! Schedulers compute a learning rate value for a given epoch/step number.
//! They do **not** modify the optimizer directly; instead the caller reads
//! the new learning rate and updates `optimizer.lr`:
//!
//! ```rust
//! use tensor_crab::nn::{Module, Sequential, Linear};
//! use tensor_crab::optim::{Optimizer, SGD};
//! use tensor_crab::optim::scheduler::StepLR;
//!
//! let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
//! let mut opt = SGD::new(model.parameters(), 0.1);
//! let scheduler = StepLR::new(0.1, 10, 0.1);
//!
//! for epoch in 0..30 {
//!     opt.lr = scheduler.get_lr(epoch);
//!     // … training code …
//! }
//! ```

/// Step learning rate scheduler.
///
/// Multiplies the base learning rate by `gamma` every `step_size` epochs,
/// giving a staircase decay schedule:
///
/// ```text
/// lr(epoch) = base_lr * gamma ^ floor(epoch / step_size)
/// ```
///
/// # Example
/// ```
/// use tensor_crab::optim::scheduler::StepLR;
///
/// let s = StepLR::new(0.1, 10, 0.5);
/// assert!((s.get_lr(0) - 0.1).abs() < 1e-6);
/// assert!((s.get_lr(9) - 0.1).abs() < 1e-6);
/// assert!((s.get_lr(10) - 0.05).abs() < 1e-6);
/// assert!((s.get_lr(20) - 0.025).abs() < 1e-6);
/// ```
pub struct StepLR {
    /// The initial learning rate.
    pub base_lr: f32,
    /// Number of epochs between each decay step.
    pub step_size: usize,
    /// Multiplicative decay factor applied every `step_size` epochs.
    pub gamma: f32,
}

impl StepLR {
    /// Creates a new `StepLR` scheduler.
    ///
    /// # Arguments
    /// * `base_lr`   — starting learning rate.
    /// * `step_size` — number of epochs before each decay.
    /// * `gamma`     — multiplicative decay factor (typically < 1).
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        assert!(step_size > 0, "StepLR: step_size must be > 0");
        Self {
            base_lr,
            step_size,
            gamma,
        }
    }

    /// Returns the learning rate for `epoch` (0-indexed).
    pub fn get_lr(&self, epoch: usize) -> f32 {
        self.base_lr * self.gamma.powi((epoch / self.step_size) as i32)
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Smoothly decreases the learning rate from `lr_max` to `lr_min` following a
/// cosine curve over `t_max` epochs, then resets:
///
/// ```text
/// lr(epoch) = lr_min + 0.5 * (lr_max − lr_min) * (1 + cos(π * (epoch mod t_max) / t_max))
/// ```
///
/// # Example
/// ```
/// use tensor_crab::optim::scheduler::CosineAnnealingLR;
///
/// let s = CosineAnnealingLR::new(0.1, 0.0, 10);
/// // At epoch 0 the lr is at its maximum.
/// assert!((s.get_lr(0) - 0.1).abs() < 1e-5);
/// // At epoch 10 the lr has completed one cycle and resets to lr_max.
/// assert!((s.get_lr(10) - 0.1).abs() < 1e-5);
/// ```
pub struct CosineAnnealingLR {
    /// Maximum (initial) learning rate.
    pub lr_max: f32,
    /// Minimum learning rate at the trough.
    pub lr_min: f32,
    /// Number of epochs per cosine half-cycle.
    pub t_max: usize,
}

impl CosineAnnealingLR {
    /// Creates a new `CosineAnnealingLR` scheduler.
    ///
    /// # Arguments
    /// * `lr_max` — peak learning rate.
    /// * `lr_min` — floor learning rate.
    /// * `t_max`  — epochs per cosine half-period.
    pub fn new(lr_max: f32, lr_min: f32, t_max: usize) -> Self {
        assert!(t_max > 0, "CosineAnnealingLR: t_max must be > 0");
        assert!(
            lr_max >= lr_min,
            "CosineAnnealingLR: lr_max must be ≥ lr_min"
        );
        Self {
            lr_max,
            lr_min,
            t_max,
        }
    }

    /// Returns the learning rate for `epoch` (0-indexed).
    pub fn get_lr(&self, epoch: usize) -> f32 {
        let t = (epoch % self.t_max) as f32;
        let cos_val = (std::f32::consts::PI * t / self.t_max as f32).cos();
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos_val)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    // ── StepLR ────────────────────────────────────────────────────────────

    #[test]
    fn test_step_lr_initial() {
        let s = StepLR::new(0.1, 10, 0.5);
        assert_abs_diff_eq!(s.get_lr(0), 0.1_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(s.get_lr(9), 0.1_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_step_lr_first_decay() {
        let s = StepLR::new(0.1, 10, 0.5);
        assert_abs_diff_eq!(s.get_lr(10), 0.05_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(s.get_lr(19), 0.05_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_step_lr_multiple_decays() {
        let s = StepLR::new(1.0, 5, 0.1);
        assert_abs_diff_eq!(s.get_lr(0), 1.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(s.get_lr(5), 0.1_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(s.get_lr(10), 0.01_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_step_lr_with_optimizer() {
        use crate::autograd::{backward, Variable};
        use crate::nn::loss;
        use crate::nn::{Linear, Module, Sequential};
        use crate::optim::{Optimizer, SGD};
        use crate::tensor::Tensor;

        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = SGD::new(model.parameters(), 0.1);
        let scheduler = StepLR::new(0.1, 2, 0.5);

        for epoch in 0..6 {
            opt.lr = scheduler.get_lr(epoch);
            let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2]), false);
            let t = Variable::new(Tensor::from_vec(vec![1.0_f32], &[1, 1]), false);
            let pred = model.forward(&x);
            let l = loss::mse_loss(&pred, &t);
            backward(&l);
            opt.step();
            opt.zero_grad();
        }
        // lr at epoch 4 should be 0.1 * 0.5^2 = 0.025
        assert_abs_diff_eq!(scheduler.get_lr(4), 0.025_f32, epsilon = 1e-6);
    }

    // ── CosineAnnealingLR ─────────────────────────────────────────────────

    #[test]
    fn test_cosine_at_zero() {
        let s = CosineAnnealingLR::new(0.1, 0.0, 10);
        assert_abs_diff_eq!(s.get_lr(0), 0.1_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_at_half_period() {
        // At t_max the cosine returns to lr_max (cos(π * t_max / t_max) = cos(π) = -1
        // → lr_min + 0.5*(lr_max - lr_min)*(1 + (-1)) = lr_min)
        // Wait: at epoch = t_max we have t = 0 (because epoch % t_max = 0), so lr = lr_max.
        // The midpoint is at epoch = t_max / 2.
        let s = CosineAnnealingLR::new(1.0, 0.0, 10);
        // At epoch 5: cos(π * 5 / 10) = cos(π/2) = 0 → lr = 0 + 0.5 * 1 * (1 + 0) = 0.5
        assert_abs_diff_eq!(s.get_lr(5), 0.5_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_at_period_end() {
        // At epoch == t_max, t % t_max = 0 → same as epoch 0 → lr = lr_max.
        let s = CosineAnnealingLR::new(0.1, 0.0, 10);
        assert_abs_diff_eq!(s.get_lr(10), 0.1_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_never_below_lr_min() {
        let s = CosineAnnealingLR::new(0.1, 0.001, 20);
        for epoch in 0..100 {
            let lr = s.get_lr(epoch);
            assert!(
                lr >= s.lr_min - 1e-6,
                "lr {lr} fell below lr_min {} at epoch {epoch}",
                s.lr_min
            );
            assert!(
                lr <= s.lr_max + 1e-6,
                "lr {lr} exceeded lr_max {} at epoch {epoch}",
                s.lr_max
            );
        }
    }
}
