use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── Dropout ──────────────────────────────────────────────────────────────────

/// Randomly zeroes a fraction of the input elements during training.
///
/// During forward in training mode, each element is kept with probability
/// `1 − p` and the surviving elements are scaled by `1 / (1 − p)` so the
/// expected value is preserved (inverted dropout).
///
/// In eval mode the layer is a no-op (identity function).
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Dropout};
///
/// // In eval mode Dropout is an identity.
/// let dropout = Dropout::new(0.5);
/// dropout.set_training(false);
/// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]), false);
/// let y = dropout.forward(&x);
/// assert_eq!(y.data.to_vec(), x.data.to_vec());
/// ```
pub struct Dropout {
    /// Probability of zeroing an element (0 ≤ p < 1).
    pub p: f32,
    training: AtomicBool,
    rng: Mutex<SmallRng>,
}

impl Dropout {
    /// Creates a new `Dropout` with drop probability `p`.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout: p must be in [0, 1), got {p}"
        );
        Self {
            p,
            training: AtomicBool::new(true),
            rng: Mutex::new(SmallRng::seed_from_u64(0)),
        }
    }

    /// Creates a `Dropout` with a specific RNG seed (useful for testing).
    pub fn with_seed(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout: p must be in [0, 1), got {p}"
        );
        Self {
            p,
            training: AtomicBool::new(true),
            rng: Mutex::new(SmallRng::seed_from_u64(seed)),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return Arc::clone(input);
        }

        let numel = input.data.numel();
        let scale = 1.0 / (1.0 - self.p);

        // Generate a binary mask: 1 with prob (1-p), 0 with prob p.
        let mask_data: Vec<f32> = {
            let mut rng = self.rng.lock().expect("Dropout: rng mutex poisoned");
            (0..numel)
                .map(|_| {
                    let r: f32 = rng.gen();
                    if r >= self.p {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        let mask = Variable::new(Tensor::from_vec(mask_data, input.data.shape()), false);
        input.var_mul(&mask)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![]
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
