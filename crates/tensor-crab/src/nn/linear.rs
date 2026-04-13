use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── Weight initialisation helpers ───────────────────────────────────────────

/// Xavier (Glorot) uniform initialisation.
///
/// Samples from U(−a, a) where a = √(6 / (fan_in + fan_out)).
///
/// Recommended for tanh / sigmoid activations.
fn xavier_uniform(in_features: usize, out_features: usize, seed: u64) -> Tensor {
    let mut rng = SmallRng::seed_from_u64(seed);
    let limit = ((6.0_f32) / (in_features + out_features) as f32).sqrt();
    let numel = in_features * out_features;
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-limit..=limit)).collect();
    Tensor::from_vec(data, &[out_features, in_features])
}

/// Kaiming (He) uniform initialisation.
///
/// Samples from U(−a, a) where a = √(6 / fan_in).
///
/// Recommended for ReLU activations.
fn kaiming_uniform(in_features: usize, out_features: usize, seed: u64) -> Tensor {
    let mut rng = SmallRng::seed_from_u64(seed);
    let limit = (6.0_f32 / in_features as f32).sqrt();
    let numel = in_features * out_features;
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-limit..=limit)).collect();
    Tensor::from_vec(data, &[out_features, in_features])
}

// ─── Linear layer ────────────────────────────────────────────────────────────

/// Fully-connected linear layer: `output = input @ weightᵀ + bias`.
///
/// Weight shape: `[out_features, in_features]`  
/// Bias shape: `[out_features]`
///
/// By default weights are initialised with **Xavier uniform** and bias is
/// zeroed.  Use [`Linear::new_kaiming`] for ReLU-focused networks.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Linear};
///
/// let layer = Linear::new(3, 2);
/// let x = Variable::new(Tensor::randn_seeded(&[4, 3], 0), false);
/// let y = layer.forward(&x);
/// assert_eq!(y.data.shape(), &[4, 2]);
/// ```
pub struct Linear {
    /// Weight matrix of shape `[out_features, in_features]`.
    pub weight: Arc<Variable>,
    /// Bias vector of shape `[out_features]`.
    pub bias: Arc<Variable>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl Linear {
    /// Creates a new `Linear` layer with **Xavier uniform** weight init.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight_data = xavier_uniform(in_features, out_features, 0);
        let bias_data = Tensor::zeros(&[out_features]);
        Self {
            weight: Variable::new(weight_data, true),
            bias: Variable::new(bias_data, true),
            in_features,
            out_features,
        }
    }

    /// Creates a new `Linear` layer with **Kaiming uniform** weight init.
    ///
    /// Preferred when the layer is followed by a ReLU activation.
    pub fn new_kaiming(in_features: usize, out_features: usize) -> Self {
        let weight_data = kaiming_uniform(in_features, out_features, 0);
        let bias_data = Tensor::zeros(&[out_features]);
        Self {
            weight: Variable::new(weight_data, true),
            bias: Variable::new(bias_data, true),
            in_features,
            out_features,
        }
    }

    /// Creates a `Linear` layer from pre-existing weight and bias tensors.
    ///
    /// Both parameters are marked `requires_grad = true`.  Useful when loading
    /// saved weights.
    pub fn from_weights(weight: Tensor, bias: Tensor) -> Self {
        let in_features = weight.shape()[1];
        let out_features = weight.shape()[0];
        Self {
            weight: Variable::new(weight, true),
            bias: Variable::new(bias, true),
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    /// Forward pass: `output = input @ weightᵀ + bias`.
    ///
    /// - `input`: `[batch, in_features]`
    /// - output: `[batch, out_features]`
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        // weight.T has shape [in_features, out_features]
        let wt = self.weight.var_transpose();
        // [batch, in_features] @ [in_features, out_features] → [batch, out_features]
        let xw = input.var_matmul(&wt);
        // bias [out_features] is broadcast over the batch dimension.
        xw.var_add(&self.bias)
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        vec![Arc::clone(&self.weight), Arc::clone(&self.bias)]
    }
}
