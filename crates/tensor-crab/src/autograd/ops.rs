/// Variable-level autograd operations.
///
/// Each public method on [`Variable`] corresponds to one differentiable
/// operation.  Under the hood every method:
/// 1. Computes the forward-pass result using the underlying [`Tensor`].
/// 2. (If any input `requires_grad`) builds a [`Node`] whose [`BackwardFn`]
///    knows how to compute the input gradients from the output gradient.
/// 3. Returns a new `Arc<Variable>` wrapping the result.
///
/// The backward functions follow the standard chain-rule formulas:
///
/// | Op | Forward | dL/d(lhs) | dL/d(rhs) |
/// |---|---|---|---|
/// | add | z = x + y | grad | grad |
/// | sub | z = x − y | grad | −grad |
/// | mul | z = x * y | grad * y | grad * x |
/// | matmul | z = x @ y | grad @ yᵀ | xᵀ @ grad |
/// | neg | z = −x | −grad | — |
/// | relu | z = max(0,x) | grad * (x > 0) | — |
/// | sigmoid | z = σ(x) | grad * z*(1−z) | — |
/// | log | z = ln(x) | grad / x | — |
/// | exp | z = eˣ | grad * z | — |
/// | sum | z = Σ x | broadcast(grad → x.shape) | — |
use std::sync::Arc;

use crate::tensor::Tensor;

use super::{
    graph::{BackwardFn, Node},
    Variable,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Creates a non-leaf Variable whose `requires_grad` flag is `true` iff at
/// least one of `inputs` requires grad.
fn make_output(
    data: Tensor,
    inputs: Vec<Arc<Variable>>,
    backward_fn: Box<dyn BackwardFn>,
) -> Arc<Variable> {
    let requires_grad = inputs.iter().any(|v| v.requires_grad);
    if requires_grad {
        let node = Arc::new(Node {
            backward_fn,
            inputs,
        });
        Variable::with_grad_fn(data, true, node)
    } else {
        // No gradient tracking needed — return a plain leaf-like Variable.
        Variable::new(data, false)
    }
}

// ─── Binary ops ──────────────────────────────────────────────────────────────

/// `AddBackward`: both inputs receive `grad_output` unchanged.
struct AddBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl BackwardFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            grad_output.sum_to(&self.lhs_shape),
            grad_output.sum_to(&self.rhs_shape),
        ]
    }
}

/// `SubBackward`: lhs gets `grad_output`, rhs gets `-grad_output`.
struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl BackwardFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            grad_output.sum_to(&self.lhs_shape),
            grad_output.neg().sum_to(&self.rhs_shape),
        ]
    }
}

/// `MulBackward`: saves both forward tensors for the cross-product rule.
struct MulBackward {
    lhs_data: Tensor,
    rhs_data: Tensor,
}

impl BackwardFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let lhs_shape = self.lhs_data.shape().to_vec();
        let rhs_shape = self.rhs_data.shape().to_vec();
        vec![
            grad_output
                .mul(&self.rhs_data)
                .expect("MulBackward: mul failed")
                .sum_to(&lhs_shape),
            grad_output
                .mul(&self.lhs_data)
                .expect("MulBackward: mul failed")
                .sum_to(&rhs_shape),
        ]
    }
}

/// `MatmulBackward`: saves both forward tensors.
///
/// dL/dX = dL/dZ @ Yᵀ  
/// dL/dY = Xᵀ  @ dL/dZ
struct MatmulBackward {
    lhs_data: Tensor,
    rhs_data: Tensor,
}

impl BackwardFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let rhs_t = self
            .rhs_data
            .transpose()
            .expect("MatmulBackward: transpose of rhs failed");
        let lhs_t = self
            .lhs_data
            .transpose()
            .expect("MatmulBackward: transpose of lhs failed");
        vec![
            grad_output
                .matmul(&rhs_t)
                .expect("MatmulBackward: grad @ rhs.T failed"),
            lhs_t
                .matmul(grad_output)
                .expect("MatmulBackward: lhs.T @ grad failed"),
        ]
    }
}

// ─── Unary ops ────────────────────────────────────────────────────────────────

/// `NegBackward`: grad_output is negated.
struct NegBackward;

impl BackwardFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output.neg()]
    }
}

/// `ReluBackward`: mask where forward input > 0.
struct ReluBackward {
    input_data: Tensor,
}

impl BackwardFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mask: Vec<f32> = self
            .input_data
            .to_vec()
            .into_iter()
            .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let mask_tensor = Tensor::from_vec(mask, self.input_data.shape());
        vec![grad_output
            .mul(&mask_tensor)
            .expect("ReluBackward: mul failed")]
    }
}

/// `SigmoidBackward`: saves the forward output `z = σ(x)`.
///
/// dL/dx = grad * z * (1 − z)
struct SigmoidBackward {
    output_data: Tensor,
}

impl BackwardFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let ones = Tensor::ones(self.output_data.shape());
        let one_minus_z = ones
            .sub(&self.output_data)
            .expect("SigmoidBackward: 1 - z failed");
        let z_times_1mz = self
            .output_data
            .mul(&one_minus_z)
            .expect("SigmoidBackward: z*(1-z) failed");
        vec![grad_output
            .mul(&z_times_1mz)
            .expect("SigmoidBackward: grad * z*(1-z) failed")]
    }
}

/// `LogBackward`: saves the forward input `x`.
///
/// dL/dx = grad / x
struct LogBackward {
    input_data: Tensor,
}

impl BackwardFn for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output
            .div(&self.input_data)
            .expect("LogBackward: grad / x failed")]
    }
}

/// `ExpBackward`: saves the forward output `z = exp(x)`.
///
/// dL/dx = grad * z
struct ExpBackward {
    output_data: Tensor,
}

impl BackwardFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![grad_output
            .mul(&self.output_data)
            .expect("ExpBackward: grad * exp(x) failed")]
    }
}

/// `SumBackward`: broadcasts the scalar gradient back to the input shape.
struct SumBackward {
    input_shape: Vec<usize>,
}

impl BackwardFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad_output is shape [1]; broadcast it to input_shape.
        vec![grad_output.expand_to(&self.input_shape)]
    }
}

// ─── Variable ops ─────────────────────────────────────────────────────────────

impl Variable {
    /// Element-wise addition with broadcasting.
    ///
    /// # Panics (from underlying Tensor)
    /// Panics if shapes are not broadcast-compatible.
    pub fn var_add(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        let lhs_shape = self.data.shape().to_vec();
        let rhs_shape = other.data.shape().to_vec();
        let output = self
            .data
            .add(&other.data)
            .expect("Variable::var_add: shape mismatch");
        make_output(
            output,
            vec![Arc::clone(self), Arc::clone(other)],
            Box::new(AddBackward {
                lhs_shape,
                rhs_shape,
            }),
        )
    }

    /// Element-wise subtraction with broadcasting.
    pub fn var_sub(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        let lhs_shape = self.data.shape().to_vec();
        let rhs_shape = other.data.shape().to_vec();
        let output = self
            .data
            .sub(&other.data)
            .expect("Variable::var_sub: shape mismatch");
        make_output(
            output,
            vec![Arc::clone(self), Arc::clone(other)],
            Box::new(SubBackward {
                lhs_shape,
                rhs_shape,
            }),
        )
    }

    /// Element-wise multiplication with broadcasting.
    pub fn var_mul(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        let lhs_data = self.data.clone();
        let rhs_data = other.data.clone();
        let output = self
            .data
            .mul(&other.data)
            .expect("Variable::var_mul: shape mismatch");
        make_output(
            output,
            vec![Arc::clone(self), Arc::clone(other)],
            Box::new(MulBackward { lhs_data, rhs_data }),
        )
    }

    /// Element-wise division.  No broadcasting gradient for the divisor is
    /// implemented yet — both tensors must have the same shape.
    pub fn var_div(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        // dL/dx = grad / y,  dL/dy = -grad * x / y^2
        let lhs_data = self.data.clone();
        let rhs_data = other.data.clone();
        let output = self
            .data
            .div(&other.data)
            .expect("Variable::var_div: shape mismatch");

        struct DivBackward {
            lhs_data: Tensor,
            rhs_data: Tensor,
        }
        impl BackwardFn for DivBackward {
            fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
                // dL/dx = grad / y
                let grad_x = grad_output
                    .div(&self.rhs_data)
                    .expect("DivBackward: grad/y failed");
                // dL/dy = -grad * x / y^2
                let y_sq = self
                    .rhs_data
                    .mul(&self.rhs_data)
                    .expect("DivBackward: y*y failed");
                let grad_y = grad_output
                    .mul(&self.lhs_data)
                    .expect("DivBackward: grad*x failed")
                    .div(&y_sq)
                    .expect("DivBackward: (grad*x)/y^2 failed")
                    .neg();
                vec![grad_x, grad_y]
            }
        }

        make_output(
            output,
            vec![Arc::clone(self), Arc::clone(other)],
            Box::new(DivBackward { lhs_data, rhs_data }),
        )
    }

    /// 2-D matrix multiplication.
    pub fn var_matmul(self: &Arc<Self>, other: &Arc<Self>) -> Arc<Self> {
        let lhs_data = self.data.clone();
        let rhs_data = other.data.clone();
        let output = self
            .data
            .matmul(&other.data)
            .expect("Variable::var_matmul: dimension mismatch");
        make_output(
            output,
            vec![Arc::clone(self), Arc::clone(other)],
            Box::new(MatmulBackward { lhs_data, rhs_data }),
        )
    }

    /// Negation: `-x`.
    pub fn var_neg(self: &Arc<Self>) -> Arc<Self> {
        let output = self.data.neg();
        make_output(output, vec![Arc::clone(self)], Box::new(NegBackward))
    }

    /// ReLU activation: `max(0, x)`.
    pub fn var_relu(self: &Arc<Self>) -> Arc<Self> {
        let input_data = self.data.clone();
        let output = self.data.relu();
        make_output(
            output,
            vec![Arc::clone(self)],
            Box::new(ReluBackward { input_data }),
        )
    }

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn var_sigmoid(self: &Arc<Self>) -> Arc<Self> {
        let output = self.data.sigmoid();
        let output_data = output.clone();
        make_output(
            output,
            vec![Arc::clone(self)],
            Box::new(SigmoidBackward { output_data }),
        )
    }

    /// Natural logarithm: `ln(x)`.
    pub fn var_log(self: &Arc<Self>) -> Arc<Self> {
        let input_data = self.data.clone();
        let output = self.data.log();
        make_output(
            output,
            vec![Arc::clone(self)],
            Box::new(LogBackward { input_data }),
        )
    }

    /// Exponential: `e^x`.
    pub fn var_exp(self: &Arc<Self>) -> Arc<Self> {
        let output = self.data.exp();
        let output_data = output.clone();
        make_output(
            output,
            vec![Arc::clone(self)],
            Box::new(ExpBackward { output_data }),
        )
    }

    /// Sums all elements to a scalar tensor of shape `[1]`.
    ///
    /// The backward pass broadcasts the scalar gradient back to the original
    /// shape, so every element receives the same gradient.
    pub fn var_sum(self: &Arc<Self>) -> Arc<Self> {
        let input_shape = self.data.shape().to_vec();
        let output = self.data.sum();
        make_output(
            output,
            vec![Arc::clone(self)],
            Box::new(SumBackward { input_shape }),
        )
    }
}
