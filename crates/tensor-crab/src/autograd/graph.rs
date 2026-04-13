use std::sync::Arc;

use crate::tensor::Tensor;

use super::Variable;

/// Defines how to compute input gradients for one operation in the compute graph.
///
/// Every operation that supports autograd (add, mul, matmul, relu, …) has a
/// corresponding struct that implements this trait.  The struct captures any
/// tensors from the forward pass that are required by the backward pass (e.g.
/// `MulBackward` saves both inputs so it can compute `grad * other` and
/// `grad * self`).
pub trait BackwardFn: Send + Sync {
    /// Given `grad_output` — the gradient flowing *into* this node's output —
    /// returns the gradients for each input, in the same order as
    /// [`Node::inputs`].
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

/// A node in the implicit computation graph.
///
/// Each non-leaf [`Variable`] holds exactly one `Node` (via
/// `Variable::grad_fn`).  The node stores:
/// - the [`BackwardFn`] that knows how to compute input gradients, and
/// - `Arc` references to the input [`Variable`]s so the backward pass can
///   traverse the full graph.
pub struct Node {
    /// Function that computes gradients for each input.
    pub backward_fn: Box<dyn BackwardFn>,
    /// The input variables to this operation.
    pub inputs: Vec<Arc<Variable>>,
}
