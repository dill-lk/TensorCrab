use std::sync::{Arc, Mutex};

use crate::tensor::Tensor;

use super::graph::Node;

/// A tensor that participates in automatic differentiation.
///
/// A `Variable` wraps a [`Tensor`] and adds:
/// - `requires_grad`: whether gradients should be tracked through this variable.
/// - `grad`: the accumulated gradient computed by [`crate::autograd::engine::backward`].
/// - `grad_fn`: the [`Node`] that created this variable (absent for leaf nodes).
///
/// # Constructing leaf variables
///
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
///
/// // A trainable parameter — leaf node with requires_grad = true.
/// let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
/// assert!(x.requires_grad);
/// assert!(x.grad().is_none());
/// ```
pub struct Variable {
    /// The forward-pass data of this variable.
    pub data: Tensor,

    /// Whether gradients should be tracked for this variable.
    pub requires_grad: bool,

    /// Accumulated gradient, filled in by [`crate::autograd::backward`].
    ///
    /// `None` until `backward()` has been called and this variable has a
    /// non-zero contribution to the scalar loss.
    pub(crate) grad: Mutex<Option<Tensor>>,

    /// The graph node that produced this variable.
    ///
    /// `None` for leaf variables (constants and user-created parameters).
    pub grad_fn: Option<Arc<Node>>,
}

impl Variable {
    /// Creates a leaf [`Variable`] with the given tensor data and gradient
    /// tracking flag.
    ///
    /// Leaf variables have no `grad_fn`; their gradients are filled in
    /// directly by the backward pass.
    pub fn new(data: Tensor, requires_grad: bool) -> Arc<Self> {
        Arc::new(Self {
            data,
            requires_grad,
            grad: Mutex::new(None),
            grad_fn: None,
        })
    }

    /// Creates a non-leaf [`Variable`] — the result of an operation.
    ///
    /// This is called internally by operation implementations (see
    /// [`crate::autograd::ops`]).
    pub(crate) fn with_grad_fn(data: Tensor, requires_grad: bool, node: Arc<Node>) -> Arc<Self> {
        Arc::new(Self {
            data,
            requires_grad,
            grad: Mutex::new(None),
            grad_fn: Some(node),
        })
    }

    /// Returns a clone of the accumulated gradient tensor, if one exists.
    ///
    /// Returns `None` for variables where `requires_grad = false`, or before
    /// `backward()` has been called.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad
            .lock()
            .expect("Variable::grad: mutex poisoned")
            .clone()
    }

    /// Resets the accumulated gradient to `None`.
    ///
    /// Call this before each training step so that gradients from the
    /// previous step do not contaminate the current one.
    ///
    /// # Example
    /// ```
    /// use std::sync::Arc;
    /// use tensor_crab::tensor::Tensor;
    /// use tensor_crab::autograd::{Variable, backward};
    ///
    /// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
    /// let y = x.var_sum();
    /// backward(&y);
    /// assert!(x.grad().is_some());
    /// x.zero_grad();
    /// assert!(x.grad().is_none());
    /// ```
    pub fn zero_grad(&self) {
        let mut g = self
            .grad
            .lock()
            .expect("Variable::zero_grad: mutex poisoned");
        *g = None;
    }

    /// Accumulates `grad_to_add` into this variable's gradient.
    ///
    /// If no gradient has been accumulated yet, this sets `grad` to
    /// `grad_to_add`.  Otherwise it adds element-wise.
    pub(crate) fn accumulate_grad(&self, grad_to_add: &Tensor) {
        let mut g = self
            .grad
            .lock()
            .expect("Variable::accumulate_grad: mutex poisoned");
        *g = Some(match g.take() {
            None => grad_to_add.clone(),
            Some(existing) => existing
                .add(grad_to_add)
                .expect("accumulate_grad: shape mismatch between existing grad and new grad"),
        });
    }
}
