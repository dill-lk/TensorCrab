use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

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
    ///
    /// Wrapped in an `RwLock` so that optimizer implementations can update
    /// parameters in-place via [`Variable::set_data`] while allowing many
    /// concurrent readers during the forward/backward pass.
    data: RwLock<Tensor>,

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
            data: RwLock::new(data),
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
            data: RwLock::new(data),
            requires_grad,
            grad: Mutex::new(None),
            grad_fn: Some(node),
        })
    }

    /// Returns a read guard giving shared access to the forward-pass tensor.
    ///
    /// The guard implements [`std::ops::Deref`]`<Target = `[`Tensor`]`>`, so all
    /// `Tensor` methods are available directly on it.  Multiple threads can hold
    /// read guards concurrently; a write guard (from [`Variable::set_data`])
    /// blocks until all readers have released their guards.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::autograd::Variable;
    /// use tensor_crab::tensor::Tensor;
    ///
    /// let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), false);
    /// assert_eq!(x.data().shape(), &[2]);
    /// ```
    pub fn data(&self) -> RwLockReadGuard<'_, Tensor> {
        self.data.read().expect("Variable::data: rwlock poisoned")
    }

    /// Replaces the tensor data of this variable in-place.
    ///
    /// This is the mechanism through which optimizers update trainable
    /// parameters after computing gradients.  It acquires an exclusive write
    /// lock, so it blocks until all concurrent readers (e.g. forward-pass
    /// threads) have released their read guards.
    ///
    /// In the typical sequential training loop
    /// (forward → backward → `optimizer.step()`) there are no live read guards
    /// when this is called.
    ///
    /// # Panics
    /// Panics if the internal `RwLock` is poisoned.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::autograd::Variable;
    /// use tensor_crab::tensor::Tensor;
    ///
    /// let x = Variable::new(Tensor::zeros(&[2]), true);
    /// x.set_data(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]));
    /// assert_eq!(x.data().to_vec(), vec![1.0, 2.0]);
    /// ```
    pub fn set_data(&self, new_data: Tensor) {
        *self
            .data
            .write()
            .expect("Variable::set_data: rwlock poisoned") = new_data;
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
