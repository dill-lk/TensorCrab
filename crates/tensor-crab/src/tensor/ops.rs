use std::fmt;

use crate::error::TensorError;

use super::shape::Shape;
use super::{IndexIterator, Tensor};

// ─── Broadcasting helpers ─────────────────────────────────────────────────────

/// Computes the output shape when broadcasting `a_shape` against `b_shape`
/// following NumPy broadcasting rules (right-align, pad with 1 on the left).
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    let out_ndim = a.len().max(b.len());
    let a_pad = out_ndim - a.len();
    let b_pad = out_ndim - b.len();

    let mut out = Vec::with_capacity(out_ndim);
    for i in 0..out_ndim {
        let da = if i < a_pad { 1 } else { a[i - a_pad] };
        let db = if i < b_pad { 1 } else { b[i - b_pad] };

        if da == db || db == 1 {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else {
            return Err(TensorError::BroadcastError {
                a: a.to_vec(),
                b: b.to_vec(),
            });
        }
    }
    Ok(out)
}

/// Maps an output multi-dimensional index back to the corresponding index into
/// `tensor_shape`, applying broadcast semantics (size-1 dims map to 0).
fn broadcast_index(out_idx: &[usize], tensor_shape: &[usize], out_ndim: usize) -> Vec<usize> {
    let pad = out_ndim - tensor_shape.len();
    out_idx[pad..]
        .iter()
        .zip(tensor_shape.iter())
        .map(|(&oi, &ts)| if ts == 1 { 0 } else { oi })
        .collect()
}

// ─── Element-wise and scalar ops ──────────────────────────────────────────────

impl Tensor<f32> {
    /// Element-wise addition with broadcasting.
    ///
    /// Returns a new tensor whose shape is the broadcast of `self` and `other`.
    ///
    /// # Errors
    /// Returns [`TensorError::BroadcastError`] if the shapes are incompatible.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
    /// let b = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0], &[3]);
    /// let c = a.add(&b).unwrap();
    /// assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    /// ```
    pub fn add(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        self.elementwise(other, |a, b| a + b)
    }

    /// Element-wise subtraction with broadcasting.
    ///
    /// # Errors
    /// Returns [`TensorError::BroadcastError`] if the shapes are incompatible.
    pub fn sub(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        self.elementwise(other, |a, b| a - b)
    }

    /// Element-wise multiplication with broadcasting.
    ///
    /// # Errors
    /// Returns [`TensorError::BroadcastError`] if the shapes are incompatible.
    pub fn mul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        self.elementwise(other, |a, b| a * b)
    }

    /// Element-wise division with broadcasting.
    ///
    /// # Errors
    /// Returns [`TensorError::BroadcastError`] if the shapes are incompatible.
    pub fn div(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        self.elementwise(other, |a, b| a / b)
    }

    /// Adds a scalar value to every element.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]);
    /// assert_eq!(a.add_scalar(5.0).to_vec(), vec![6.0, 7.0]);
    /// ```
    pub fn add_scalar(&self, scalar: f32) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v + scalar).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Multiplies every element by a scalar value.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f32, 3.0], &[2]);
    /// assert_eq!(a.mul_scalar(4.0).to_vec(), vec![8.0, 12.0]);
    /// ```
    pub fn mul_scalar(&self, scalar: f32) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v * scalar).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Generic helper that performs a broadcastable element-wise operation.
    fn elementwise(
        &self,
        other: &Tensor<f32>,
        op: impl Fn(f32, f32) -> f32,
    ) -> Result<Tensor<f32>, TensorError> {
        let out_shape = broadcast_shapes(&self.shape.dims, &other.shape.dims)?;
        let out_ndim = out_shape.len();
        let numel: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(numel);

        for idx in IndexIterator::new(out_shape.clone()) {
            let a_idx = broadcast_index(&idx, &self.shape.dims, out_ndim);
            let b_idx = broadcast_index(&idx, &other.shape.dims, out_ndim);
            data.push(op(self.get_at(&a_idx), other.get_at(&b_idx)));
        }

        Ok(Tensor::from_vec(data, &out_shape))
    }
}

// ─── Matrix ops ───────────────────────────────────────────────────────────────

impl Tensor<f32> {
    /// Matrix multiplication of two 2-D tensors: `(m, k) @ (k, n) → (m, n)`.
    ///
    /// # Errors
    /// - [`TensorError::TransposeError`] if either tensor is not 2-D.
    /// - [`TensorError::MatmulError`] if the inner dimensions don't match.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    /// let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
    /// let c = a.matmul(&b).unwrap();
    /// // [[19, 22], [43, 50]]
    /// assert_eq!(c.to_vec()[0], 19.0);
    /// ```
    pub fn matmul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::TransposeError {
                ndim: self.shape.ndim(),
            });
        }
        if other.shape.ndim() != 2 {
            return Err(TensorError::TransposeError {
                ndim: other.shape.ndim(),
            });
        }

        let m = self.shape.dims[0];
        let k = self.shape.dims[1];
        let k2 = other.shape.dims[0];
        let n = other.shape.dims[1];

        if k != k2 {
            return Err(TensorError::MatmulError {
                lhs_cols: k,
                rhs_rows: k2,
            });
        }

        let mut data = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += self.get_at(&[i, p]) * other.get_at(&[p, j]);
                }
                data[i * n + j] = acc;
            }
        }

        Ok(Tensor::from_vec(data, &[m, n]))
    }

    /// Returns a zero-copy transpose of a 2-D tensor.
    ///
    /// The underlying storage is shared (via `Arc`); only the shape and
    /// strides are swapped.  The returned tensor is non-contiguous.
    ///
    /// # Errors
    /// Returns [`TensorError::TransposeError`] if the tensor is not 2-D.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    /// let t = a.transpose().unwrap();
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.get_at(&[0, 1]), 3.0); // was a[1][0]
    /// ```
    pub fn transpose(&self) -> Result<Tensor<f32>, TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::TransposeError {
                ndim: self.shape.ndim(),
            });
        }

        let new_dims = vec![self.shape.dims[1], self.shape.dims[0]];
        let new_strides = vec![self.shape.strides[1], self.shape.strides[0]];

        Ok(Tensor {
            storage: self.storage.clone(), // zero-copy: Arc clone
            shape: Shape {
                dims: new_dims,
                strides: new_strides,
            },
            offset: self.offset,
        })
    }

    /// Returns a view (or copy, if non-contiguous) with a new shape.
    ///
    /// The total number of elements must be preserved.
    ///
    /// # Errors
    /// Returns [`TensorError::ReshapeError`] if the element counts differ.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = a.reshape(&[3, 2]).unwrap();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor<f32>, TensorError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.shape.numel() {
            return Err(TensorError::ReshapeError {
                from: self.shape.dims.clone(),
                to: new_shape.to_vec(),
            });
        }

        // Make the tensor contiguous first (a no-op if already contiguous).
        let contiguous = self.contiguous();
        Ok(Tensor {
            storage: contiguous.storage,
            shape: Shape::row_major(new_shape),
            offset: 0,
        })
    }

    /// Reshapes the tensor to a 1-D vector of all its elements.
    ///
    /// Equivalent to `reshape(&[numel])`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    /// assert_eq!(a.flatten().shape(), &[4]);
    /// ```
    pub fn flatten(&self) -> Tensor<f32> {
        let n = self.shape.numel();
        // SAFETY: numel can never mismatch when we construct from to_vec().
        self.reshape(&[n])
            .expect("flatten: reshape to [numel] must always succeed")
    }
}

// ─── Reduction ops ────────────────────────────────────────────────────────────

impl Tensor<f32> {
    /// Reduces all elements to their sum, returning a scalar tensor of shape `[1]`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
    /// assert_eq!(a.sum().to_vec()[0], 6.0);
    /// ```
    pub fn sum(&self) -> Tensor<f32> {
        let s: f32 = self.to_vec().iter().sum();
        Tensor::from_vec(vec![s], &[1])
    }

    /// Reduces all elements to their arithmetic mean, returning a scalar tensor
    /// of shape `[1]`.
    ///
    /// Returns `0.0` for an empty tensor.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]);
    /// assert_eq!(a.mean().to_vec()[0], 2.5);
    /// ```
    pub fn mean(&self) -> Tensor<f32> {
        let n = self.shape.numel();
        if n == 0 {
            return Tensor::from_vec(vec![0.0_f32], &[1]);
        }
        let s: f32 = self.to_vec().iter().sum();
        #[allow(clippy::cast_precision_loss)]
        // Precision loss is acceptable — we are working in f32 throughout.
        let mean = s / n as f32;
        Tensor::from_vec(vec![mean], &[1])
    }

    /// Reduces all elements to the maximum, returning a scalar tensor of shape `[1]`.
    ///
    /// Returns `f32::NEG_INFINITY` for an empty tensor.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![3.0_f32, 1.0, 4.0], &[3]);
    /// assert_eq!(a.max().to_vec()[0], 4.0);
    /// ```
    pub fn max(&self) -> Tensor<f32> {
        let m = self.to_vec().into_iter().fold(f32::NEG_INFINITY, f32::max);
        Tensor::from_vec(vec![m], &[1])
    }

    /// Reduces all elements to the minimum, returning a scalar tensor of shape `[1]`.
    ///
    /// Returns `f32::INFINITY` for an empty tensor.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![3.0_f32, 1.0, 4.0], &[3]);
    /// assert_eq!(a.min().to_vec()[0], 1.0);
    /// ```
    pub fn min(&self) -> Tensor<f32> {
        let m = self.to_vec().into_iter().fold(f32::INFINITY, f32::min);
        Tensor::from_vec(vec![m], &[1])
    }
}

// ─── Unary math ops (needed by autograd) ─────────────────────────────────────

impl Tensor<f32> {
    /// Applies the ReLU activation element-wise: `max(0, x)`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![-1.0_f32, 0.0, 2.0], &[3]);
    /// assert_eq!(a.relu().to_vec(), vec![0.0, 0.0, 2.0]);
    /// ```
    pub fn relu(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v.max(0.0)).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Applies the sigmoid activation element-wise: `1 / (1 + exp(-x))`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![0.0_f32], &[1]);
    /// assert!((a.sigmoid().to_vec()[0] - 0.5).abs() < 1e-6);
    /// ```
    pub fn sigmoid(&self) -> Tensor<f32> {
        let data: Vec<f32> = self
            .to_vec()
            .into_iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Applies the natural logarithm element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, std::f32::consts::E], &[2]);
    /// let b = a.log();
    /// assert!((b.to_vec()[0]).abs() < 1e-6);
    /// assert!((b.to_vec()[1] - 1.0).abs() < 1e-6);
    /// ```
    pub fn log(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v.ln()).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Applies `e^x` element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![0.0_f32, 1.0], &[2]);
    /// let b = a.exp();
    /// assert!((b.to_vec()[0] - 1.0).abs() < 1e-6);
    /// assert!((b.to_vec()[1] - std::f32::consts::E).abs() < 1e-5);
    /// ```
    pub fn exp(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v.exp()).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Negates every element: `-x`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, -2.0], &[2]);
    /// assert_eq!(a.neg().to_vec(), vec![-1.0, 2.0]);
    /// ```
    pub fn neg(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| -v).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Expands (broadcasts) a scalar tensor of shape `[1]` to the given shape
    /// by repeating the single value.
    ///
    /// Used internally by the autograd backward pass to propagate a scalar
    /// gradient back to a multi-element input.
    ///
    /// # Panics
    /// Panics if `self` is not a scalar (shape `[1]`).
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let scalar = Tensor::from_vec(vec![3.0_f32], &[1]);
    /// let expanded = scalar.expand_to(&[2, 3]);
    /// assert_eq!(expanded.shape(), &[2, 3]);
    /// assert!(expanded.to_vec().iter().all(|&v| v == 3.0));
    /// ```
    pub fn expand_to(&self, shape: &[usize]) -> Tensor<f32> {
        assert_eq!(
            self.shape.numel(),
            1,
            "expand_to: only scalar tensors (numel == 1) can be expanded"
        );
        let val = self.to_vec()[0];
        let numel: usize = shape.iter().product();
        Tensor::from_vec(vec![val; numel], shape)
    }

    /// Reduces the tensor along every axis that differs from `target_shape`,
    /// handling the gradient un-broadcasting needed during backprop.
    ///
    /// Rules (NumPy broadcasting in reverse):
    /// - Leading axes that don't exist in `target_shape` are summed away.
    /// - Axes where `target_shape` has size 1 (but `self` has size > 1) are
    ///   summed, then the axis is kept with size 1.
    ///
    /// If `self.shape() == target_shape` the tensor is returned as-is.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// // Gradient for a [1, 4] input that was broadcast to [3, 4].
    /// let grad = Tensor::from_vec(vec![1.0_f32; 12], &[3, 4]);
    /// let reduced = grad.sum_to(&[1, 4]);
    /// assert_eq!(reduced.shape(), &[1, 4]);
    /// assert_eq!(reduced.to_vec(), vec![3.0; 4]);
    /// ```
    pub fn sum_to(&self, target_shape: &[usize]) -> Tensor<f32> {
        if self.shape() == target_shape {
            return self.clone();
        }

        let self_shape = self.shape.dims.clone();
        let out_ndim = self_shape.len();
        let target_ndim = target_shape.len();
        // Pad target_shape with leading 1s so it matches self's ndim.
        let pad = out_ndim.saturating_sub(target_ndim);
        let padded_target: Vec<usize> = std::iter::repeat_n(1, pad)
            .chain(target_shape.iter().copied())
            .collect();

        // Collect axes where we need to reduce.
        let reduce_axes: Vec<usize> = (0..out_ndim)
            .filter(|&i| padded_target[i] == 1 && self_shape[i] != 1)
            .collect();

        // Sum over those axes one at a time (always axis 0 after collapsing).
        let mut result = self.clone();
        for &axis in &reduce_axes {
            result = result.sum_axis_keepdim(axis);
        }

        // Drop leading size-1 axes that correspond to the padding we added.
        if pad > 0 {
            // After the reductions, the leading `pad` dims are all 1.
            // Reshape to remove them.
            let final_shape: Vec<usize> = result.shape()[pad..].to_vec();
            result = result
                .reshape(&final_shape)
                .expect("sum_to: reshape to target_shape failed");
        }
        result
    }

    /// Sums along `axis`, keeping that axis with size 1 in the output.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let s = a.sum_axis_keepdim(0);
    /// assert_eq!(s.shape(), &[1, 3]);
    /// assert_eq!(s.to_vec(), vec![5.0, 7.0, 9.0]);
    /// ```
    pub fn sum_axis_keepdim(&self, axis: usize) -> Tensor<f32> {
        let ndim = self.shape.ndim();
        assert!(
            axis < ndim,
            "sum_axis_keepdim: axis {axis} out of bounds for {ndim}D tensor"
        );

        // Output shape: same as self but axis dim = 1.
        let mut out_shape = self.shape.dims.clone();
        out_shape[axis] = 1;
        let out_numel: usize = out_shape.iter().product();
        let mut out_data = vec![0.0_f32; out_numel];
        let out_shape_obj = crate::tensor::shape::Shape::row_major(&out_shape);

        for idx in super::IndexIterator::new(self.shape.dims.clone()) {
            let val = self.get_at(&idx);
            // Map index to output index: zero out the reduced axis.
            let mut out_idx = idx.clone();
            out_idx[axis] = 0;
            let flat = out_shape_obj.flat_index(&out_idx);
            out_data[flat] += val;
        }

        Tensor::from_vec(out_data, &out_shape)
    }

    /// Applies the hyperbolic tangent element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![0.0_f32], &[1]);
    /// assert!((a.tanh().to_vec()[0]).abs() < 1e-6);
    /// ```
    pub fn tanh(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v.tanh()).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Applies the square root element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![4.0_f32, 9.0], &[2]);
    /// assert_eq!(a.sqrt().to_vec(), vec![2.0, 3.0]);
    /// ```
    pub fn sqrt(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v.sqrt()).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Clamps every element to the range `[min_val, max_val]`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![-1.0_f32, 0.5, 2.0], &[3]);
    /// assert_eq!(a.clamp(0.0, 1.0).to_vec(), vec![0.0, 0.5, 1.0]);
    /// ```
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Tensor<f32> {
        let data: Vec<f32> = self
            .to_vec()
            .into_iter()
            .map(|v| v.clamp(min_val, max_val))
            .collect();
        Tensor::from_vec(data, &self.shape.dims)
    }

    /// Broadcasts this tensor to `target_shape` by repeating elements along
    /// any axis where `self.shape[axis] == 1`.
    ///
    /// The number of dimensions of `self` must be ≤ `target_shape.len()`.  Any
    /// missing leading dimensions are treated as size 1 (NumPy-style padding).
    ///
    /// # Panics
    /// Panics if a dimension of `self` is neither 1 nor equal to the
    /// corresponding dimension of `target_shape`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
    /// let b = a.broadcast_to(&[4, 3]);
    /// assert_eq!(b.shape(), &[4, 3]);
    /// assert_eq!(b.to_vec()[3], 1.0); // second row same as first
    /// ```
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Tensor<f32> {
        let out_ndim = target_shape.len();
        let in_ndim = self.shape.ndim();
        assert!(
            out_ndim >= in_ndim,
            "broadcast_to: target ndim {out_ndim} < input ndim {in_ndim}"
        );
        let pad = out_ndim - in_ndim;
        // Validate broadcast compatibility.
        for (i, (&pi, &ti)) in self
            .shape
            .dims
            .iter()
            .zip(target_shape[pad..].iter())
            .enumerate()
        {
            assert!(
                pi == 1 || pi == ti,
                "broadcast_to: cannot broadcast dim {i} from size {pi} to {ti}"
            );
        }
        let numel: usize = target_shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        for idx in super::IndexIterator::new(target_shape.to_vec()) {
            // Map target index to source index, treating padded/size-1 dims as 0.
            let src_idx: Vec<usize> = idx[pad..]
                .iter()
                .zip(self.shape.dims.iter())
                .map(|(&oi, &ts)| if ts == 1 { 0 } else { oi })
                .collect();
            data.push(self.get_at(&src_idx));
        }
        Tensor::from_vec(data, target_shape)
    }

    /// Applies an element-wise power of 2 (`x²`).
    ///
    /// Equivalent to `self.mul(self)` but avoids creating a temporary Variable.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::from_vec(vec![2.0_f32, 3.0], &[2]);
    /// assert_eq!(a.square().to_vec(), vec![4.0, 9.0]);
    /// ```
    pub fn square(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.to_vec().into_iter().map(|v| v * v).collect();
        Tensor::from_vec(data, &self.shape.dims)
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for Tensor<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut indices = vec![0usize; self.shape.ndim()];
        fmt_recursive(f, self, &mut indices, 0)
    }
}

/// Recursively formats a tensor dimension by dimension.
fn fmt_recursive(
    f: &mut fmt::Formatter<'_>,
    tensor: &Tensor<f32>,
    indices: &mut Vec<usize>,
    dim: usize,
) -> fmt::Result {
    if dim == tensor.shape.ndim() {
        // Base case: all dimensions have been fixed — print the scalar.
        write!(f, "{}", tensor.get_at(indices))
    } else {
        write!(f, "[")?;
        for i in 0..tensor.shape.dims[dim] {
            if i > 0 {
                write!(f, ", ")?;
            }
            indices[dim] = i;
            fmt_recursive(f, tensor, indices, dim + 1)?;
        }
        write!(f, "]")
    }
}
