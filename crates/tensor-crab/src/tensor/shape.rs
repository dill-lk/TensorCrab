/// Shape and row-major strides for an N-dimensional tensor.
///
/// Strides encode how many elements to skip in flat storage to advance by one
/// step along each dimension, enabling zero-copy views (e.g. transpose).
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    /// Length of each dimension (e.g. `[3, 4]` for a 3 × 4 matrix).
    pub dims: Vec<usize>,
    /// Number of elements to skip per step along each dimension.
    pub strides: Vec<usize>,
}

impl Shape {
    /// Creates a row-major `Shape` from the given dimension sizes.
    ///
    /// For a shape `[d0, d1, ..., dn]` the strides are computed as:
    /// `strides[i] = strides[i+1] * dims[i+1]`, with `strides[ndim-1] = 1`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::shape::Shape;
    /// let s = Shape::row_major(&[3, 4]);
    /// assert_eq!(s.strides, vec![4, 1]);
    /// ```
    pub fn row_major(dims: &[usize]) -> Self {
        let ndim = dims.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Self {
            dims: dims.to_vec(),
            strides,
        }
    }

    /// Returns the total number of elements (product of all dimensions).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Computes the flat storage index from a multi-dimensional index array.
    ///
    /// `flat = sum(indices[i] * strides[i])` for all `i`.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(idx, stride)| idx * stride)
            .sum()
    }
}
