pub mod data;
pub mod ops;
pub mod shape;

use std::fmt;

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::device::Device;
use data::Storage;
use shape::Shape;

/// An N-dimensional array with copy-on-write shared storage.
///
/// `Tensor<T>` is the core type in TensorCrab. It holds a ref-counted
/// [`Storage<T>`] alongside a [`Shape`] (dims + strides) and an `offset`,
/// so that operations such as [`Tensor::transpose`] produce a zero-copy view
/// of the same underlying memory.
///
/// For the current Stage 1 implementation all mathematical operations are
/// defined only for `T = f32`.  The generic parameter is kept so that `f64`
/// (and integer dtypes) can be added in later stages without breaking the API.
///
/// # Example
/// ```
/// use tensor_crab::tensor::Tensor;
/// let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
/// assert_eq!(a.shape(), &[2, 2]);
/// ```
#[derive(Clone)]
pub struct Tensor<T = f32> {
    pub(crate) storage: Storage<T>,
    pub(crate) shape: Shape,
    /// Offset into the storage where this view starts.
    pub(crate) offset: usize,
    /// The device where this tensor's data resides.
    device: Device,
}

// ─── Debug impl ──────────────────────────────────────────────────────────────

impl<T: fmt::Debug + Clone> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?})", self.to_vec_debug())
    }
}

impl<T: fmt::Debug + Clone> Tensor<T> {
    fn to_vec_debug(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.shape.numel());
        for idx in IndexIterator::new(self.shape.dims.clone()) {
            out.push(self.storage.as_slice()[self.shape.flat_index(&idx) + self.offset].clone());
        }
        out
    }
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl<T: Clone + Default> Tensor<T> {
    /// Creates a tensor filled with the type's default value (usually zero).
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let t: Tensor<f32> = Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.numel(), 6);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![T::default(); numel], shape)
    }
}

impl<T: Clone> Tensor<T> {
    /// Creates a tensor from a flat `Vec<T>` and a shape.
    ///
    /// # Panics
    /// Panics if `data.len()` does not equal the product of `shape`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
    /// assert_eq!(t.shape(), &[3]);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "from_vec: data length {} does not match shape numel {} for shape {:?}",
            data.len(),
            numel,
            shape
        );
        Self {
            storage: Storage::new(data),
            shape: Shape::row_major(shape),
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Creates a tensor filled with `value`.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let t = Tensor::full(1.0_f32, &[2, 2]);
    /// assert_eq!(t.numel(), 4);
    /// ```
    pub fn full(value: T, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self::from_vec(vec![value; numel], shape)
    }
}

impl Tensor<f32> {
    /// Creates a tensor filled with ones (`1.0_f32`).
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let t = Tensor::ones(&[3]);
    /// assert_eq!(t.to_vec()[0], 1.0_f32);
    /// ```
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(1.0_f32, shape)
    }

    /// Creates a tensor whose values are drawn from a standard normal
    /// distribution N(0, 1) using a **deterministic** seed of `0`.
    ///
    /// Equivalent to `randn_seeded(shape, 0)`.  Prefer
    /// [`Tensor::randn_seeded`] in tests so that seeds are explicit.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let t = Tensor::randn(&[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn randn(shape: &[usize]) -> Self {
        Self::randn_seeded(shape, 0)
    }

    /// Creates a tensor whose values are drawn from N(0, 1) using a fixed
    /// `seed`, guaranteeing reproducible output across runs.
    ///
    /// Uses a Box-Muller transform on uniform samples from [`SmallRng`].
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// let a = Tensor::randn_seeded(&[4], 42);
    /// let b = Tensor::randn_seeded(&[4], 42);
    /// assert_eq!(a.to_vec(), b.to_vec());
    /// ```
    pub fn randn_seeded(shape: &[usize], seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);

        // Box-Muller transform: pairs of uniform samples → normal samples.
        let mut i = 0;
        while i < numel {
            // Clamp u1 away from 0 to avoid log(0).
            let u1: f32 = (rng.gen::<f32>()).max(f32::EPSILON);
            let u2: f32 = rng.gen::<f32>();
            let mag = (-2.0_f32 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(z0);
            i += 1;
            if i < numel {
                let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();
                data.push(z1);
                i += 1;
            }
        }

        Self::from_vec(data, shape)
    }
}

// ─── Core accessors ──────────────────────────────────────────────────────────

impl<T: Copy> Tensor<T> {
    /// Returns the dimension sizes of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape.dims
    }

    /// Returns the per-dimension strides.
    pub fn strides(&self) -> &[usize] {
        &self.shape.strides
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns `true` if the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.shape.numel() == 0
    }

    /// Returns the device where this tensor's data resides.
    ///
    /// CPU tensors always return [`Device::Cpu`].
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the element at a multi-dimensional index.
    ///
    /// The index length must equal `self.ndim()`.
    pub fn get_at(&self, indices: &[usize]) -> T {
        let flat = self.shape.flat_index(indices) + self.offset;
        self.storage.as_slice()[flat]
    }

    /// Returns `true` when the tensor's memory is laid out in row-major order
    /// with no offset (i.e. no transpose or slice has been applied).
    pub fn is_contiguous(&self) -> bool {
        let expected = Shape::row_major(&self.shape.dims);
        self.offset == 0 && self.shape.strides == expected.strides
    }

    /// Collects all elements in logical (row-major) order into a `Vec`.
    ///
    /// For a contiguous tensor this is a plain slice copy; for non-contiguous
    /// tensors (e.g. after transpose) each element is looked up via its
    /// multi-dimensional index.
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            self.storage.as_slice()[..self.shape.numel()].to_vec()
        } else {
            IndexIterator::new(self.shape.dims.clone())
                .map(|idx| self.get_at(&idx))
                .collect()
        }
    }

    /// Returns a new contiguous tensor with the same values in row-major order.
    ///
    /// If the tensor is already contiguous this is a cheap `Arc` clone of the
    /// storage; otherwise elements are copied into fresh storage.
    pub fn contiguous(&self) -> Tensor<T> {
        if self.is_contiguous() {
            self.clone()
        } else {
            Tensor::from_vec(self.to_vec(), &self.shape.dims)
        }
    }

    /// Returns a tensor on the requested `device`.
    ///
    /// For [`Device::Cpu`] this is a clone.  For GPU devices, use
    /// `CudaTensor::from_cpu()` instead — this method returns
    /// [`crate::error::TensorError::UnsupportedOperation`] for GPU targets to guide
    /// callers toward the correct API.
    ///
    /// # Errors
    /// Returns [`crate::error::TensorError::UnsupportedOperation`] when `device` is a
    /// GPU device.
    ///
    /// # Example
    /// ```
    /// use tensor_crab::tensor::Tensor;
    /// use tensor_crab::device::Device;
    ///
    /// let t = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]);
    /// let same = t.to_device(&Device::Cpu).unwrap();
    /// assert_eq!(same.to_vec(), t.to_vec());
    /// ```
    pub fn to_device(&self, device: &Device) -> Result<Tensor<T>, crate::error::TensorError> {
        match device {
            Device::Cpu => Ok(self.clone()),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => Err(crate::error::TensorError::UnsupportedOperation(
                "use CudaTensor::from_cpu() for GPU transfer".to_string(),
            )),
        }
    }
}

// ─── Index iterator (crate-private) ─────────────────────────────────────────

/// Lazy iterator over all multi-dimensional indices in row-major order.
///
/// `IndexIterator::new(&[2, 3])` yields:
/// `[0,0]`, `[0,1]`, `[0,2]`, `[1,0]`, `[1,1]`, `[1,2]`.
pub(crate) struct IndexIterator {
    dims: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    /// Creates a new iterator over all indices for the given dimensions.
    pub(crate) fn new(dims: Vec<usize>) -> Self {
        let ndim = dims.len();
        let done = dims.contains(&0);
        Self {
            current: vec![0; ndim],
            dims,
            done,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment the multi-dimensional counter (last dimension ticks fastest).
        let ndim = self.dims.len();
        let mut carry = true;
        for i in (0..ndim).rev() {
            if carry {
                self.current[i] += 1;
                if self.current[i] >= self.dims[i] {
                    self.current[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            // We overflowed all dimensions — exhausted.
            self.done = true;
        }

        Some(result)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── constructors ──────────────────────────────────────────────────────

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zeros() {
        let t: Tensor<f32> = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[4]);
        assert_eq!(t.to_vec(), vec![1.0_f32; 4]);
    }

    #[test]
    fn test_randn_seeded_deterministic() {
        let a = Tensor::randn_seeded(&[8], 42);
        let b = Tensor::randn_seeded(&[8], 42);
        assert_eq!(a.to_vec(), b.to_vec());
    }

    #[test]
    fn test_randn_different_seeds() {
        let a = Tensor::randn_seeded(&[8], 1);
        let b = Tensor::randn_seeded(&[8], 2);
        assert_ne!(a.to_vec(), b.to_vec());
    }

    // ── element-wise ops ──────────────────────────────────────────────────

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[4]);
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.to_vec(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_vec(vec![2.0_f32, 3.0], &[2]);
        let b = Tensor::from_vec(vec![4.0_f32, 5.0], &[2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::from_vec(vec![6.0_f32, 8.0], &[2]);
        let b = Tensor::from_vec(vec![2.0_f32, 4.0], &[2]);
        let c = a.div(&b).unwrap();
        assert_eq!(c.to_vec(), vec![3.0, 2.0]);
    }

    // ── broadcasting ─────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_add_3x1_plus_1x4() {
        // a: [[1], [2], [3]]  shape [3, 1]
        // b: [[10, 20, 30, 40]]  shape [1, 4]
        // result shape: [3, 4]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3, 1]);
        let b = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0, 40.0], &[1, 4]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
        #[rustfmt::skip]
        let expected = vec![
            11.0, 21.0, 31.0, 41.0,
            12.0, 22.0, 32.0, 42.0,
            13.0, 23.0, 33.0, 43.0,
        ];
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_broadcast_scalar_shape_1() {
        // [1] broadcasts to [4]
        let a = Tensor::from_vec(vec![10.0_f32], &[1]);
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_broadcast_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        assert!(a.add(&b).is_err());
    }

    // ── scalar ops ────────────────────────────────────────────────────────

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        let b = a.add_scalar(10.0);
        assert_eq!(b.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::from_vec(vec![2.0_f32, 3.0, 4.0], &[3]);
        let b = a.mul_scalar(2.0);
        assert_eq!(b.to_vec(), vec![4.0, 6.0, 8.0]);
    }

    // ── matmul ────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_2x2() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(
            c.to_vec().as_slice(),
            [19.0_f32, 22.0, 43.0, 50.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
        // = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]]
        // = [[22, 28], [49, 64]]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(
            c.to_vec().as_slice(),
            [22.0_f32, 28.0, 49.0, 64.0].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3, 1]);
        assert!(a.matmul(&b).is_err());
    }

    // ── transpose ────────────────────────────────────────────────────────

    #[test]
    fn test_transpose_zero_copy() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = a.transpose().unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        // Element at [r, c] of transposed = element at [c, r] of original.
        assert_abs_diff_eq!(t.get_at(&[0, 0]), 1.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(t.get_at(&[1, 0]), 2.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(t.get_at(&[2, 0]), 3.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(t.get_at(&[0, 1]), 4.0_f32, epsilon = 1e-6);
        // Verify the Arc is shared (zero-copy) by checking storage pointer equality.
        assert!(std::ptr::eq(
            a.storage.as_slice().as_ptr(),
            t.storage.as_slice().as_ptr()
        ));
    }

    #[test]
    fn test_transpose_to_vec() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let t = a.transpose().unwrap();
        // [[1,2],[3,4]]^T = [[1,3],[2,4]]
        assert_eq!(t.to_vec(), vec![1.0_f32, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_non_square() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = a.transpose().unwrap();
        // [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
        assert_eq!(t.to_vec(), vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ── reshape / flatten ─────────────────────────────────────────────────

    #[test]
    fn test_reshape() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = a.reshape(&[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        assert!(a.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_flatten() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let f = a.flatten();
        assert_eq!(f.shape(), &[4]);
        assert_eq!(f.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_flatten_after_transpose() {
        // Flatten a transposed (non-contiguous) tensor → should respect logical order.
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let t = a.transpose().unwrap(); // [[1,3],[2,4]]
        let f = t.flatten();
        assert_eq!(f.to_vec(), vec![1.0_f32, 3.0, 2.0, 4.0]);
    }

    // ── reductions ────────────────────────────────────────────────────────

    #[test]
    fn test_sum() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let s = a.sum();
        assert_abs_diff_eq!(s.to_vec()[0], 10.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let m = a.mean();
        assert_abs_diff_eq!(m.to_vec()[0], 2.5_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_max() {
        let a = Tensor::from_vec(vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
        let m = a.max();
        assert_abs_diff_eq!(m.to_vec()[0], 9.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_min() {
        let a = Tensor::from_vec(vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
        let m = a.min();
        assert_abs_diff_eq!(m.to_vec()[0], 1.0_f32, epsilon = 1e-6);
    }

    // ── display ───────────────────────────────────────────────────────────

    #[test]
    fn test_display_1d() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
        let s = format!("{a}");
        assert!(s.contains('1') && s.contains('2') && s.contains('3'));
    }

    #[test]
    fn test_display_2d() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let s = format!("{a}");
        // Verify the output is non-empty and doesn't panic.
        assert!(!s.is_empty());
    }

    #[test]
    fn test_display_matmul_milestone() {
        // Milestone from roadmap: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.matmul(&b).unwrap();
        let s = format!("{c}");
        assert!(s.contains("19") && s.contains("22") && s.contains("43") && s.contains("50"));
    }

    // ── index iterator ────────────────────────────────────────────────────

    #[test]
    fn test_index_iterator_2d() {
        let indices: Vec<Vec<usize>> = IndexIterator::new(vec![2, 3]).collect();
        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[5], vec![1, 2]);
    }

    #[test]
    fn test_index_iterator_empty() {
        let indices: Vec<Vec<usize>> = IndexIterator::new(vec![0, 3]).collect();
        assert_eq!(indices.len(), 0);
    }
}
