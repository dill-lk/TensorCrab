use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::tensor::Tensor;

/// A mini-batch data loader that wraps paired (input, target) tensors.
///
/// The dataset is stored as two `Tensor`s where the first axis indexes
/// individual samples.  On each epoch the loader can optionally shuffle the
/// sample order before yielding batches.
///
/// # Limitations
/// - Only supports `f32` tensors.
/// - The last batch may be smaller than `batch_size` if the dataset size is
///   not evenly divisible (no `drop_last` option yet).
///
/// # Example
/// ```
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::optim::DataLoader;
///
/// // 4 samples, 2 features each, with scalar targets.
/// let x = Tensor::from_vec(
///     vec![1.0_f32, 0.0,  0.0, 1.0,  1.0, 1.0,  0.0, 0.0],
///     &[4, 2],
/// );
/// let y = Tensor::from_vec(vec![1.0_f32, 1.0, 0.0, 0.0], &[4, 1]);
///
/// let loader = DataLoader::new(x, y, 2, false);
/// let batches: Vec<_> = loader.iter_epoch(None).collect();
/// assert_eq!(batches.len(), 2);
/// for (bx, by) in &batches {
///     assert_eq!(bx.shape()[0], 2);  // batch_size = 2
///     assert_eq!(by.shape()[0], 2);
/// }
/// ```
pub struct DataLoader {
    /// Input tensor, shape `[n_samples, …]`.
    pub inputs: Tensor,
    /// Target tensor, shape `[n_samples, …]`.
    pub targets: Tensor,
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle the dataset at the start of each epoch.
    pub shuffle: bool,
}

impl DataLoader {
    /// Creates a new `DataLoader`.
    ///
    /// # Panics
    /// Panics if `inputs` and `targets` have different leading-axis sizes, or
    /// if `batch_size` is 0.
    pub fn new(inputs: Tensor, targets: Tensor, batch_size: usize, shuffle: bool) -> Self {
        assert_eq!(
            inputs.shape()[0],
            targets.shape()[0],
            "DataLoader: inputs and targets must have the same number of samples \
             (got {} vs {})",
            inputs.shape()[0],
            targets.shape()[0],
        );
        assert!(batch_size > 0, "DataLoader: batch_size must be > 0");
        Self {
            inputs,
            targets,
            batch_size,
            shuffle,
        }
    }

    /// Returns the total number of samples in the dataset.
    pub fn n_samples(&self) -> usize {
        self.inputs.shape()[0]
    }

    /// Returns the number of mini-batches per epoch (ceiling division).
    pub fn n_batches(&self) -> usize {
        self.n_samples().div_ceil(self.batch_size)
    }

    /// Iterates over mini-batches for one epoch.
    ///
    /// Each item is a `(input_batch, target_batch)` pair of new `Tensor`s with
    /// the same shape as the originals except the first axis equals the
    /// current batch size.
    ///
    /// The `seed` parameter controls the RNG when `shuffle = true`.  Pass
    /// `None` to use a time-derived seed, or `Some(seed)` for reproducible
    /// results.
    pub fn iter_epoch(&self, seed: Option<u64>) -> DataLoaderIter<'_> {
        let n = self.n_samples();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            let mut rng = match seed {
                Some(s) => SmallRng::seed_from_u64(s),
                None => SmallRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }

        DataLoaderIter {
            loader: self,
            indices,
            cursor: 0,
        }
    }
}

/// Iterator yielding `(input_batch, target_batch)` pairs.
///
/// Created by [`DataLoader::iter_epoch`].
pub struct DataLoaderIter<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    cursor: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.indices.len() {
            return None;
        }

        let end = (self.cursor + self.loader.batch_size).min(self.indices.len());
        let batch_idx = &self.indices[self.cursor..end];
        self.cursor = end;

        let (x_batch, y_batch) = gather_rows(&self.loader.inputs, &self.loader.targets, batch_idx);
        Some((x_batch, y_batch))
    }
}

// ─── Row gathering ────────────────────────────────────────────────────────────

/// Gathers the rows identified by `indices` from `inputs` and `targets`.
///
/// Handles multi-dimensional tensors: the first axis is the sample axis;
/// all remaining axes are kept intact.
fn gather_rows(inputs: &Tensor, targets: &Tensor, indices: &[usize]) -> (Tensor, Tensor) {
    let x_batch = gather_single(inputs, indices);
    let y_batch = gather_single(targets, indices);
    (x_batch, y_batch)
}

fn gather_single(t: &Tensor, indices: &[usize]) -> Tensor {
    let shape = t.shape();
    let n_rows = indices.len();
    // Number of elements per row.
    let row_size: usize = shape[1..].iter().product();
    let all_data = t.to_vec();

    let mut out_data = Vec::with_capacity(n_rows * row_size);
    for &i in indices {
        let start = i * row_size;
        out_data.extend_from_slice(&all_data[start..start + row_size]);
    }

    let mut out_shape = shape.to_vec();
    out_shape[0] = n_rows;
    Tensor::from_vec(out_data, &out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize) -> (Tensor, Tensor) {
        let x_data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        (
            Tensor::from_vec(x_data, &[n, 2]),
            Tensor::from_vec(y_data, &[n, 1]),
        )
    }

    #[test]
    fn test_n_batches() {
        let (x, y) = make_dataset(10);
        let dl = DataLoader::new(x, y, 3, false);
        // ceil(10 / 3) = 4
        assert_eq!(dl.n_batches(), 4);
    }

    #[test]
    fn test_iter_epoch_no_shuffle_all_rows() {
        let (x, y) = make_dataset(6);
        let dl = DataLoader::new(x, y, 2, false);
        let batches: Vec<_> = dl.iter_epoch(None).collect();
        assert_eq!(batches.len(), 3);
        for (bx, by) in &batches {
            assert_eq!(bx.shape()[0], 2);
            assert_eq!(by.shape()[0], 2);
        }
    }

    #[test]
    fn test_iter_epoch_last_batch_smaller() {
        let (x, y) = make_dataset(7);
        let dl = DataLoader::new(x, y, 3, false);
        let batches: Vec<_> = dl.iter_epoch(None).collect();
        assert_eq!(batches.len(), 3);
        // Last batch should have 1 sample.
        assert_eq!(batches[2].0.shape()[0], 1);
    }

    #[test]
    fn test_iter_epoch_sequential_order() {
        // Without shuffling the batches should cover all samples in order.
        let (x, y) = make_dataset(4);
        let dl = DataLoader::new(x, y, 2, false);
        let batches: Vec<_> = dl.iter_epoch(None).collect();

        let first_row_x = batches[0].0.to_vec()[0]; // first element of first batch
        let third_row_x = batches[1].0.to_vec()[0]; // first element of second batch

        // Without shuffle, sample 0 comes first, then sample 2.
        assert!(
            first_row_x < third_row_x,
            "samples should be in ascending order without shuffle"
        );
    }

    #[test]
    fn test_iter_epoch_with_shuffle_covers_all() {
        // Shuffled batches should still cover every sample exactly once.
        let (x, y) = make_dataset(8);
        let dl = DataLoader::new(x, y, 2, true);
        let batches: Vec<_> = dl.iter_epoch(Some(42)).collect();
        assert_eq!(batches.len(), 4);

        let total_samples: usize = batches.iter().map(|(bx, _)| bx.shape()[0]).sum();
        assert_eq!(total_samples, 8, "all 8 samples should appear in batches");
    }

    #[test]
    fn test_iter_epoch_1d_targets() {
        // Targets with shape [n] should still work.
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4, 1]);
        let y = Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 1.0], &[4, 1]);
        let dl = DataLoader::new(x, y, 2, false);
        let batches: Vec<_> = dl.iter_epoch(None).collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_data_loader_in_training_loop() {
        // Smoke test: use DataLoader inside a simple training loop.
        use crate::autograd::{backward, Variable};
        use crate::nn::{loss, Linear, Module, Sequential};
        use crate::optim::{Optimizer, SGD};
        let (x_tensor, y_tensor) = make_dataset(8);
        let model = Sequential::new(vec![Box::new(Linear::new(2, 1))]);
        let mut opt = SGD::new(model.parameters(), 0.01);
        let dl = DataLoader::new(x_tensor, y_tensor, 4, false);

        for (bx, by) in dl.iter_epoch(None) {
            let x_var = Variable::new(bx, false);
            let y_var = Variable::new(by, false);
            let pred = model.forward(&x_var);
            let l = loss::mse_loss(&pred, &y_var);
            backward(&l);
            opt.step();
            opt.zero_grad();
        }
        // Just verify no panic occurred.
    }
}
