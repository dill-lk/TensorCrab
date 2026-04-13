use std::io::{Read, Write};
use std::sync::Arc;

use crate::autograd::Variable;
use crate::tensor::Tensor;

use super::Module;

// ─── Sequential ──────────────────────────────────────────────────────────────

/// An ordered container that chains modules together.
///
/// The output of each module is fed as input to the next.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tensor_crab::tensor::Tensor;
/// use tensor_crab::autograd::Variable;
/// use tensor_crab::nn::{Module, Sequential, Linear, ReLU};
///
/// let model = Sequential::new(vec![
///     Box::new(Linear::new(4, 8)),
///     Box::new(ReLU::new()),
///     Box::new(Linear::new(8, 2)),
/// ]);
///
/// let x = Variable::new(Tensor::randn_seeded(&[3, 4], 0), false);
/// let y = model.forward(&x);
/// assert_eq!(y.data.shape(), &[3, 2]);
/// ```
pub struct Sequential {
    /// The ordered list of modules.
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates a new `Sequential` from a list of boxed modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        self.layers
            .iter()
            .fold(Arc::clone(input), |x, layer| layer.forward(&x))
    }

    fn parameters(&self) -> Vec<Arc<Variable>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn set_training(&self, training: bool) {
        for layer in &self.layers {
            layer.set_training(training);
        }
    }
}

// ─── Binary save / load ───────────────────────────────────────────────────────
//
// Format:
//   magic:    6 bytes  b"TCRAB1"
//   n_params: u64 LE
//   For each parameter:
//     ndim:   u32 LE
//     dims:   ndim * u64 LE
//     data:   numel * f32 LE

const MAGIC: &[u8; 6] = b"TCRAB1";

impl Sequential {
    /// Saves all trainable parameters to a binary file.
    ///
    /// Use [`Sequential::load_weights`] to restore them into a model with the
    /// same architecture.
    ///
    /// # Errors
    /// Returns an `io::Error` if the file cannot be created or written.
    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;

        let params = self.parameters();
        f.write_all(&(params.len() as u64).to_le_bytes())?;

        for param in &params {
            let shape = param.data.shape();
            f.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &d in shape {
                f.write_all(&(d as u64).to_le_bytes())?;
            }
            for v in param.data.to_vec() {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Loads parameter values from a binary file saved by
    /// [`Sequential::save_weights`].
    ///
    /// The model must have the same architecture (same number of parameters
    /// with identical shapes) as when the file was saved.  The loaded data is
    /// returned as a `Vec<Tensor>` in parameter order (matching the order
    /// returned by [`Module::parameters`]).
    ///
    /// # Errors
    /// Returns an `io::Error` if the file cannot be read, the magic bytes do
    /// not match, or the parameter count / shapes differ.
    pub fn load_weights(&self, path: &str) -> std::io::Result<Vec<Tensor>> {
        let mut f = std::fs::File::open(path)?;

        let mut magic = [0u8; 6];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic bytes — not a TensorCrab weight file",
            ));
        }

        let mut n_buf = [0u8; 8];
        f.read_exact(&mut n_buf)?;
        let n_params = u64::from_le_bytes(n_buf) as usize;

        let current_params = self.parameters();
        if current_params.len() != n_params {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "parameter count mismatch: file has {n_params}, model has {}",
                    current_params.len()
                ),
            ));
        }

        let mut loaded = Vec::with_capacity(n_params);
        for (i, current) in current_params.iter().enumerate() {
            let mut ndim_buf = [0u8; 4];
            f.read_exact(&mut ndim_buf)?;
            let ndim = u32::from_le_bytes(ndim_buf) as usize;

            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                let mut d_buf = [0u8; 8];
                f.read_exact(&mut d_buf)?;
                shape.push(u64::from_le_bytes(d_buf) as usize);
            }

            if shape.as_slice() != current.data.shape() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "shape mismatch for parameter {i}: file {:?}, model {:?}",
                        shape,
                        current.data.shape()
                    ),
                ));
            }

            let numel: usize = shape.iter().product();
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                let mut v_buf = [0u8; 4];
                f.read_exact(&mut v_buf)?;
                data.push(f32::from_le_bytes(v_buf));
            }
            loaded.push(Tensor::from_vec(data, &shape));
        }
        Ok(loaded)
    }
}
