/// Typed errors for TensorCrab operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Two tensors had incompatible shapes for an operation.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Broadcasting failed because shapes are not compatible.
    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError {
        /// Shape of the first tensor.
        a: Vec<usize>,
        /// Shape of the second tensor.
        b: Vec<usize>,
    },

    /// Matrix multiplication dimension mismatch.
    #[error("matmul dimension mismatch: lhs columns {lhs_cols} != rhs rows {rhs_rows}")]
    MatmulError {
        /// Number of columns in the left-hand side tensor.
        lhs_cols: usize,
        /// Number of rows in the right-hand side tensor.
        rhs_rows: usize,
    },

    /// Cannot reshape a tensor into the target shape.
    #[error("reshape: cannot reshape {from:?} into {to:?}")]
    ReshapeError {
        /// Original shape.
        from: Vec<usize>,
        /// Target shape.
        to: Vec<usize>,
    },

    /// Transpose requires a 2-D tensor.
    #[error("transpose requires a 2D tensor, got {ndim}D")]
    TransposeError {
        /// Number of dimensions of the tensor that was passed.
        ndim: usize,
    },

    /// An axis index was out of range for the tensor's number of dimensions.
    #[error("axis {axis} out of bounds for tensor with {ndim} dimensions")]
    AxisError {
        /// The axis that was requested.
        axis: usize,
        /// The actual number of dimensions.
        ndim: usize,
    },

    /// A multi-dimensional index was out of bounds.
    #[error("index {index:?} out of bounds for shape {shape:?}")]
    IndexError {
        /// The index that was provided.
        index: Vec<usize>,
        /// The shape of the tensor.
        shape: Vec<usize>,
    },

    /// Invalid padding specification.
    #[error("invalid padding: {msg}")]
    PaddingError {
        /// Description of what went wrong.
        msg: String,
    },

    /// Cannot squeeze a dimension that is not size 1.
    #[error("cannot squeeze axis {axis}: dimension has size {size}, not 1")]
    SqueezeError {
        /// The axis that was requested.
        axis: usize,
        /// The actual size of that dimension.
        size: usize,
    },

    /// Gather index out of range or shape mismatch.
    #[error("gather error: {msg}")]
    GatherError {
        /// Description of what went wrong.
        msg: String,
    },

    /// Cannot split or chunk the tensor as requested.
    #[error("chunk error: {msg}")]
    ChunkError {
        /// Description of what went wrong.
        msg: String,
    },

    /// Invalid permutation for `permute`.
    #[error("permutation error: {msg}")]
    PermutationError {
        /// Description of what went wrong.
        msg: String,
    },
}
