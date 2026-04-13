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
}
