use std::sync::Arc;

/// A contiguous, ref-counted block of heap memory shared across tensor views.
///
/// Multiple [`crate::tensor::Tensor`] instances may point to the same `Storage`
/// via an [`Arc`], allowing zero-copy views (transpose, reshape, etc.).
#[derive(Debug, Clone)]
pub struct Storage<T> {
    data: Arc<Vec<T>>,
}

impl<T: Clone> Storage<T> {
    /// Wraps `data` in a new `Storage`.
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Returns the number of elements in the storage.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the storage contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice view of the underlying data.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}
