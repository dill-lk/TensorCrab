# Contributing to TensorCrab 🦀

First off — thank you for wanting to contribute. TensorCrab is an ambitious project and every contribution matters.

## Getting Started

### Prerequisites
- Rust stable (latest) — install via [rustup.rs](https://rustup.rs)
- `cargo` (comes with Rust)
- Optional: CUDA toolkit if working on GPU backend

### Setup
```bash
git clone https://github.com/yourusername/tensor-crab
cd tensor-crab
cargo build
cargo test
```

All tests should pass on a clean clone.

## How to Contribute

### Reporting Bugs
Open a GitHub Issue with:
- What you expected to happen
- What actually happened
- Minimal code to reproduce it
- Your OS and Rust version (`rustc --version`)

### Suggesting Features
Open a GitHub Discussion (not an Issue) to talk through the idea first. Check the roadmap — it might already be planned.

### Submitting Code

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Write your code
3. Write tests for it
4. Run `cargo test` — all tests must pass
5. Run `cargo clippy` — no warnings allowed
6. Run `cargo fmt` — code must be formatted
7. Open a Pull Request with a clear description

## Code Standards

### Tests are mandatory
Every public function needs a test. No exceptions.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add() {
        let a = Tensor::from([1.0, 2.0, 3.0]);
        let b = Tensor::from([4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }
}
```

### Doc comments are mandatory on public API
```rust
/// Performs matrix multiplication of two tensors.
///
/// # Arguments
/// * `other` - The tensor to multiply with. Must have compatible shape.
///
/// # Example
/// ```
/// let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
/// let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);
/// let c = a.matmul(&b);
/// ```
pub fn matmul(&self, other: &Tensor) -> Tensor { ... }
```

### Unsafe code
Only use `unsafe` when absolutely necessary. Always add a comment explaining why it's safe:

```rust
// SAFETY: We know the pointer is valid because we just allocated it
// and have exclusive ownership through the borrow checker.
unsafe { ptr.write(value); }
```

## Areas That Need Help

- **Core tensor ops** — more operations (conv, pool, etc.)
- **Tests** — more edge cases, numerical accuracy tests
- **Benchmarks** — comparison benchmarks vs numpy/PyTorch
- **Documentation** — examples, tutorials, explanations
- **WASM** — browser integration, JS API design
- **Examples** — real model examples (MNIST, XOR, etc.)

## Questions?

Open a GitHub Discussion — we're friendly! 🦀
