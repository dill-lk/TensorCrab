# TensorCrab — Development Plan 🦀

> **Agent Rule:** If your implementation differs from what's described here, update this file to match what was actually built. This doc should always reflect reality, not the original plan.

---

## Current Implementation State

> Agents: keep this section accurate.

| Module | File(s) | Status |
|---|---|---|
| Storage | `src/tensor/data.rs` | 🟢 Done |
| Shape/Stride | `src/tensor/shape.rs` | 🟢 Done |
| Tensor struct | `src/tensor/mod.rs` | 🟢 Done |
| Tensor ops | `src/tensor/ops.rs` | 🟢 Done |
| Variable | `src/autograd/variable.rs` | 🟢 Done |
| Compute graph | `src/autograd/graph.rs` | 🟢 Done |
| Backward pass | `src/autograd/engine.rs` | 🟢 Done |
| Autograd ops | `src/autograd/ops.rs` | 🟢 Done |
| Module trait | `src/nn/mod.rs` | 🟢 Done |
| Linear layer | `src/nn/linear.rs` | 🟢 Done |
| Activations | `src/nn/activations.rs` | 🟢 Done |
| Sequential | `src/nn/sequential.rs` | 🟢 Done |
| Loss functions | `src/nn/loss.rs` | 🟢 Done |
| BatchNorm1d | `src/nn/batchnorm.rs` | 🟢 Done |
| Dropout | `src/nn/dropout.rs` | 🟢 Done |
| Optimizer trait | `src/optim/mod.rs` | 🟢 Done |
| SGD | `src/optim/sgd.rs` | 🟢 Done |
| Adam | `src/optim/adam.rs` | 🟢 Done |
| AdamW | `src/optim/adamw.rs` | 🟢 Done |
| Schedulers | `src/optim/scheduler.rs` | 🟢 Done |
| DataLoader | `src/optim/dataloader.rs` | 🟢 Done |

---

## Repository Structure

```
tensor-crab/
├── Cargo.toml
├── README.md
├── LICENSE
├── docs/
│   ├── what-is.md
│   ├── roadmap.md       ← update after every feature
│   ├── plan.md          ← update if impl differs from plan
│   ├── agents.md        ← read before contributing
│   └── architecture.md
├── crates/
│   └── tensor-crab/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── tensor/
│           │   ├── mod.rs
│           │   ├── data.rs
│           │   ├── ops.rs
│           │   └── shape.rs
│           ├── autograd/
│           │   ├── mod.rs
│           │   ├── variable.rs
│           │   ├── graph.rs
│           │   └── backward.rs
│           ├── nn/
│           │   ├── mod.rs
│           │   ├── linear.rs
│           │   ├── activations.rs
│           │   └── loss.rs
│           └── optim/
│               ├── mod.rs
│               ├── sgd.rs
│               └── adam.rs
├── examples/
│   ├── xor.rs
│   ├── mnist.rs
│   └── linear_regression.rs
├── benches/
│   └── matmul.rs
└── tests/
    ├── tensor_tests.rs
    └── autograd_tests.rs
```

---

## Cargo.toml

```toml
[package]
name = "tensor-crab"
version = "0.1.0"
edition = "2021"
description = "Rust-native ML library. No Python. No GIL. Just speed."
license = "Apache-2.0"
repository = "https://github.com/yourusername/tensor-crab"

[dependencies]
thiserror = "1.0"
rayon = "1.8"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }
```

---

## Phase 1 — Tensor Engine

### Step 1: Raw storage (`src/tensor/data.rs`)
```rust
pub struct Storage<T> {
    data: Vec<T>,
}

impl<T> Storage<T> {
    pub fn new(data: Vec<T>) -> Self { Self { data } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn as_slice(&self) -> &[T] { &self.data }
}
```

### Step 2: Shape and strides (`src/tensor/shape.rs`)
```rust
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Shape {
    pub fn row_major(dims: &[usize]) -> Self {
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Self { dims: dims.to_vec(), strides }
    }
    pub fn numel(&self) -> usize { self.dims.iter().product() }
}
```

### Step 3: Tensor struct (`src/tensor/mod.rs`)
```rust
use std::sync::Arc;

pub struct Tensor<T = f32> {
    storage: Arc<Storage<T>>,
    shape: Shape,
    offset: usize,
}

impl<T: Float> Tensor<T> {
    pub fn zeros(shape: &[usize]) -> Self { ... }
    pub fn ones(shape: &[usize]) -> Self { ... }
    pub fn randn(shape: &[usize]) -> Self { ... }
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self { ... }
}
```

### Step 4: Operations (`src/tensor/ops.rs`)
```rust
impl<T: Float> Tensor<T> {
    pub fn add(&self, other: &Self) -> Self { ... }
    pub fn sub(&self, other: &Self) -> Self { ... }
    pub fn mul(&self, other: &Self) -> Self { ... }
    pub fn div(&self, other: &Self) -> Self { ... }
    pub fn matmul(&self, other: &Self) -> Self { ... }
    pub fn transpose(&self) -> Self { ... }
    pub fn reshape(&self, shape: &[usize]) -> Self { ... }
    pub fn sum(&self) -> Self { ... }
    pub fn mean(&self) -> Self { ... }
}
```

---

## Phase 2 — Autograd

### Computation graph node
```rust
pub struct Node {
    pub grad_fn: Option<Box<dyn BackwardFn>>,
    pub inputs: Vec<Weak<RefCell<Node>>>,
}

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor>;
}
```

### Variable
```rust
pub struct Variable {
    pub data: Tensor,
    pub grad: RefCell<Option<Tensor>>,
    pub requires_grad: bool,
    pub node: Arc<RefCell<Node>>,
}
```

### Backward
```rust
pub fn backward(root: &Variable) {
    let order = topological_sort(root);
    for var in order.iter().rev() {
        if let Some(grad_fn) = &var.node.borrow().grad_fn {
            let grads = grad_fn.backward(&var.grad.borrow().clone().unwrap());
            // accumulate into input vars
        }
    }
}
```

---

## Testing Strategy

Every op gets a numerical gradient check:

```rust
fn numerical_grad(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, eps: f32) -> Tensor {
    // (f(x + eps) - f(x - eps)) / (2 * eps)
}

#[test]
fn test_matmul_grad() {
    let x = Variable::randn(&[3, 4], true);
    let y = Variable::randn(&[4, 5], true);
    let z = x.matmul(&y).sum();
    z.backward();
    let num = numerical_grad(|x| x.matmul(&y.data).sum(), &x.data, 1e-5);
    assert_abs_diff_eq!(x.grad(), num, epsilon = 1e-4);
}
```

---

## First Commands

```bash
# Bootstrap the project
cargo new tensor-crab --lib
cd tensor-crab

# Verify it builds
cargo build

# Run tests
cargo test

# Lint
cargo clippy

# Format
cargo fmt
```
