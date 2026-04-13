---
id: architecture
title: Architecture
sidebar_label: Architecture
---

# TensorCrab Architecture 🦀

A deep dive into how TensorCrab is designed internally.

---

## Design Principles

1. **Zero-cost abstractions** — high-level API compiles down to machine code with no runtime overhead.
2. **Composability** — every component is a building block that snaps together.
3. **Correctness first** — verified with tests and numerical gradient checks.
4. **No hidden allocations** — memory usage is explicit and predictable.

---

## Layer Overview

```
User Code
    │
    ▼
┌──────────────────────────────┐
│     nn::Module               │  defines forward() and parameters()
│  (Linear, ReLU, Sequential…) │
└──────────────┬───────────────┘
               │ forward(input: &Arc<Variable>) -> Arc<Variable>
               ▼
┌──────────────────────────────┐
│     Arc<Variable>            │  Tensor + grad + grad_fn + requires_grad
└──────────────┬───────────────┘
               │ every op creates a new node and records inputs
               ▼
┌──────────────────────────────┐
│     Computation Graph        │  implicit DAG of all ops since last zero_grad()
│     (dynamic, implicit)      │
└──────────────┬───────────────┘
               │ backward() → topological sort + chain rule
               ▼
┌──────────────────────────────┐
│     Gradient Tensors         │  accumulated in variable.grad
└──────────────┬───────────────┘
               │ optimizer reads .grad from each parameter
               ▼
┌──────────────────────────────┐
│     Optimizer                │  SGD / Adam / AdamW: updates param.data in-place
└──────────────┬───────────────┘
               │ zero_grad() clears all .grad for next iteration
               ▼
         Next Training Step
```

**Build order (never skip a layer):**
```
Storage → Shape → Tensor → Variable → ComputeGraph → Backward → Module → Linear → Activations → Loss → Optimizer
```

---

## Layer 0 — Storage

The lowest level: a reference-counted contiguous block of memory.

```
Arc<Storage<T>>
└── data: Vec<T>   — contiguous block of f32
```

`Arc<Storage<T>>` lets multiple `Tensor`s share the same underlying data without copying. Transpose, reshape, and slicing all share the same `Storage`.

---

## Layer 1 — Tensor

A **view** into a `Storage` with shape, strides, and an offset.

```
Tensor<T>
├── storage: Arc<Storage<T>>   shared memory
├── shape:   Vec<usize>        e.g. [3, 4]
├── strides: Vec<usize>        e.g. [4, 1]  (row-major)
└── offset:  usize             where in storage this view starts
```

### Why strides?

Strides allow zero-copy transforms:

| Operation | Shape | Strides | Notes |
|---|---|---|---|
| Original `[3, 4]` | `[3, 4]` | `[4, 1]` | row-major |
| `transpose()` | `[4, 3]` | `[1, 4]` | no copy — same data |
| `slice(1..)` | `[2, 4]` | `[4, 1]` | offset bumped by 4 |

### Element access formula

```
index(i, j) = offset + i * strides[0] + j * strides[1]
```

### Memory layout diagram

```
┌────────────────────────────────────────────┐
│               Arc<Storage>                 │
│  [f32, f32, f32, f32, f32, f32, f32, f32…] │  ← one contiguous Vec
└────────────────────────────────────────────┘
         ↑             ↑             ↑
    Tensor A       Tensor B      Tensor C
   (full view)   (transposed)    (slice)
   shape:[3,4]   shape:[4,3]    shape:[2,4]
   strides:[4,1] strides:[1,4]  offset:4
```

Multiple tensors can point to the same storage. **No copying unless you call `.contiguous()`.**

---

## Layer 2 — Autograd

### Variable

A `Variable` wraps a `Tensor` and adds:

```rust
pub struct Variable {
    data:          RwLock<Tensor>,       // current tensor value
    grad:          Mutex<Option<Tensor>>, // accumulated gradient
    requires_grad: bool,                  // should we track this?
    node:          Arc<Node>,            // backward node in the graph
}
```

`Variable` is always heap-allocated and reference-counted as `Arc<Variable>`.

Data is accessed via:
- `.data()` → returns an `RwLockReadGuard` (shared read access)
- `.set_data(t)` → replaces the tensor (exclusive write)

### Computation Graph

When you compute `z = x.var_matmul(&y)`, TensorCrab:
1. Computes the forward result (a new `Tensor`).
2. Creates a `MatmulBackward` node that holds saved copies of `x` and `y`.
3. Wraps the result in an `Arc<Variable>` pointing to that node.

```
z = x.var_matmul(&y)

Graph:
  x ──┐
      ├─→ [MatmulBackward] ──→ z
  y ──┘
```

### Backward Pass

`backward(&z)`:
1. Performs a **topological sort** of the graph starting from `z`.
2. Visits nodes in **reverse topological order**.
3. For each node, calls its `backward()` method with the incoming gradient.
4. **Accumulates** computed partial derivatives into each variable's `.grad`.

```rust
pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}
```

Every supported op has a corresponding backward function:

| Forward op | Backward node | Gradient formula |
|---|---|---|
| `var_add(a, b)` | `AddBackward` | `dL/da = dL/dz`, `dL/db = dL/dz` |
| `var_mul(a, b)` | `MulBackward` | `dL/da = dL/dz * b`, `dL/db = dL/dz * a` |
| `var_matmul(a, b)` | `MatmulBackward` | `dL/da = dL/dz @ b.T`, `dL/db = a.T @ dL/dz` |
| `var_relu(x)` | `ReLUBackward` | `dL/dx = dL/dz * (x > 0)` |
| `var_sigmoid(x)` | `SigmoidBackward` | `dL/dx = dL/dz * σ(x) * (1 - σ(x))` |
| `var_sum(x)` | `SumBackward` | `dL/dx = dL/dz * ones_like(x)` |

---

## Layer 3 — Neural Network Modules

`Module` is a trait:

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable>;
    fn parameters(&self) -> Vec<Arc<Variable>>;
    fn zero_grad(&self);
}
```

`Sequential` folds through all layers:

```rust
impl Module for Sequential {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable> {
        self.layers.iter().fold(input.clone(), |x, layer| layer.forward(&x))
    }
}
```

Weight saving/loading serialises each parameter tensor to a flat binary file in layer-parameter order.

---

## Layer 4 — Optimizers

Optimizers receive `Vec<Arc<Variable>>` (the model parameters) and update `.data` in-place after reading `.grad`.

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
```

`Adam` and `AdamW` maintain first and second moment state per parameter as `Vec<Tensor>`, indexed in the same order as the parameter list. This is why they need `&mut self`.

---

## Thread Safety

All public types implement `Send + Sync`:
- `Storage` is `Arc`-wrapped.
- `Variable.data` is protected by `RwLock`.
- `Variable.grad` is protected by `Mutex`.
- All `BackwardFn` implementations are `Send + Sync`.

Computation graphs are currently built on a single thread. True parallel autograd is a planned future feature.

---

## Error Handling

TensorCrab uses `thiserror` for typed errors — no panics in library code for user-facing operations:

```rust
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    #[error("reshape error: cannot reshape {from:?} → {to:?}")]
    ReshapeError { from: Vec<usize>, to: Vec<usize> },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```
