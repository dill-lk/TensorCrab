# TensorCrab Architecture 🦀

A deep dive into how TensorCrab is designed internally.

## Design Principles

1. **Zero-cost abstractions** — high-level API compiles down to machine code with no runtime overhead
2. **Composability** — every component is a building block that snaps together
3. **Correctness first** — verified with tests and numeric gradient checks
4. **No hidden allocations** — memory usage is explicit and predictable

---

## Layer 0 — Storage

The lowest level. Raw memory.

```
Storage<T>
├── data: Vec<T>       // contiguous block of memory
└── len: usize
```

Storage is ref-counted via `Arc<Storage<T>>` so tensors can share memory without copying (views, slices, transpose all share the same underlying data).

---

## Layer 1 — Tensor

A view into a Storage with a shape and stride.

```
Tensor<T>
├── storage: Arc<Storage<T>>   // shared memory
├── shape: Vec<usize>          // e.g. [3, 4] for a 3x4 matrix
├── strides: Vec<usize>        // e.g. [4, 1] for row-major
└── offset: usize              // where in storage this view starts
```

### Why strides?

Strides let us implement transpose, slicing, and reshaping **without copying data**:

- A `[3, 4]` tensor has strides `[4, 1]` (row-major)
- Transposing gives shape `[4, 3]` and strides `[1, 4]` — same data, different view
- Slicing `tensor[1..]` just changes the offset — no copy

### Element access

```
index(i, j) = offset + i * strides[0] + j * strides[1]
```

---

## Layer 2 — Autograd

The computation graph sits on top of Tensor.

### Variables

A `Variable` wraps a `Tensor` and adds:
- `grad: Option<Tensor>` — accumulated gradient
- `requires_grad: bool` — should we track this?
- `grad_fn: Option<Arc<dyn BackwardFn>>` — how to compute gradient

### The Graph

When you do `let z = x.matmul(&y)`, TensorCrab:
1. Computes the forward result (a new Tensor)
2. Creates a `MatmulBackward` node that holds references to `x` and `y`
3. Wraps the result in a `Variable` pointing to that node

This builds a DAG (directed acyclic graph) implicitly as you compute.

```
z = x.matmul(y)

Graph:
  x ──┐
      ├─→ [MatmulBackward] ──→ z
  y ──┘
```

### Backward Pass

`z.backward()` does a topological sort of the graph and visits each node in reverse order, calling `backward()` on each and passing the incoming gradient.

```rust
pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

struct MatmulBackward {
    lhs: Tensor,   // saved x
    rhs: Tensor,   // saved y
}

impl BackwardFn for MatmulBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        // d(loss)/dx = grad @ y.T
        // d(loss)/dy = x.T @ grad
        vec![
            grad.matmul(&self.rhs.transpose()),
            self.lhs.transpose().matmul(grad),
        ]
    }
}
```

---

## Layer 3 — Neural Network Modules

Built on top of Variables.

### The Module Trait

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<&Variable>;
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
```

### Sequential — compose modules

```rust
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Variable {
        self.layers.iter().fold(input.clone(), |x, layer| layer.forward(&x))
    }
}
```

---

## Layer 4 — Optimizers

Optimizers take the parameters (Variables with gradients) and update them.

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
```

Adam maintains state (first and second moment estimates) per parameter, which is why it needs `&mut self`.

---

## Memory Layout

```
┌─────────────────────────────────────────┐
│              Arc<Storage>               │
│  [f32, f32, f32, f32, f32, f32, f32...] │  ← one contiguous Vec
└─────────────────────────────────────────┘
        ↑           ↑           ↑
   Tensor A     Tensor B     Tensor C
  (full view)  (transposed)  (slice)
  shape:[3,4]  shape:[4,3]  shape:[2,4]
  strides:[4,1] strides:[1,4] offset:4
```

Multiple tensors can point to the same storage with different shapes/strides/offsets. **No copying unless you explicitly call `.contiguous()`.**

---

## Thread Safety

All public types implement `Send + Sync`. Storage is `Arc`-wrapped so it can be safely shared across threads. Autograd graphs are built on a single thread (like PyTorch's default mode) — parallel autograd is a future feature.

---

## Error Handling

TensorCrab uses `thiserror` for typed errors — no panics in library code (except genuine programmer errors like shape mismatches, which panic with a clear message).

```rust
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },
}
```
