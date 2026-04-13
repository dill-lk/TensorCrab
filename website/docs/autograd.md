---
id: autograd
title: Autograd — Automatic Differentiation
sidebar_label: Autograd
---

# Autograd — Automatic Differentiation 🔁

TensorCrab's autograd engine computes gradients automatically using **reverse-mode automatic differentiation** (backpropagation). As you perform operations on `Variable`s, TensorCrab silently builds a dynamic computation graph (DAG). When you call `backward()`, it traverses the graph in reverse topological order and accumulates gradients using the chain rule.

---

## Variables

A `Variable` is a `Tensor` wrapped with gradient bookkeeping. It is always heap-allocated and reference-counted (`Arc<Variable>`):

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::Variable;

// requires_grad = true  → track gradients for this variable
let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);

// requires_grad = false → treat as a constant (no gradient)
let c = Variable::new(Tensor::from_vec(vec![1.0_f32, 1.0], &[2]), false);
```

### Accessing Data and Gradients

```rust
// Read the underlying tensor (returns an RwLock read guard)
let data = x.data();
println!("{:?}", data.shape()); // [2]

// Read the accumulated gradient after backward()
if let Some(grad) = x.grad() {
    println!("{:?}", grad.to_vec());
}

// Reset gradient to zero
x.zero_grad();
```

---

## Forward Operations

All operations on `Arc<Variable>` record themselves in the graph. The result is a new `Arc<Variable>` that points to the backward node:

```rust
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
let y = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);

// Arithmetic
let add  = x.var_add(&y);
let sub  = x.var_sub(&y);
let mul  = x.var_mul(&y);
let div  = x.var_div(&y);

// Scalar operations
let scaled = x.var_mul_scalar(3.0);
let offset = x.var_add_scalar(1.0);

// Matrix operations (requires 2-D tensors)
let w = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]), true);
let v = Variable::new(Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], &[2, 2]), true);
let mm = w.var_matmul(&v);
let tr = w.var_transpose();

// Activations
let relu    = x.var_relu();
let sigmoid = x.var_sigmoid();
let tanh    = x.var_tanh();
let neg     = x.var_neg();
let log     = x.var_log();
let exp     = x.var_exp();
let sqrt    = x.var_sqrt();

// Reductions
let sum  = x.var_sum();              // scalar
let ssum = x.var_sum_keepdim(0);     // keeps the axis dimension
```

---

## Backward Pass

Call `backward()` on any scalar `Variable` to propagate gradients back through the entire graph:

```rust
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);

// z = sum(x²)
let z = x.var_mul(&x).var_sum();

// Compute dz/dx = 2x
backward(&z);

let grad = x.grad().unwrap();
println!("{:?}", grad.to_vec()); // [4.0, 6.0]
```

:::info
`backward()` expects a **scalar** — a `Variable` whose underlying tensor contains a single element. Use `.var_sum()` or `.var_mean()` to reduce a loss to a scalar before calling `backward()`.
:::

---

## Multi-Variable Graphs

Gradients flow through any graph, no matter how complex:

```rust
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 2.0], &[2]), true);
let w = Variable::new(Tensor::from_vec(vec![3.0_f32, 4.0], &[2]), true);
let b = Variable::new(Tensor::from_vec(vec![0.5_f32], &[1]), true);

// y = sum(w * x) + b
let y = w.var_mul(&x).var_sum().var_add_scalar(0.5);

// Scalar loss
let loss = y.var_mul_scalar(1.0); // identity
backward(&loss);

// dL/dw = x, dL/dx = w
println!("{:?}", w.grad().unwrap().to_vec()); // [1.0, 2.0]
println!("{:?}", x.grad().unwrap().to_vec()); // [3.0, 4.0]
```

---

## Resetting Gradients

Call `zero_grad()` on each parameter before the next training step to prevent gradient accumulation:

```rust
// After opt.step():
for param in model.parameters() {
    param.zero_grad();
}

// Or use the optimizer's zero_grad():
opt.zero_grad();
```

---

## How It Works Internally

Every op creates a backward node that captures the inputs needed to compute gradients:

```
z = x.var_mul(&x).var_sum()

Computation graph:
  x ──┐
      ├─→ [MulBackward] ─→ product ─→ [SumBackward] ─→ z
  x ──┘
```

When `backward(&z)` is called:
1. A **topological sort** of the graph is computed.
2. Nodes are visited in **reverse order** (from `z` back to `x`).
3. Each node's `backward()` method is called with the incoming gradient, computing the local partial derivatives.
4. Results are **accumulated** into each variable's `.grad` field.

See [Architecture →](./architecture) for a deeper dive into the internals.
