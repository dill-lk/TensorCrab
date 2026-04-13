---
id: variable
title: Variable API Reference
sidebar_label: Variable
---

# Variable API Reference

`tensor_crab::autograd::Variable`

A reference-counted wrapper around a `Tensor` that participates in automatic differentiation. Always handled as `Arc<Variable>`.

---

## Constructor

| Method | Signature | Description |
|---|---|---|
| `new` | `(data: Tensor, requires_grad: bool) -> Arc<Variable>` | Create a new variable. Set `requires_grad = true` for parameters you want to differentiate through. |

---

## Data Access

| Method | Return type | Description |
|---|---|---|
| `data()` | `RwLockReadGuard<Tensor>` | Read-only access to the underlying tensor |
| `set_data(Tensor)` | `()` | Replace the tensor data in-place (used by optimizers) |
| `requires_grad()` | `bool` | Whether this variable participates in gradient computation |

---

## Gradient

| Method | Return type | Description |
|---|---|---|
| `grad()` | `Option<Tensor>` | Accumulated gradient after `backward()`. `None` if not yet computed or `requires_grad = false`. |
| `zero_grad()` | `()` | Reset gradient to zero. Call before each training step. |

---

## Arithmetic Operations

All operations return `Arc<Variable>`. If any input `requires_grad`, the result also requires grad and a backward node is recorded.

| Method | Description |
|---|---|
| `var_add(&other)` | Element-wise addition |
| `var_sub(&other)` | Element-wise subtraction |
| `var_mul(&other)` | Element-wise multiplication |
| `var_div(&other)` | Element-wise division |
| `var_add_scalar(f32)` | Add scalar to every element |
| `var_mul_scalar(f32)` | Multiply every element by scalar |
| `var_neg()` | Negate every element |

---

## Matrix Operations

| Method | Description |
|---|---|
| `var_matmul(&other)` | Matrix multiplication (requires 2-D tensors) |
| `var_transpose()` | Transpose (zero-copy view) |

---

## Activation Functions

| Method | Formula |
|---|---|
| `var_relu()` | `max(0, x)` element-wise |
| `var_sigmoid()` | `1 / (1 + exp(-x))` element-wise |
| `var_tanh()` | `tanh(x)` element-wise |
| `var_exp()` | `exp(x)` element-wise |
| `var_log()` | Natural log element-wise |
| `var_sqrt()` | Square root element-wise |

---

## Reductions

| Method | Description |
|---|---|
| `var_sum()` | Reduce to scalar (sum of all elements) |
| `var_sum_keepdim(axis: usize)` | Sum along one axis, keeping that dimension |

---

## Backward

```rust
use tensor_crab::autograd::backward;

// Computes gradients for all variables in the graph leading to `loss`.
// `loss` must be a scalar (numel() == 1).
backward(&loss);
```

`backward` performs a topological sort and visits each node in reverse order, accumulating gradients via the chain rule.

---

## Example

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};

// w and x are parameters
let w = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
let x = Variable::new(Tensor::from_vec(vec![1.0_f32, 1.0], &[2]), false);

// z = sum(w * x)
let z = w.var_mul(&x).var_sum();
backward(&z);

// dz/dw = x = [1.0, 1.0]
println!("{:?}", w.grad().unwrap().to_vec());
```
