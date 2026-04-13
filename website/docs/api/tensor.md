---
id: tensor
title: Tensor API Reference
sidebar_label: Tensor
---

# Tensor API Reference

`tensor_crab::tensor::Tensor<T>`

A generic N-dimensional array. The default type parameter is `f32`.

---

## Constructors

| Method | Signature | Description |
|---|---|---|
| `from_vec` | `(data: Vec<T>, shape: &[usize]) -> Tensor<T>` | Create from flat Vec with explicit shape |
| `zeros` | `(shape: &[usize]) -> Tensor<T>` | All-zero tensor |
| `ones` | `(shape: &[usize]) -> Tensor<T>` | All-ones tensor |
| `full` | `(value: T, shape: &[usize]) -> Tensor<T>` | Fill with a constant |
| `randn` | `(shape: &[usize]) -> Tensor<T>` | Random normal (non-deterministic) |
| `randn_seeded` | `(shape: &[usize], seed: u64) -> Tensor<T>` | Random normal with fixed seed |

---

## Inspection

| Method | Return type | Description |
|---|---|---|
| `shape()` | `&[usize]` | Shape as a slice |
| `strides()` | `&[usize]` | Strides (row-major by default) |
| `ndim()` | `usize` | Number of dimensions |
| `numel()` | `usize` | Total number of elements |
| `is_empty()` | `bool` | True if `numel() == 0` |
| `is_contiguous()` | `bool` | True if layout matches row-major strides |
| `get_at(&[usize])` | `T` | Element at the given multi-dimensional index |
| `to_vec()` | `Vec<T>` | Copy all elements into a flat Vec |

---

## Element-Wise Operations

All return `Result<Tensor<f32>, TensorError>` and support broadcasting.

| Method | Formula | Notes |
|---|---|---|
| `add(&other)` | `a + b` | Broadcasts if shapes are compatible |
| `sub(&other)` | `a - b` | Broadcasts if shapes are compatible |
| `mul(&other)` | `a * b` | Element-wise, not matrix multiply |
| `div(&other)` | `a / b` | Element-wise |

---

## Scalar Operations

| Method | Return | Description |
|---|---|---|
| `scalar_mul(f32)` | `Tensor<f32>` | Multiply every element by scalar |
| `scalar_add(f32)` | `Tensor<f32>` | Add scalar to every element |

---

## Unary Operations

| Method | Return | Description |
|---|---|---|
| `neg()` | `Tensor<f32>` | Negate every element |
| `relu()` | `Tensor<f32>` | `max(0, x)` element-wise |
| `sigmoid()` | `Tensor<f32>` | `1 / (1 + exp(-x))` element-wise |
| `tanh()` | `Tensor<f32>` | `tanh(x)` element-wise |
| `exp()` | `Tensor<f32>` | `exp(x)` element-wise |
| `log()` | `Tensor<f32>` | Natural log element-wise |
| `sqrt()` | `Tensor<f32>` | Square root element-wise |

---

## Matrix Operations

| Method | Return | Description |
|---|---|---|
| `matmul(&other)` | `Result<Tensor<f32>, TensorError>` | Matrix multiply; requires compatible inner dims |
| `transpose()` | `Tensor<f32>` | Zero-copy transpose (swaps last two dims' strides) |
| `reshape(&[usize])` | `Result<Tensor<f32>, TensorError>` | Reinterpret shape; zero-copy if contiguous |
| `flatten()` | `Tensor<f32>` | Collapse to 1D |

---

## Reduction Operations

| Method | Return | Description |
|---|---|---|
| `sum()` | `Tensor<f32>` | Scalar — sum of all elements |
| `mean()` | `Tensor<f32>` | Scalar — mean of all elements |
| `max()` | `Tensor<f32>` | Scalar — maximum element |
| `min()` | `Tensor<f32>` | Scalar — minimum element |
| `sum_axis(axis: usize)` | `Tensor<f32>` | Sum along axis, removing that dimension |

---

## Memory

| Method | Return | Description |
|---|---|---|
| `contiguous()` | `Tensor<T>` | Force a contiguous copy (use when views interfere) |
| `is_contiguous()` | `bool` | Whether the tensor is already row-major contiguous |

---

## Display

`Tensor<T>` implements `std::fmt::Display`:

```rust
let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
println!("{t}");
// [[1.0, 2.0],
//  [3.0, 4.0]]
```

---

## Errors

| Variant | Triggered by |
|---|---|
| `TensorError::ShapeMismatch` | `matmul` with incompatible inner dims |
| `TensorError::BroadcastError` | element-wise ops on incompatible shapes |
| `TensorError::ReshapeError` | `reshape` that changes `numel()` |
| `TensorError::Io` | `save_weights` / `load_weights` failures |
