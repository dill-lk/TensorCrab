---
id: tensor
title: Tensor Engine
sidebar_label: Tensor Engine
---

# Tensor Engine 🧮

`Tensor<T>` is the foundation of TensorCrab — an N-dimensional array that supports broadcasting, matrix ops, and reductions with zero-copy views.

---

## Creating Tensors

```rust
use tensor_crab::tensor::Tensor;

// From a flat Vec with an explicit shape
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// All zeros
let z = Tensor::zeros(&[3, 4]);

// All ones
let o = Tensor::ones(&[2, 2]);

// Fill with a constant
let f = Tensor::full(7.0_f32, &[3, 3]);

// Random normal (non-deterministic)
let r = Tensor::randn(&[4, 4]);

// Random normal with a fixed seed (deterministic — use in tests)
let rs = Tensor::randn_seeded(&[4, 4], 42);
```

---

## Inspecting a Tensor

```rust
let t = Tensor::from_vec(vec![1.0_f32; 12], &[3, 4]);

println!("{:?}", t.shape());    // [3, 4]
println!("{:?}", t.strides());  // [4, 1]
println!("{}", t.ndim());       // 2
println!("{}", t.numel());      // 12
println!("{}", t.is_empty());   // false

// Get element at position [1, 2]
let val = t.get_at(&[1, 2]);    // 1.0

// Convert to Vec
let v: Vec<f32> = t.to_vec();
```

---

## Element-Wise Operations

All element-wise ops return `Result<Tensor<f32>, TensorError>` and support **broadcasting**.

```rust
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0, 40.0], &[2, 2]);

let add  = a.add(&b).unwrap();   // [[11, 22], [33, 44]]
let sub  = a.sub(&b).unwrap();   // [[-9, -18], [-27, -36]]
let mul  = a.mul(&b).unwrap();   // [[10, 40], [90, 160]]
let div  = a.div(&b).unwrap();   // [[0.1, 0.1], [0.1, 0.1]]
```

### Scalar Operations

```rust
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);

let scaled = a.scalar_mul(3.0); // [3.0, 6.0, 9.0]
let offset = a.scalar_add(1.0); // [2.0, 3.0, 4.0]
```

### Unary Operations

```rust
let a = Tensor::from_vec(vec![-1.0_f32, 0.0, 2.0, 3.0], &[4]);

let neg  = a.neg();   // [1.0, 0.0, -2.0, -3.0]
let relu = a.relu();  // [0.0, 0.0,  2.0,  3.0]
let sig  = a.sigmoid();
let tanh = a.tanh();
let exp  = a.exp();
let log  = a.log();   // NaN for non-positive values
let sqrt = a.sqrt();
```

---

## Matrix Operations

```rust
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);

// Matrix multiplication — panics with TensorError on shape mismatch
let c = a.matmul(&b).unwrap(); // [[19, 22], [43, 50]]

// Transpose — zero-copy (only swaps strides)
let t = a.transpose(); // [[1, 3], [2, 4]]

// Reshape — zero-copy if contiguous
let flat = a.reshape(&[4]).unwrap(); // [1, 2, 3, 4]

// Flatten to 1D
let flat2 = a.flatten();
```

---

## Reduction Operations

```rust
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

let s = a.sum();   // 21.0
let m = a.mean();  // 3.5
let mx = a.max();  // 6.0
let mn = a.min();  // 1.0

// Along an axis, keeping dimensions
let row_sum = a.sum_axis(1); // [6.0, 15.0]
```

---

## Broadcasting

Broadcasting follows the same rules as NumPy. Shapes are aligned from the right, and dimensions of size 1 are broadcast to match the other operand:

```rust
// [2, 1] + [1, 3] → [2, 3]
let col = Tensor::from_vec(vec![10.0_f32, 20.0], &[2, 1]);
let row = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
let out = col.add(&row).unwrap();
// [[11, 12, 13],
//  [21, 22, 23]]

// [3] + scalar → broadcast
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]);
let b = Tensor::from_vec(vec![100.0_f32], &[1]);
let c = a.add(&b).unwrap(); // [101, 102, 103]
```

---

## Views and Memory

Transpose and reshape return **views** — they share the same underlying memory with different shape/stride metadata. No copying unless you explicitly call `.contiguous()`.

```rust
let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);

let t = a.transpose();      // zero-copy — shares storage with `a`
let c = t.contiguous();     // forces a new contiguous copy
println!("{}", t.is_contiguous()); // false
println!("{}", c.is_contiguous()); // true
```

---

## Display

Tensors implement `Display` for readable output:

```rust
let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
println!("{t}");
// [[1.0, 2.0, 3.0],
//  [4.0, 5.0, 6.0]]
```

---

## Error Handling

Tensor operations that can fail (shape mismatches, etc.) return `Result<Tensor<f32>, TensorError>`:

```rust
use tensor_crab::TensorError;

let a = Tensor::from_vec(vec![1.0_f32; 6], &[2, 3]);
let b = Tensor::from_vec(vec![1.0_f32; 4], &[2, 2]);

match a.matmul(&b) {
    Ok(c)  => println!("Result: {c}"),
    Err(e) => println!("Error: {e}"),
    // Error: shape mismatch: inner dims 3 ≠ 2
}
```

See the [Tensor API Reference →](./api/tensor) for a complete method list.
