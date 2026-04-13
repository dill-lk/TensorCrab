# TensorCrab 🦀

**A blazing-fast machine learning library, written entirely in Rust.**

*No Python. No GIL. No overhead. Just pure, type-safe speed.*

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dill-lk/TensorCrab/blob/main/LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)

---

## What is TensorCrab?

TensorCrab gives you everything you need to build and train neural networks — N-dimensional tensors, automatic differentiation, composable layers, and gradient-based optimizers — all in a single Rust library with zero Python dependency.

Think of it as PyTorch, but:

- 🔒 **Memory-safe by default** — the compiler catches bugs before they ship
- ⚡ **Native speed** — no interpreter, no GIL, no FFI overhead
- 📦 **Single binary** — deploy as one self-contained executable
- 🦀 **Idiomatic Rust** — works naturally with the rest of your Rust project

---

## Quick Start

```rust
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::optim::{Optimizer, Adam};
use tensor_crab::tensor::Tensor;

// Build a 3-layer network
let model = Sequential::new(vec![
    Box::new(Linear::new(2, 16)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(16, 1)),
]);

// One training step
let mut opt = Adam::new(model.parameters(), 0.001);

let x      = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0], &[1, 2]), false);
let target = Variable::new(Tensor::from_vec(vec![1.0_f32],      &[1, 1]), false);

let pred = model.forward(&x);
let l    = loss::mse_loss(&pred, &target);
backward(&l);   // gradients flow through the whole network automatically
opt.step();
opt.zero_grad();
```

---

## Features

### 🧮 Tensors

N-dimensional arrays with broadcasting, matrix ops, and reductions.

```rust
use tensor_crab::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);

let c = a.matmul(&b).unwrap();  // [[19, 22], [43, 50]]
let s = a.sum();                // 10.0
```

### 🔁 Autograd

Every operation is recorded in a computation graph. Call `backward()` once and gradients flow everywhere.

```rust
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
let loss = x.var_mul(&x).var_sum();  // loss = x[0]² + x[1]²
backward(&loss);

let grad = x.grad().unwrap();  // [4.0, 6.0]
```

### 🧠 Neural Network Layers

| Layer | Description |
|---|---|
| `Linear` | Fully connected layer with Xavier / Kaiming init |
| `ReLU`, `Sigmoid`, `Tanh` | Element-wise activations |
| `Softmax` | Numerically stable probability distribution |
| `BatchNorm1d` | Batch normalisation with learnable γ and β |
| `Dropout` | Inverted dropout with configurable probability |
| `Sequential` | Chain layers into a model |

### 📉 Loss Functions

```rust
use tensor_crab::nn::loss;

let l = loss::mse_loss(&pred, &target);           // regression
let l = loss::bce_loss(&pred, &target);           // binary classification
let l = loss::cross_entropy_loss(&logits, &tgt);  // multi-class
```

### 🚀 Optimizers

`SGD`, `Adam`, and `AdamW` with optional momentum, weight decay, and learning rate schedulers (`StepLR`, `CosineAnnealingLR`).

### 💾 Save & Load

```rust
model.save_weights("checkpoint.bin").unwrap();
model.load_weights("checkpoint.bin").unwrap();
```

### ⚡ CUDA / GPU (optional)

Enable GPU acceleration by building with the `cuda` feature:

```toml
[dependencies]
tensor-crab = { version = "0.1", features = ["cuda"] }
```

Requires an NVIDIA GPU, CUDA Toolkit ≥ 11.x, and the `CUDA_PATH` environment variable set.

---

## Installation

```toml
[dependencies]
tensor-crab = "0.1"
```

---

## Roadmap

| Stage | Status |
|---|---|
| 🧮 Tensor Engine | 🟢 Done |
| 🔁 Autograd | 🟢 Done |
| 🧠 NN Layers | 🟢 Done |
| 🚀 Optimizers | 🟢 Done |
| ⚡ CUDA / GPU | 🟢 Done |
| 🌐 WebAssembly | 🔴 Planned |
| 🌍 Ecosystem | 🔴 Planned |

---

## License

Apache 2.0 — see [LICENSE](https://github.com/dill-lk/TensorCrab/blob/main/LICENSE).
