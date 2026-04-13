<div>
  <img src="https://github.com/user-attachments/assets/e2390b00-39ed-4d66-9257-700f8aa10897" alt="tensorcrab_ascii_svg" />

  
</div>

<div align="center">

**A blazing-fast machine learning library, written entirely in Rust.**

*No Python. No GIL. No overhead. Just pure, type-safe speed.*

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](#)

</div>

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

### Train a model in 10 lines

```rust
use tensor_crab::prelude::*;
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::autograd::{Variable, backward};
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
opt.step();     // update every weight
opt.zero_grad();
```

---

## Features

### 🧮 Tensors
N-dimensional arrays with broadcasting, matrix ops, and reductions — the foundation everything else builds on.

```rust
use tensor_crab::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);

let c = a.matmul(&b).unwrap();  // [[19, 22], [43, 50]]
let s = a.sum();                // 10.0

// Broadcasting: [2, 1] + [1, 3] → [2, 3]
let row = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[1, 3]);
let col = Tensor::from_vec(vec![10.0_f32, 20.0],    &[2, 1]);
let out = col.add(&row).unwrap();  // [[11, 12, 13], [21, 22, 23]]
```

### 🔁 Autograd
Every operation records itself in a computation graph. Call `backward()` once and gradients flow everywhere automatically.

```rust
use std::sync::Arc;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);

let loss = x.var_mul(&x).var_sum();  // loss = x[0]² + x[1]²
backward(&loss);

let grad = x.grad().unwrap();
// grad = 2x = [4.0, 6.0]
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

```rust
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, Dropout, BatchNorm1d};

let model = Sequential::new(vec![
    Box::new(Linear::new(784, 256)),
    Box::new(BatchNorm1d::new(256)),
    Box::new(ReLU::new()),
    Box::new(Dropout::new(0.3)),
    Box::new(Linear::new(256, 10)),
]);
```

### 📉 Loss Functions

```rust
use tensor_crab::nn::loss;

// Regression
let l = loss::mse_loss(&pred, &target);

// Binary classification
let l = loss::bce_loss(&pred, &target);

// Multi-class classification
let l = loss::cross_entropy_loss(&logits, &one_hot_target);
```

### 🚀 Optimizers

| Optimizer | When to use |
|---|---|
| `SGD` | Simple tasks; can add momentum for faster convergence |
| `Adam` | Most deep learning tasks — adaptive learning rate |
| `AdamW` | Transformers and networks where weight decay matters |

```rust
use tensor_crab::optim::{Optimizer, SGD, Adam, AdamW};

// SGD with momentum
let mut opt = SGD::new(model.parameters(), 0.01)
    .with_momentum(0.9)
    .with_weight_decay(1e-4);

// Adam (sensible defaults: β₁=0.9, β₂=0.999)
let mut opt = Adam::new(model.parameters(), 0.001);

// AdamW with decoupled weight decay
let mut opt = AdamW::new(model.parameters(), 0.001)
    .with_weight_decay(0.01);
```

### 📅 Learning Rate Schedulers

Schedulers compute the learning rate for each epoch — just assign `opt.lr`:

```rust
use tensor_crab::optim::scheduler::{StepLR, CosineAnnealingLR};

// Halve the lr every 10 epochs
let scheduler = StepLR::new(0.1, 10, 0.5);

// Smooth cosine decay from 0.1 → 0.0 over 50 epochs
let scheduler = CosineAnnealingLR::new(0.1, 0.0, 50);

for epoch in 0..100 {
    opt.lr = scheduler.get_lr(epoch);
    // ... train ...
}
```

### 📦 DataLoader

Load, batch, and shuffle your dataset automatically:

```rust
use tensor_crab::optim::DataLoader;
use tensor_crab::tensor::Tensor;

let x = Tensor::from_vec(/* 1000 samples × 16 features */, &[1000, 16]);
let y = Tensor::from_vec(/* 1000 labels */, &[1000, 1]);

let loader = DataLoader::new(x, y, /*batch_size=*/ 32, /*shuffle=*/ true);

for (x_batch, y_batch) in loader.iter_epoch(None) {
    // x_batch: [32, 16], y_batch: [32, 1]
}
```

### 💾 Save & Load

```rust
model.save_weights("checkpoint.bin").unwrap();

// Later, restore on a model with the same architecture:
let tensors = model.load_weights("checkpoint.bin").unwrap();
```

---

## A Full Training Loop

```rust
use std::sync::Arc;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::optim::{DataLoader, Optimizer, Adam};
use tensor_crab::optim::scheduler::CosineAnnealingLR;
use tensor_crab::tensor::Tensor;

fn main() {
    // Model
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 32)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(32, 1)),
    ]);

    // Optimizer + scheduler
    let mut opt = Adam::new(model.parameters(), 0.01);
    let scheduler = CosineAnnealingLR::new(0.01, 1e-5, 50);

    // Data
    let x = Tensor::randn_seeded(&[200, 2], 42);
    let y = Tensor::zeros(&[200, 1]);
    let loader = DataLoader::new(x, y, 32, true);

    // Training loop
    for epoch in 0..50 {
        opt.lr = scheduler.get_lr(epoch);

        for (x_batch, y_batch) in loader.iter_epoch(Some(epoch as u64)) {
            let x_var = Variable::new(x_batch, false);
            let y_var = Variable::new(y_batch, false);

            let pred = model.forward(&x_var);
            let l    = loss::mse_loss(&pred, &y_var);

            backward(&l);
            opt.step();
            opt.zero_grad();
        }
    }

    model.save_weights("model.bin").unwrap();
    println!("Training complete.");
}
```

---

## Installation

> TensorCrab is not yet published to crates.io.  
> To use it today, add it as a path or git dependency:

```toml
[dependencies]
tensor-crab = { git = "https://github.com/dill-lk/TensorCrab" }
```

---

## Why not PyTorch / TensorFlow?

| | Python (PyTorch) | 🦀 TensorCrab |
|---|---|---|
| Language | Python + C++ | Pure Rust |
| Memory safety | Manual / GC | Compile-time |
| Deployment | Heavy containers, Python runtime | Single binary |
| Concurrency | GIL limits parallelism | True multi-threading |
| Overhead | Python interpreter + FFI | Zero |

TensorCrab is the right choice when you need ML as part of a larger Rust application — a game engine, a web server, an embedded system — and you can't afford the weight of a Python runtime.

---

## Roadmap

| Stage | Status |
|---|---|
| 🧮 Tensor Engine | 🟢 Done |
| 🔁 Autograd | 🟢 Done |
| 🧠 NN Layers | 🟢 Done |
| 🚀 Optimizers | 🟢 Done |
| 🌐 WebAssembly | 🔴 Planned |
| ⚡ CUDA / GPU | 🟢 Done |
| 🌍 Ecosystem | 🔴 Planned |

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).
