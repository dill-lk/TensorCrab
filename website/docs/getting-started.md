---
id: getting-started
title: Getting Started
sidebar_label: Getting Started
---

# Getting Started

This guide walks you through adding TensorCrab to a Rust project and running your first tensor operations, gradient computation, and mini training loop.

---

## Prerequisites

- Rust 1.70+ (edition 2021)
- `cargo` in your PATH

Install Rust via [rustup](https://rustup.rs) if you haven't already:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

## Installation

TensorCrab is not yet published to crates.io. Add it as a Git dependency:

```toml title="Cargo.toml"
[dependencies]
tensor-crab = { git = "https://github.com/dill-lk/TensorCrab" }
```

---

## Your First Tensor

```rust
use tensor_crab::tensor::Tensor;

fn main() {
    // Create a 2×2 matrix
    let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);

    // Matrix multiplication
    let c = a.matmul(&b).unwrap();
    println!("{c}"); // [[19, 22], [43, 50]]

    // Element-wise operations
    let sum = a.add(&b).unwrap();
    println!("{sum}"); // [[6, 8], [10, 12]]

    // Reduction
    println!("{}", a.sum()); // 10.0
}
```

---

## Automatic Differentiation

TensorCrab automatically builds a computation graph as you compute. Call `backward()` once to compute all gradients:

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};

fn main() {
    // x requires a gradient
    let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);

    // z = sum(x²)
    let z = x.var_mul(&x).var_sum();

    // Compute dz/dx
    backward(&z);

    let grad = x.grad().unwrap();
    println!("{:?}", grad.to_vec()); // [4.0, 6.0]  (= 2x)
}
```

---

## Build a Neural Network

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::optim::{Optimizer, Adam};

fn main() {
    // 3-layer MLP: 2 → 16 → 1
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 16)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(16, 1)),
    ]);

    let mut opt = Adam::new(model.parameters(), 0.001);

    // Single training step
    let x      = Variable::new(Tensor::from_vec(vec![0.0_f32, 1.0], &[1, 2]), false);
    let target = Variable::new(Tensor::from_vec(vec![1.0_f32],       &[1, 1]), false);

    let pred = model.forward(&x);
    let l    = loss::mse_loss(&pred, &target);

    backward(&l);   // compute all gradients
    opt.step();     // update weights
    opt.zero_grad(); // reset gradients for next step

    println!("Training step complete");
}
```

---

## Full Training Loop

Here is a complete example using the `DataLoader` and a learning-rate scheduler:

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};
use tensor_crab::optim::{DataLoader, Optimizer, Adam};
use tensor_crab::optim::scheduler::CosineAnnealingLR;

fn main() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 32)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(32, 1)),
    ]);

    let mut opt = Adam::new(model.parameters(), 0.01);
    let scheduler = CosineAnnealingLR::new(0.01, 1e-5, 50);

    // Synthetic dataset: 200 samples, 2 features
    let x = Tensor::randn_seeded(&[200, 2], 42);
    let y = Tensor::zeros(&[200, 1]);
    let loader = DataLoader::new(x, y, 32, true);

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

        if epoch % 10 == 0 {
            println!("Epoch {epoch} done");
        }
    }

    // Save weights
    model.save_weights("model.bin").unwrap();
    println!("Training complete. Weights saved to model.bin");
}
```

---

## Using the Prelude

For convenience, import the most common types at once:

```rust
use tensor_crab::prelude::*;
// Gives you: Tensor, Variable, backward, Module, Sequential,
//            Optimizer, Adam, AdamW, SGD, StepLR, DataLoader, TensorError
```

---

## Next Steps

- [Tensor Engine →](./tensor) — shapes, broadcasting, all tensor ops
- [Autograd →](./autograd) — how the computation graph works
- [Neural Networks →](./nn) — layers, loss functions, save/load
- [Optimizers →](./optim) — SGD, Adam, AdamW, schedulers
- [DataLoader →](./dataloader) — batching and shuffling datasets
