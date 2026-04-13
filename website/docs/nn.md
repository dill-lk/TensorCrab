---
id: nn
title: Neural Network Layers
sidebar_label: Neural Networks
---

# Neural Network Layers 🧠

TensorCrab provides composable building blocks for neural networks, all built on the `Module` trait. Every layer is `Send + Sync` so models can be safely moved across threads.

---

## The Module Trait

Every layer implements `Module`:

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Arc<Variable>) -> Arc<Variable>;
    fn parameters(&self) -> Vec<Arc<Variable>>;
    fn zero_grad(&self);
}
```

| Method | Description |
|---|---|
| `forward(input)` | Compute the layer's output |
| `parameters()` | Return all learnable parameters |
| `zero_grad()` | Reset all parameter gradients to zero |

---

## Layers

### Linear (Fully Connected)

`Linear` implements `y = xW^T + b`. Weights are initialized with **Xavier uniform** by default; use `new_kaiming` for ReLU networks.

```rust
use tensor_crab::nn::Linear;

// Xavier init (default — good for Sigmoid/Tanh)
let layer = Linear::new(784, 256);

// Kaiming init (better for ReLU networks)
let layer = Linear::new_kaiming(784, 256);

// From pre-computed weight and bias tensors
let layer = Linear::from_weights(weight, bias);
```

### ReLU

Applies `max(0, x)` element-wise.

```rust
use tensor_crab::nn::ReLU;
let relu = ReLU::new();
```

### Sigmoid

Applies `1 / (1 + e^{-x})` element-wise. Output is in (0, 1).

```rust
use tensor_crab::nn::Sigmoid;
let sigmoid = Sigmoid::new();
```

### Tanh

Applies `tanh(x)` element-wise. Output is in (-1, 1).

```rust
use tensor_crab::nn::Tanh;
let tanh = Tanh::new();
```

### Softmax

Computes a numerically stable probability distribution over the last dimension. Useful as the final layer for multi-class classification.

```rust
use tensor_crab::nn::Softmax;
let softmax = Softmax::new();
```

### BatchNorm1d

Batch normalisation with learnable scale (γ) and shift (β) parameters. Applied over feature dimension of a 2D input `[batch, features]`.

```rust
use tensor_crab::nn::BatchNorm1d;
let bn = BatchNorm1d::new(256); // 256 features
```

### Dropout

Inverted dropout. During forward, each neuron is zeroed with probability `p` and the remaining outputs are scaled by `1/(1-p)`.

:::note
Dropout is always active in TensorCrab's current implementation. A `training` mode flag is planned.
:::

```rust
use tensor_crab::nn::Dropout;
let drop = Dropout::new(0.3); // 30% dropout rate
```

---

## Sequential

`Sequential` chains layers in order. `forward` passes the output of each layer as input to the next.

```rust
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout};

let model = Sequential::new(vec![
    Box::new(Linear::new(784, 256)),
    Box::new(BatchNorm1d::new(256)),
    Box::new(ReLU::new()),
    Box::new(Dropout::new(0.3)),
    Box::new(Linear::new(256, 10)),
]);

let output = model.forward(&input); // shape: [batch, 10]
let params = model.parameters();    // all learnable tensors
```

---

## Loss Functions

Import loss functions from `tensor_crab::nn::loss`:

```rust
use tensor_crab::nn::loss;
```

### MSE Loss — Regression

Mean Squared Error: `L = mean((pred - target)²)`

```rust
let l = loss::mse_loss(&pred, &target);
```

### BCE Loss — Binary Classification

Binary Cross-Entropy: `L = -mean(target * log(pred) + (1 - target) * log(1 - pred))`

Expects predictions in (0, 1) — apply `Sigmoid` before calling this.

```rust
let l = loss::bce_loss(&pred, &target);
```

### Cross-Entropy Loss — Multi-Class Classification

`L = -mean(sum(target * log(softmax(pred))))`

Accepts raw logits — applies log-softmax internally.

```rust
// target: one-hot encoded, shape [batch, num_classes]
let l = loss::cross_entropy_loss(&logits, &one_hot_target);
```

---

## Save and Load Weights

`Sequential` supports binary serialisation of all parameters:

```rust
// Save
model.save_weights("checkpoint.bin").unwrap();

// Load — model must have the same architecture
model.load_weights("checkpoint.bin").unwrap();
```

The format is a flat binary file of `f32` values in layer-parameter order.

---

## A Full Network Example

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout, loss};
use tensor_crab::optim::{Optimizer, Adam};

fn main() {
    // Build model
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 64)),
        Box::new(BatchNorm1d::new(64)),
        Box::new(ReLU::new()),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(64, 1)),
    ]);

    let mut opt = Adam::new(model.parameters(), 0.001);

    // Training data
    let x_data = Tensor::randn_seeded(&[32, 2], 0);
    let y_data = Tensor::zeros(&[32, 1]);

    let x = Variable::new(x_data, false);
    let y = Variable::new(y_data, false);

    // Training step
    let pred = model.forward(&x);
    let l    = loss::mse_loss(&pred, &y);
    backward(&l);
    opt.step();
    opt.zero_grad();

    println!("Loss computed successfully");
}
```

See the [API Reference →](./api/nn) for a complete method list.
