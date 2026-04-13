---
id: nn
title: Neural Network API Reference
sidebar_label: nn Modules
---

# Neural Network API Reference

`tensor_crab::nn`

---

## Module Trait

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
| `parameters()` | Return all learnable tensors (weights and biases) |
| `zero_grad()` | Reset all parameter gradients to zero |

---

## Linear

`tensor_crab::nn::Linear`

Applies `y = x W^T + b`.

| Constructor | Description |
|---|---|
| `Linear::new(in: usize, out: usize)` | Xavier uniform weight init |
| `Linear::new_kaiming(in: usize, out: usize)` | Kaiming (He) init — better for ReLU networks |
| `Linear::from_weights(weight: Tensor, bias: Tensor)` | Construct from pre-computed tensors |

---

## Activation Layers

All are zero-parameter layers that implement `Module`.

| Type | Formula | Constructor |
|---|---|---|
| `ReLU` | `max(0, x)` | `ReLU::new()` |
| `Sigmoid` | `1 / (1 + e^{-x})` | `Sigmoid::new()` |
| `Tanh` | `tanh(x)` | `Tanh::new()` |
| `Softmax` | Numerically stable softmax over last dim | `Softmax::new()` |

---

## BatchNorm1d

`tensor_crab::nn::BatchNorm1d`

Batch normalisation for 2-D inputs `[batch, features]`. Has learnable γ and β parameters.

| Constructor | Description |
|---|---|
| `BatchNorm1d::new(num_features: usize)` | Initialize γ=1, β=0 |

---

## Dropout

`tensor_crab::nn::Dropout`

Inverted dropout — zeroes each element with probability `p` and scales survivors by `1/(1-p)`.

| Constructor | Description |
|---|---|
| `Dropout::new(p: f32)` | Dropout rate in `[0, 1)` |

---

## Sequential

`tensor_crab::nn::Sequential`

Chains an ordered list of `Module`s. `forward` passes each layer's output to the next.

| Method | Description |
|---|---|
| `Sequential::new(layers: Vec<Box<dyn Module>>)` | Build the container |
| `forward(&input)` | Run input through all layers in order |
| `parameters()` | Concatenated parameters from all layers |
| `zero_grad()` | Reset all layer gradients |
| `save_weights(path)` | Serialise all parameters to a binary file |
| `load_weights(path)` | Load parameters from a binary file |

---

## Loss Functions

`tensor_crab::nn::loss`

All loss functions return a scalar `Arc<Variable>` suitable for calling `backward()` on.

### `mse_loss`

```rust
pub fn mse_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable>
```

Mean Squared Error: `L = mean((pred - target)²)`

Use for regression problems.

### `bce_loss`

```rust
pub fn bce_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable>
```

Binary Cross-Entropy: `L = -mean(target * log(pred) + (1 - target) * log(1 - pred))`

Expects `pred` in `(0, 1)`. Apply `Sigmoid` before calling.

### `cross_entropy_loss`

```rust
pub fn cross_entropy_loss(pred: &Arc<Variable>, target: &Arc<Variable>) -> Arc<Variable>
```

Categorical Cross-Entropy with internal log-softmax. `target` should be one-hot encoded with shape `[batch, num_classes]`.
