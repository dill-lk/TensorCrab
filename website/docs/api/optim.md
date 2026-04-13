---
id: optim
title: Optimizer API Reference
sidebar_label: Optimizers
---

# Optimizer API Reference

`tensor_crab::optim`

---

## Optimizer Trait

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
```

| Method | Description |
|---|---|
| `step()` | Apply one gradient update to all parameters |
| `zero_grad()` | Reset all parameter gradients to zero |

All optimizers also expose a public `lr: f32` field that can be updated directly.

---

## SGD

`tensor_crab::optim::SGD`

Stochastic Gradient Descent with optional momentum and weight decay.

### Constructors / builders

| Method | Description |
|---|---|
| `SGD::new(params, lr: f32)` | Basic SGD |
| `.with_momentum(f32)` | Enable momentum (typical: `0.9`) |
| `.with_weight_decay(f32)` | L2 weight decay |

### Update rule

```
// Without momentum:
param -= lr * grad

// With momentum β:
v = β * v_prev + grad
param -= lr * v
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | `f32` | — | Learning rate (mutable) |
| `momentum` | `f32` | `0.0` | Momentum coefficient |
| `weight_decay` | `f32` | `0.0` | L2 penalty |

---

## Adam

`tensor_crab::optim::Adam`

Adaptive Moment Estimation.

### Constructors / builders

| Method | Description |
|---|---|
| `Adam::new(params, lr: f32)` | Standard Adam with defaults |
| `.with_beta1(f32)` | First moment decay (default: `0.9`) |
| `.with_beta2(f32)` | Second moment decay (default: `0.999`) |
| `.with_eps(f32)` | Numerical stability (default: `1e-8`) |
| `.with_weight_decay(f32)` | L2 weight decay |

### Update rule

```
m = β₁ * m_prev + (1 - β₁) * grad
v = β₂ * v_prev + (1 - β₂) * grad²

m̂ = m / (1 - β₁^t)
v̂ = v / (1 - β₂^t)

param -= lr * m̂ / (sqrt(v̂) + ε)
```

---

## AdamW

`tensor_crab::optim::AdamW`

Adam with decoupled weight decay. Weight decay is applied directly to parameters rather than through the gradient.

### Constructors / builders

| Method | Description |
|---|---|
| `AdamW::new(params, lr: f32)` | AdamW with defaults |
| `.with_beta1(f32)` | First moment decay (default: `0.9`) |
| `.with_beta2(f32)` | Second moment decay (default: `0.999`) |
| `.with_eps(f32)` | Numerical stability (default: `1e-8`) |
| `.with_weight_decay(f32)` | Weight decay coefficient (default: `0.01`) |

### Update rule

Same as Adam, but weight decay is decoupled:
```
param -= lr * (m̂ / (sqrt(v̂) + ε) + weight_decay * param)
```

---

## Learning Rate Schedulers

`tensor_crab::optim::scheduler`

Schedulers are stateless value calculators. Call `get_lr(epoch)` each epoch and assign to `opt.lr`.

### StepLR

| Constructor | Description |
|---|---|
| `StepLR::new(base_lr: f32, step_size: usize, gamma: f32)` | Multiply lr by `gamma` every `step_size` epochs |
| `get_lr(epoch: usize) -> f32` | Return the learning rate for this epoch |

**Formula:** `lr = base_lr * gamma^floor(epoch / step_size)`

### CosineAnnealingLR

| Constructor | Description |
|---|---|
| `CosineAnnealingLR::new(lr_max: f32, lr_min: f32, t_max: usize)` | Cosine decay from `lr_max` to `lr_min` over `t_max` epochs |
| `get_lr(epoch: usize) -> f32` | Return the learning rate for this epoch |

**Formula:** `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / T_max))`

---

## DataLoader

`tensor_crab::optim::DataLoader`

Batches and optionally shuffles a dataset each epoch.

| Method | Description |
|---|---|
| `DataLoader::new(inputs, targets, batch_size, shuffle)` | Construct from tensors |
| `n_samples() -> usize` | Total number of data points |
| `n_batches() -> usize` | Number of batches per epoch (`ceil(n / batch_size)`) |
| `iter_epoch(seed: Option<u64>) -> DataLoaderIter` | Returns an iterator over `(x_batch, y_batch)` tuples for one epoch |

The last batch may have fewer than `batch_size` samples if the dataset is not exactly divisible.
