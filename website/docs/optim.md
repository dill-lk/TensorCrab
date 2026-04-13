---
id: optim
title: Optimizers & Schedulers
sidebar_label: Optimizers
---

# Optimizers & Schedulers 🚀

Optimizers update model parameters after each `backward()` call using the accumulated gradients. Learning-rate schedulers adjust the learning rate over training.

---

## The Optimizer Trait

All optimizers implement:

```rust
pub trait Optimizer {
    fn step(&mut self);     // apply gradient update to all parameters
    fn zero_grad(&mut self); // reset all gradients to zero
}
```

---

## SGD — Stochastic Gradient Descent

Classic SGD with optional **momentum** and **weight decay**.

### Construction

```rust
use tensor_crab::optim::SGD;

// Basic SGD
let mut opt = SGD::new(model.parameters(), 0.01);

// With momentum (common: 0.9)
let mut opt = SGD::new(model.parameters(), 0.01)
    .with_momentum(0.9);

// With weight decay (L2 regularisation)
let mut opt = SGD::new(model.parameters(), 0.01)
    .with_weight_decay(1e-4);

// Both
let mut opt = SGD::new(model.parameters(), 0.01)
    .with_momentum(0.9)
    .with_weight_decay(1e-4);
```

### Update rule

Without momentum:
```
param = param - lr * grad
```

With momentum (`β`):
```
v    = β * v_prev + grad
param = param - lr * v
```

### When to use

- Simple tasks and quick experiments
- When you need a well-understood, interpretable optimizer
- CNNs on vision tasks (often beats Adam with the right lr + momentum)

---

## Adam — Adaptive Moment Estimation

Adam maintains per-parameter first (mean) and second (variance) moment estimates, giving it adaptive per-parameter learning rates.

### Construction

```rust
use tensor_crab::optim::Adam;

// Standard Adam (sensible defaults)
let mut opt = Adam::new(model.parameters(), 0.001);

// Custom hyperparameters
let mut opt = Adam::new(model.parameters(), 0.001)
    .with_beta1(0.9)          // first moment decay (default: 0.9)
    .with_beta2(0.999)        // second moment decay (default: 0.999)
    .with_eps(1e-8)           // numerical stability (default: 1e-8)
    .with_weight_decay(1e-4); // optional L2 penalty
```

### Update rule

```
m = β₁ * m_prev + (1 - β₁) * grad
v = β₂ * v_prev + (1 - β₂) * grad²

m̂ = m / (1 - β₁^t)   // bias correction
v̂ = v / (1 - β₂^t)

param = param - lr * m̂ / (sqrt(v̂) + ε)
```

### When to use

- Most deep learning tasks — a reliable default
- When you want fast convergence without hand-tuning lr

---

## AdamW — Adam with Decoupled Weight Decay

AdamW fixes Adam's weight-decay implementation by applying weight decay directly to the parameters rather than incorporating it into the gradient, which leads to better regularization.

### Construction

```rust
use tensor_crab::optim::AdamW;

let mut opt = AdamW::new(model.parameters(), 0.001);

// With explicit weight decay (AdamW is most useful when this is set)
let mut opt = AdamW::new(model.parameters(), 0.001)
    .with_weight_decay(0.01)
    .with_beta1(0.9)
    .with_beta2(0.999)
    .with_eps(1e-8);
```

### Update rule

Same as Adam, but weight decay is applied directly:
```
param = param - lr * (m̂ / (sqrt(v̂) + ε) + weight_decay * param)
```

### When to use

- Transformer-style models
- Any setting where you want strong, consistent regularization

---

## Updating the Learning Rate

All optimizers expose a public `lr` field. You can change it between steps or use a scheduler:

```rust
opt.lr = 0.0001; // manual update

// Or assign from a scheduler:
opt.lr = scheduler.get_lr(epoch);
```

---

## Learning Rate Schedulers

Schedulers are stateless value computers. Call `get_lr(epoch)` each epoch and assign the result to `opt.lr`.

### StepLR

Multiplies the learning rate by `gamma` every `step_size` epochs.

```rust
use tensor_crab::optim::scheduler::StepLR;

// Halve lr every 10 epochs, starting from 0.1
let scheduler = StepLR::new(0.1, 10, 0.5);

for epoch in 0..100 {
    opt.lr = scheduler.get_lr(epoch);
    // Epoch  0–9:  lr = 0.1
    // Epoch 10–19: lr = 0.05
    // Epoch 20–29: lr = 0.025
    // ...
}
```

### CosineAnnealingLR

Smoothly decays the learning rate from `lr_max` to `lr_min` following a cosine curve over `t_max` epochs.

```rust
use tensor_crab::optim::scheduler::CosineAnnealingLR;

// Decay from 0.1 to 0.0 over 50 epochs
let scheduler = CosineAnnealingLR::new(0.1, 0.0, 50);

for epoch in 0..50 {
    opt.lr = scheduler.get_lr(epoch);
    // Starts near 0.1, decays smoothly to 0.0
}
```

**Formula:**
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
```

### When to use each

| Scheduler | Best for |
|---|---|
| `StepLR` | Training where you have a rough schedule in mind |
| `CosineAnnealingLR` | Long training runs; gives a warm-finish to convergence |

---

## Training Loop Pattern

```rust
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, loss};
use tensor_crab::optim::{Optimizer, Adam};
use tensor_crab::optim::scheduler::CosineAnnealingLR;

let mut opt = Adam::new(model.parameters(), 0.01);
let scheduler = CosineAnnealingLR::new(0.01, 1e-5, 100);

for epoch in 0..100 {
    opt.lr = scheduler.get_lr(epoch);

    // --- forward pass ---
    let pred = model.forward(&x);
    let l    = loss::mse_loss(&pred, &y);

    // --- backward pass ---
    backward(&l);

    // --- parameter update ---
    opt.step();
    opt.zero_grad();
}
```

See the [API Reference →](./api/optim) for full constructor and method signatures.
