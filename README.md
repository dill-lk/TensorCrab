# 🦀 TensorCrab

> **No Python. No GIL. Just Speed.**

TensorCrab is a Rust-native machine learning library — think PyTorch, but written entirely in Rust with zero Python overhead.

> **Current Status:** 🟢 Stage 3 Complete — Neural Network Layers implemented. [See Roadmap](./docs/roadmap.md)

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};
use tensor_crab::nn::{Module, Sequential, Linear, ReLU, loss};

// Stage 3: Neural Network Layers
let model = Sequential::new(vec![
    Box::new(Linear::new(784, 128)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(128, 10)),
]);
let x = Variable::new(Tensor::randn_seeded(&[32, 784], 0), false);
let output = model.forward(&x);              // [32, 10]

let target = Variable::new(Tensor::zeros(&[32, 10]), false);
let l = loss::mse_loss(&output, &target);
backward(&l);                                // gradients flow to all weights
model.save_weights("model.bin").unwrap();
```

## Why TensorCrab?

| Feature | Python (PyTorch) | JS (TensorFlow.js) | 🦀 TensorCrab |
|---|---|---|---|
| Speed | Medium | Low | **Native** |
| Memory Safety | Manual/GC | GC | **Compile-time** |
| Deployment | Heavy containers | Easy (web) | **Single binary** |
| Concurrency | Limited (GIL) | Single-threaded | **True parallelism** |

## Progress

| Component | Status |
|---|---|
| Tensor Engine | 🟢 Done |
| Autograd | 🟢 Done |
| NN Layers | 🟢 Done |
| Optimizers | 🔴 Not started |
| WASM | 🔴 Not started |
| CUDA | 🔴 Not started |

> **Agent Rule:** Update the progress table above whenever a feature lands.

## Quick Start

```toml
# Not published yet — coming soon
[dependencies]
tensor-crab = "0.1.0"
```

## Documentation

- [What is TensorCrab?](./docs/what-is.md)
- [Roadmap](./docs/roadmap.md)
- [Development Plan](./docs/plan.md)
- [Architecture](./docs/architecture.md)
- [Agents & Workflow](./agents.md)
- [Contributing](./CONTRIBUTING.md)

## License

Apache 2.0
