# 🦀 TensorCrab

> **No Python. No GIL. Just Speed.**

TensorCrab is a Rust-native machine learning library — think PyTorch, but written entirely in Rust with zero Python overhead.

> **Current Status:** 🟢 Stage 2 Complete — Autograd Engine implemented with numerical gradient tests. [See Roadmap](./docs/roadmap.md)

```rust
use std::sync::Arc;
use tensor_crab::tensor::Tensor;
use tensor_crab::autograd::{Variable, backward};

// Stage 2: Automatic Differentiation
let x = Variable::new(Tensor::from_vec(vec![2.0_f32, 3.0], &[2]), true);
let z = x.var_mul(&x).var_sum(); // z = sum(x²)
backward(&z);
println!("{:?}", x.grad().unwrap().to_vec()); // [4.0, 6.0]  (dz/dx = 2x)
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
| NN Layers | 🔴 Not started |
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
