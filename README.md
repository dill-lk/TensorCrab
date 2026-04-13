# 🦀 TensorCrab

> **No Python. No GIL. Just Speed.**

TensorCrab is a Rust-native machine learning library — think PyTorch, but written entirely in Rust with zero Python overhead.

> **Current Status:** 🟢 Stage 1 Complete — Tensor Engine implemented. [See Roadmap](./docs/roadmap.md)

```rust
use tensor_crab::prelude::*;

let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
let c = a.matmul(&b).unwrap();
println!("{c}"); // [[19, 22], [43, 50]]
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
| Autograd | 🔴 Not started |
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
