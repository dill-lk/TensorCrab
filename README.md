# 🦀 TensorCrab

> **No Python. No GIL. Just Speed.**

TensorCrab is a Rust-native machine learning library — think PyTorch, but written entirely in Rust with zero Python overhead.

> **Current Status:** 🔴 Pre-development — docs written, no code yet. [See Roadmap](./roadmap.md)

```rust
use tensor_crab::prelude::*;

let x = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
let model = Linear::new(2, 1);
let output = model.forward(&x);
output.backward();
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
| Tensor Engine | 🔴 Not started |
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

- [What is TensorCrab?](./what-is.md)
- [Roadmap](./roadmap.md)
- [Development Plan](./plan.md)
- [Architecture](./architecture.md)
- [Agents & Workflow](./agents.md)
- [Contributing](./CONTRIBUTING.md)

## License

Apache 2.0
