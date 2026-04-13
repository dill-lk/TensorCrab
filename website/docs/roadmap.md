---
id: roadmap
title: Roadmap
sidebar_label: Roadmap
---

# TensorCrab Roadmap 🦀

## Current Status

| Stage | Status |
|---|---|
| 🧮 Tensor Engine | 🟢 Complete |
| 🔁 Autograd | 🟢 Complete |
| 🧠 NN Layers | 🟢 Complete |
| 🚀 Optimizers | 🟢 Complete |
| 🌐 WebAssembly | 🔴 Planned |
| ⚡ CUDA / GPU | 🔴 Planned |
| 🌍 Ecosystem | 🔴 Planned |

---

## Stage 1 — Tensor Engine ✅

A working N-dimensional array type with zero-copy views.

- [x] `Tensor<T>` struct with generic dtype (f32, f64)
- [x] Shape and stride system (row-major memory layout)
- [x] Basic ops: `add`, `sub`, `mul`, `div` (element-wise, with broadcasting)
- [x] Matrix ops: `matmul`, `transpose`, `reshape`, `flatten`
- [x] Reduction ops: `sum`, `mean`, `max`, `min`
- [x] Broadcasting support
- [x] Scalar and unary ops: `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `neg`
- [x] `Display` trait for pretty printing
- [x] Unit tests for all ops

---

## Stage 2 — Autograd Engine ✅

Dynamic computation graph with reverse-mode automatic differentiation.

- [x] `Variable` wrapper with `Arc`-based reference counting
- [x] Implicit computation graph (DAG)
- [x] Forward pass: execute ops and record graph
- [x] Backward pass: topological sort + chain rule
- [x] Gradient accumulation and `zero_grad()`
- [x] Supported ops: `add`, `sub`, `mul`, `div`, `matmul`, `neg`, `relu`, `sigmoid`, `tanh`, `log`, `exp`, `sqrt`, `sum`, `sum_keepdim`, `transpose`, scalar `mul`/`add`
- [x] Numerical gradient verification tests

---

## Stage 3 — Neural Network Layers ✅

Composable building blocks for neural networks.

- [x] `Module` trait — `forward()`, `parameters()`, `zero_grad()`
- [x] `Linear` layer (Xavier and Kaiming init)
- [x] Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- [x] `Sequential` container
- [x] Loss functions: `mse_loss`, `bce_loss`, `cross_entropy_loss`
- [x] `BatchNorm1d` layer
- [x] `Dropout` layer (inverted dropout)
- [x] Model save/load (binary serialisation)

---

## Stage 4 — Optimizers ✅

Gradient-based weight updates.

- [x] `Optimizer` trait
- [x] `SGD` with momentum and weight decay
- [x] `Adam` (β₁, β₂, ε, weight decay)
- [x] `AdamW` with decoupled weight decay
- [x] `StepLR` scheduler
- [x] `CosineAnnealingLR` scheduler
- [x] `DataLoader` (batching, shuffling, reproducible seeds)

---

## Stage 5 — WebAssembly 🔴 Planned

Run TensorCrab models in the browser.

- [ ] `wasm32` compilation target support
- [ ] `wasm-bindgen` integration
- [ ] JavaScript/TypeScript API bindings
- [ ] Example: run a trained model in the browser
- [ ] WebGPU acceleration via `wgpu`
- [ ] npm package: `tensor-crab-wasm`

---

## Stage 6 — GPU Acceleration 🔴 Planned

Run operations on NVIDIA GPUs.

- [ ] CUDA backend via `cuBLAS` / `cuDNN` bindings
- [ ] Device abstraction: `Device::CPU` and `Device::CUDA(n)`
- [ ] `.to(device)` method on `Tensor` and `Module`
- [ ] Async kernel execution
- [ ] Memory pooling for GPU tensors
- [ ] Mixed precision (`f16`) support

---

## Stage 7 — Ecosystem 🔴 Planned

Make TensorCrab a complete PyTorch/TensorFlow alternative.

- [ ] `tensor-crab-vision` — Conv2d, pooling, image transforms
- [ ] `tensor-crab-text` — tokenizers, embeddings, positional encoding
- [ ] `tensor-crab-hub` — save/load/share trained models
- [ ] Python bindings via `PyO3`
- [ ] Benchmarks vs PyTorch, TensorFlow, burn, candle
- [ ] Example zoo: MNIST, XOR, linear regression, transformer block

---

## Contributing

See [CONTRIBUTING.md](https://github.com/dill-lk/TensorCrab/blob/main/CONTRIBUTING.md) and the [agent constitution](https://github.com/dill-lk/TensorCrab/blob/main/agents.md) for contribution guidelines.
