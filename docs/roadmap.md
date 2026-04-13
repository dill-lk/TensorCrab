# TensorCrab Roadmap 🦀

> **Agent Rule:** Every time a feature is implemented, the implementing agent (human or AI) **MUST** update this file — check off the completed item, update the status table at the bottom, and commit this file alongside the feature code. A feature is not "done" until this roadmap is updated.

## Overview

```
Stage 1 → Tensor Engine         [Foundation]
Stage 2 → Autograd              [The Heart]
Stage 3 → Neural Network Layers [The Brain]
Stage 4 → Optimizers            [The Trainer]
Stage 5 → WASM                  [The Web]
Stage 6 → CUDA / GPU            [The Beast]
Stage 7 → Ecosystem             [The World]
```

---

## Stage 1 — Tensor Engine
**Goal:** A working N-dimensional array type in Rust.

- [x] `Tensor<T>` struct with generic dtype (f32, f64)
- [x] Shape and stride system (row-major memory layout)
- [x] Basic ops: `add`, `sub`, `mul`, `div` (element-wise)
- [x] Matrix ops: `matmul`, `transpose`, `reshape`, `flatten`
- [x] Reduction ops: `sum`, `mean`, `max`, `min`
- [x] Broadcasting support
- [ ] Indexing and slicing
- [x] `Display` trait for pretty printing
- [x] Unit tests for all ops

### Milestone
```rust
let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);
let c = a.matmul(&b);
println!("{}", c); // [[19, 22], [43, 50]]
```

---

## Stage 2 — Autograd Engine
**Goal:** Automatic differentiation via a dynamic computation graph.

- [x] `Variable` wrapper around `Tensor` that tracks gradients
- [x] Computation graph (DAG) — nodes and edges
- [x] Forward pass — execute ops and record graph
- [x] Backward pass — walk graph in reverse, apply chain rule
- [x] Gradient accumulation
- [x] `zero_grad()` to reset gradients
- [x] Support for: add, sub, mul, div, matmul, neg, relu, sigmoid, log, exp, sum
- [x] Unit tests: verify gradients numerically

### Milestone
```rust
let x = Variable::new(Tensor::from([2.0, 3.0]), requires_grad: true);
let y = x.mul(&x).sum();
y.backward();
println!("{}", x.grad()); // [4.0, 6.0]
```

---

## Stage 3 — Neural Network Layers
**Goal:** Composable building blocks for neural networks.

- [x] `Module` trait — defines `forward()` and `parameters()`
- [x] `Linear` layer (fully connected)
- [x] Activation functions: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- [x] `Sequential` container
- [x] Loss functions: `MSELoss`, `CrossEntropyLoss`, `BCELoss`
- [x] Weight initialization: Xavier, Kaiming
- [x] `BatchNorm1d` layer
- [x] Dropout layer
- [x] Model save/load (serialize to binary)

### Milestone
```rust
let model = Sequential::new(vec![
    Linear::new(784, 128),
    ReLU::new(),
    Linear::new(128, 10),
]);
let output = model.forward(&input);
```

---

## Stage 4 — Optimizers
**Goal:** Train models by updating weights with gradient descent.

- [x] `Optimizer` trait
- [x] `SGD` — stochastic gradient descent with momentum
- [x] `Adam` — adaptive moment estimation
- [x] `AdamW` — Adam with weight decay
- [x] Learning rate schedulers: StepLR, CosineAnnealing
- [x] Training loop utilities
- [x] DataLoader abstraction (batching, shuffling)

---

## Stage 5 — WebAssembly (WASM)
**Goal:** Run TensorCrab models in the browser.

- [ ] `wasm32` compilation target support
- [ ] `wasm-bindgen` integration
- [ ] JavaScript API bindings
- [ ] Example: run a trained model in the browser
- [ ] WebGPU acceleration via `wgpu`
- [ ] npm package: `tensor-crab-wasm`

---

## Stage 6 — GPU Acceleration (CUDA)
**Goal:** Run operations on NVIDIA GPUs.

- [x] `cuda` feature flag and `build.rs` that locates the CUDA Toolkit
- [x] Raw CUDA Driver API FFI bindings (`ffi.rs`) — `cuInit`, `cuMemAlloc`, `cuLaunchKernel`, events, streams, peer access
- [x] Raw CUDA Runtime API FFI bindings — `cudaMalloc`, `cudaMemcpy`, device management
- [x] `CudaDevice` — device enumeration, properties, context management, `is_available()`
- [x] `DeviceProperties` — name, memory, compute capability, multiprocessor count
- [x] `CudaStream` — RAII stream creation, synchronise, `is_done()`
- [x] `CudaEvent` — RAII events, `record`, `elapsed_ms` timing
- [x] `CudaBuffer<T>` — type-safe RAII GPU memory (alloc, free, H2D, D2H, D2D copies)
- [x] `CudaModule` / `CudaFunction` — PTX JIT compilation and kernel function lookup
- [x] `grid_size_1d` helper and `DEFAULT_BLOCK_SIZE` constant
- [x] Embedded PTX kernels for `f32`: `add`, `sub`, `mul`, `div`, `relu`, `neg`, `abs`, `sqrt`, `sq`, `exp`, `log`, `fill`, `add_scalar`, `mul_scalar`
- [x] `CudaTensor` — GPU-backed f32 tensor with `from_cpu` / `to_cpu` round-trips
- [x] `CudaTensor` ops: `add`, `sub`, `mul`, `div`, `add_scalar`, `mul_scalar`, `relu`, `neg`, `abs`, `sqrt`, `square`, `exp`, `log`, `reshape`, `flatten`
- [ ] Device abstraction: `Device::CPU` and `Device::CUDA(n)` on existing `Tensor`
- [ ] `.to(device)` method on Tensor and Variable
- [ ] Async kernel execution (streams wired to `CudaTensor` ops)
- [ ] cuBLAS GEMM for `matmul`
- [ ] Memory pooling for GPU tensors
- [ ] Mixed precision (f16) support

---

## Stage 7 — Ecosystem
**Goal:** Make TensorCrab a complete drop-in alternative to PyTorch/TensorFlow.

- [ ] `tensor-crab-vision` — Conv2d, pooling, image transforms (like torchvision)
- [ ] `tensor-crab-text` — tokenizers, embeddings, positional encoding
- [ ] `tensor-crab-hub` — save/load/share trained models
- [ ] Python bindings via `PyO3` — optional interop for migrating PyTorch users
- [ ] Benchmarks vs PyTorch, TensorFlow, burn, candle
- [ ] Full documentation site
- [ ] Example zoo: MNIST, XOR, linear regression, transformer block

---

## Timeline (Estimated)

| Stage | Target |
|---|---|
| Stage 1 — Tensor Engine | Month 1 |
| Stage 2 — Autograd | Month 2 |
| Stage 3 — NN Layers | Month 3 |
| Stage 4 — Optimizers | Month 4 |
| Stage 5 — WASM | Month 5-6 |
| Stage 6 — CUDA | Month 7-9 |
| Stage 7 — Ecosystem | Month 10-12 |

---

## Current Status

🟢 **Stage 1 complete** — Tensor Engine implemented and tested.
🟢 **Stage 2 complete** — Autograd Engine implemented with numerical gradient verification.
🟢 **Stage 3 complete** — Neural Network Layers implemented.
🟢 **Stage 4 complete** — Optimizers (SGD, Adam, AdamW), schedulers (StepLR, CosineAnnealing), and DataLoader implemented.
🟡 **Stage 6 in progress** — CUDA backend started: `cuda` feature flag, raw FFI bindings (Driver + Runtime API), `CudaDevice`, `CudaStream`, `CudaEvent`, `CudaBuffer<T>`, PTX module/kernel loader, embedded PTX kernels for 14 f32 ops, and `CudaTensor` GPU tensor with element-wise operations. All compiles cleanly (zero clippy warnings) and existing 62 tests pass.

## Completion Tracker

> **Agents:** update this table every time items above get checked off.

| Stage | Status | Last Updated By |
|---|---|---|
| Stage 1 — Tensor Engine | 🟢 Done | Claude |
| Stage 2 — Autograd | 🟢 Done | Claude |
| Stage 3 — NN Layers | 🟢 Done | Claude |
| Stage 4 — Optimizers | 🟢 Done | Claude |
| Stage 5 — WASM | 🔴 Not started | — |
| Stage 6 — CUDA GPU | 🟡 In progress | Claude |
| Stage 7 — Ecosystem | 🔴 Not started | — |
