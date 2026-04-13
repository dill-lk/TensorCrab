# What is TensorCrab? 🦀

## One Line

TensorCrab is PyTorch and TensorFlow — recreated from scratch in pure Rust.

---

## The Problem With Python ML

Every major ML library today (PyTorch, TensorFlow, JAX) is built on Python. Python is great for prototyping, but it has real limitations:

- **Global Interpreter Lock (GIL)** — true multi-threading is impossible
- **Runtime overhead** — interpreted language, every op has hidden cost
- **Garbage collection** — unpredictable memory pauses
- **Deployment nightmare** — shipping a model means shipping Python, CUDA, PyTorch, and hundreds of dependencies
- **Binary size** — a basic PyTorch install is 800MB+

## What TensorCrab Does

TensorCrab reimplements the core of PyTorch/TensorFlow in Rust:

| PyTorch concept | TensorCrab equivalent |
|---|---|
| `torch.Tensor` | `Tensor<f32>` |
| `torch.autograd` | `tensor_crab::autograd` |
| `torch.nn.Module` | `tensor_crab::nn::Module` trait |
| `torch.nn.Linear` | `tensor_crab::nn::Linear` |
| `torch.optim.Adam` | `tensor_crab::optim::Adam` |
| `torch.utils.data.DataLoader` | `tensor_crab::data::DataLoader` |

Same ideas. Same concepts. Rust speed and safety.

## What You Get

### Memory Safety Without GC
Rust's borrow checker catches memory errors at compile time. No leaks. No crashes. No garbage collection pauses.

### True Parallelism
No GIL means real multi-threading across all CPU cores — no workarounds needed.

### Native Speed
Rust compiles to machine code. Tensor operations run as fast as hand-written C++.

### Single Binary Deployment
Your entire trained model ships as one self-contained binary. No Python runtime. No pip. No venv. One file.

### WASM Support
Compile your model to WebAssembly and run it in the browser — something Python simply cannot do.

## What It Is Not

- Not a Python wrapper or binding
- Not a research-only toy
- Not trying to replace Python's ecosystem overnight — just the core engine

## Who It Is For

- Rust developers who want to do ML without leaving the ecosystem
- Engineers deploying models in performance-critical or resource-constrained environments  
- Developers targeting edge devices or WebAssembly
- Anyone frustrated by Python's limitations in production AI

## The Name

Ferris 🦀 is the mascot of Rust. Tensors are the fundamental data structure of ML. TensorCrab = ML, the Rust way.
