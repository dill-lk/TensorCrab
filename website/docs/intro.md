---
id: intro
title: What is TensorCrab?
sidebar_label: Introduction
slug: /intro
---

# What is TensorCrab? 🦀

TensorCrab is a **Rust-native machine learning library** — the core of PyTorch and TensorFlow, reimplemented from scratch in pure Rust.

## One-Line Summary

> PyTorch + TensorFlow, recreated in Rust. No Python. No GIL. No overhead.

---

## The Problem With Python ML

Every major ML framework today — PyTorch, TensorFlow, JAX — is built on Python. Python is excellent for prototyping but has fundamental limitations in production:

| Problem | Impact |
|---|---|
| **Global Interpreter Lock (GIL)** | True multi-threading is impossible |
| **Runtime overhead** | Every tensor op has a hidden interpreter cost |
| **Garbage collection** | Unpredictable memory pauses |
| **Deployment complexity** | Shipping a model means shipping Python + CUDA + PyTorch + hundreds of deps |
| **Binary size** | A basic PyTorch install is 800 MB+ |

---

## What TensorCrab Provides

TensorCrab reimplements the same core concepts, but in idiomatic Rust:

| PyTorch | TensorCrab |
|---|---|
| `torch.Tensor` | `Tensor<f32>` |
| `torch.autograd` | `tensor_crab::autograd` |
| `torch.nn.Module` | `tensor_crab::nn::Module` trait |
| `torch.nn.Linear` | `tensor_crab::nn::Linear` |
| `torch.optim.Adam` | `tensor_crab::optim::Adam` |
| `torch.utils.data.DataLoader` | `tensor_crab::optim::DataLoader` |

Same ideas. Same mental model. Rust speed and safety.

---

## Key Properties

### 🔒 Memory Safety Without a GC
Rust's ownership and borrow-checker prevent every class of memory error — leaks, use-after-free, double-frees — at **compile time**. No garbage collection pauses.

### ⚡ Native Speed
TensorCrab compiles to machine code. Tensor operations run as fast as hand-written C++. No interpreter. No FFI overhead. No hidden copies.

### 🔄 True Parallelism
No GIL. You can spawn real OS threads, use Rayon for data-parallel compute, or split training across cores without any workarounds.

### 📦 Single Binary Deployment
Your entire trained model ships as one self-contained binary. No Python runtime. No pip. No virtual environments. One file, drop it anywhere.

### 🌐 WebAssembly Ready *(planned)*
Compile your model to WASM and run it in the browser — something Python simply cannot do.

---

## What It Is Not

- ❌ Not a Python wrapper or FFI binding
- ❌ Not a research prototype — designed for production use
- ❌ Not trying to replace Python's ecosystem overnight — it targets the **core engine**

---

## Who It Is For

- **Rust developers** who want ML without leaving the ecosystem
- **Engineers deploying models** in performance-critical or resource-constrained environments
- **Developers targeting edge devices** or WebAssembly environments
- **Anyone frustrated** by Python's limitations in production AI systems

---

## The Name

Ferris 🦀 is the mascot of Rust. Tensors are the fundamental data structure of ML.  
**TensorCrab = ML, the Rust way.**

---

## Ready to Start?

Head to [Getting Started](./getting-started) to add TensorCrab to your project and run your first tensor op.
