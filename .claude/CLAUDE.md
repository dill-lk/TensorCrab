# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorCrab is a Rust-native machine learning library — a PyTorch equivalent written entirely in Rust with zero Python overhead. Currently in **pre-development** phase (docs written, no code yet).

## Common Commands

```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run a single test
cargo test test_name

# Lint with clippy (zero warnings required)
cargo clippy -- -D warnings

# Format code
cargo fmt

# Build documentation
cargo doc --no-deps
```

## Architecture

TensorCrab follows a layered architecture:
1. **Storage** — Raw contiguous memory, ref-counted via `Arc<Storage<T>>`
2. **Tensor** — View into Storage with shape, strides, and offset
3. **Autograd** — Computation graph (Variable + BackwardFn)
4. **NN Modules** — `Module` trait for layers (Linear, ReLU, etc.)
5. **Optimizers** — SGD, Adam, AdamW

### Key Design Principles

- **Zero-cost abstractions** — high-level API compiles to machine code
- **No hidden allocations** — memory usage is explicit
- **Copy-on-view** — transpose/slice/reshape share storage, only call `.contiguous()` to copy
- **Send + Sync** — all public types are thread-safe

### Data Flow

```
User Code → nn::Module → Variable → ComputeGraph → backward() → Gradients → Optimizer → update
```

### Layer Dependencies (build order)

```
Storage → Shape → Tensor → Variable → ComputeGraph → Backward → Module → Linear → Activations → Loss → Optimizer
```

## Agent Workflow

This project has strict agent rules (see `agents.md` for full details):

1. Never start implementing without reading `plan.md` and `architecture.md`
2. Every feature must include: code + tests + doc comments
3. After completing a feature, ALWAYS update:
   - `roadmap.md` — check off items
   - `README.md` — update progress table
   - `plan.md` — mark module status
4. All quality checks must pass before committing

### Commit Message Format

```
feat(tensor): implement matmul + transpose ops

- add Tensor::matmul with shape validation
- add Tensor::transpose via stride manipulation (no copy)
- numerical tests for both ops
- roadmap Stage 1 items 3/9 checked off
```

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, progress table |
| `docs/architecture.md` | Deep dive into layer design |
| `docs/plan.md` | Implementation phases and current status |
| `docs/roadmap.md` | Feature tracking (update after each feature) |
| `agents.md` | Agent constitution, workflow rules |

## Current Status

All components are in "not started" state. The repository structure in `docs/plan.md` shows the expected file layout under `crates/tensor-crab/src/`.

## Code Standards

- **Doc comments mandatory** on all public functions
- **Tests mandatory** — every public function needs at least one test
- **No unwrap()** in library code without a comment explaining why it's safe
- **Use typed errors** (thiserror) for user-facing failures
- **Deterministic tests** — use fixed seeds, not random