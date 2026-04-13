# 🦀 TensorCrab Project Context

TensorCrab is a Rust-native machine learning library designed for high performance with zero Python overhead and no GIL. It aims to be a Rust alternative to PyTorch, prioritizing safety, speed, and composability.

## 🔴 Current Status: Pre-development
The project is currently in the **documentation and planning phase**. The architecture, roadmap, and development plan are established, but no source code has been written yet.

## 🛠 Technology Stack
- **Language:** Rust (Edition 2021)
- **Parallelism:** `rayon` for data-parallel operations.
- **Error Handling:** `thiserror` for typed, user-facing errors.
- **Serialization:** `serde` for model saving/loading.
- **Testing:** Unit tests + numerical gradient checks (using `approx`).
- **Performance:** `criterion` for benchmarking.

## 🏗 Architecture Layers
1. **Layer 0 — Storage:** Raw memory management (`Vec<T>` in `Arc`).
2. **Layer 1 — Tensor:** View into storage with shape, strides, and offset.
3. **Layer 2 — Autograd:** Implicit computation graph (DAG) for automatic differentiation.
4. **Layer 3 — NN Modules:** Composable layers (`Linear`, `ReLU`, `Sequential`).
5. **Layer 4 — Optimizers:** Weight updates (`SGD`, `Adam`).

## ⚖️ Development Conventions (Agent Constitution)
Every contribution **MUST** follow the non-negotiable obligations in `agents.md`:
1. **Write the Code:** Implement features exactly as described in `docs/plan.md`.
2. **Write the Tests:** Unit tests + numerical verification for all mathematical ops.
3. **Update Roadmap:** Check off items in `docs/roadmap.md` and update the status table.
4. **Update README:** Keep the progress table in `README.md` accurate.
5. **Update Plan:** Mark module status in `docs/plan.md`.
6. **Quality Check:** All changes must pass `cargo test`, `cargo clippy`, `cargo fmt`, and `cargo doc`.

## 🚀 Key Commands
| Action | Command |
|---|---|
| **Initialize Project** | `cargo new tensor-crab --lib` |
| **Build** | `cargo build` |
| **Test** | `cargo test` |
| **Lint** | `cargo clippy -- -D warnings` |
| **Format** | `cargo fmt` |
| **Docs** | `cargo doc --no-deps` |

## 📂 Key Documentation
- `agents.md`: The "Project Law" for all contributors (Human and AI).
- `docs/architecture.md`: Detailed internal design of the engine.
- `docs/plan.md`: Living implementation state and file structure.
- `docs/roadmap.md`: Overall project progress and milestones.

## ⚠️ Important Rules
- **No `unwrap()`:** Avoid panics in library code. Use `expect()` with a justification or return a `Result`.
- **Atomic Commits:** Features, tests, and documentation updates must be committed together.
- **Deterministic Tests:** Use fixed seeds for random tensor generation in tests.
- **Doc Comments:** Mandatory on all public functions and structs.
