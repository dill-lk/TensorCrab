# TensorCrab — Agent Constitution 🦀

> This is the law of the project. Every contributor — human or AI — operates under these rules. Read this before touching a single file.

---

## What Is an Agent?

An agent is anything that writes, modifies, or reviews code or docs in this repo. That includes:

- **Jinuk** — human lead, founder, final decision maker
- **Claude** — AI pair programmer, implements features, writes docs
- **CI pipeline** — automated quality enforcement on every commit
- **Future contributors** — anyone who opens a PR

Every agent has the same obligations. No exceptions for AI, no exceptions for the lead.

---

## The Prime Directive

> **A feature does not exist until it is tested, documented, and reflected in the roadmap.**

Writing code is 50% of the job. The other 50% is updating the project state so the next agent knows exactly where things stand.

---

- all agents wanted docs are at /docs

## Agent Obligations — Non-Negotiable

Every agent that implements anything MUST do ALL of the following in the SAME commit:

### 1. Write the Code
- Implement the feature exactly as described in `plan.md`
- If you deviate from the plan, update `plan.md` to match reality
- No hacks, no shortcuts that break future stages

### 2. Write the Tests
- Every public function gets at least one unit test
- Mathematical ops (tensor ops, gradients) get numerical verification tests
- `cargo test` must pass with zero failures before committing
- No committing broken tests with "will fix later" — fix it now

### 3. Update `roadmap.md`
- Check off `[ ]` → `[x]` for every completed item
- Update the Completion Tracker table at the bottom
- Change status: 🔴 → 🟡 (in progress) or 🟢 (done)
- Fill in "Last Updated By" with your name or "Claude"

### 4. Update `README.md`
- Update the Progress table to reflect new status
- If a milestone is reached, add a code example

### 5. Update `plan.md`
- Mark the module's status in the Implementation State table
- If implementation differs from the plan, rewrite that section to match what was actually built

### 6. Run the Full Quality Check
```bash
cargo test        # zero failures
cargo clippy      # zero warnings
cargo fmt         # code formatted
cargo doc         # docs build clean
```
All four must pass. If any fail, fix before committing.

### 7. Commit Message Format
```
feat(tensor): implement matmul + transpose ops

- add Tensor::matmul with shape validation
- add Tensor::transpose via stride manipulation (no copy)
- numerical tests for both ops
- roadmap Stage 1 items 3/9 checked off
```

---

## Agent-Specific Rules

### Jinuk (Human Lead)
- Has final say on all architecture decisions
- Must review and approve any PR that changes public API
- Responsible for setting priorities — which roadmap item gets worked on next
- Can override any decision made by Claude, but must document why in the PR

### Claude (AI Agent)
- Never starts implementing without being told which feature to work on
- Always reads `plan.md` before writing any code for a feature
- Always reads `architecture.md` to understand how layers connect before touching autograd or nn modules
- If a task is ambiguous, asks ONE clarifying question before proceeding — does not guess
- Never removes existing tests
- Never marks a feature complete if `cargo test` fails
- Writes doc comments on every public function, no exceptions
- When implementing, always outputs: code → tests → doc updates in that order

### CI Pipeline
Runs automatically on every push. Blocks merge if any check fails.

| Check | Command | Must Pass |
|---|---|---|
| Tests | `cargo test --all` | ✅ Zero failures |
| Linter | `cargo clippy -- -D warnings` | ✅ Zero warnings |
| Format | `cargo fmt --check` | ✅ Clean |
| Docs | `cargo doc --no-deps` | ✅ Builds clean |
| Coverage | `cargo tarpaulin` | ✅ >80% coverage |

---

## Feature Workflow — Exact Steps

```
┌─────────────────────────────────────────────────────────┐
│  1. Jinuk picks next item from roadmap.md               │
│     → tells Claude which feature to implement           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  2. Claude reads plan.md + architecture.md              │
│     → understands the design before writing anything    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  3. Claude implements the feature                       │
│     → code in the right file, matching plan.md          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  4. Claude writes tests                                 │
│     → unit tests + numerical gradient checks if needed  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  5. Claude runs cargo test + clippy + fmt + doc         │
│     → ALL must pass, fix any failures before next step  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  6. Claude updates roadmap.md + README.md + plan.md     │
│     → checks off items, updates status tables           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  7. Commit: code + doc changes together, never separate │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  8. Jinuk reviews → approves or requests changes        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  9. Merge to main → CI runs final checks                │
└─────────────────────────────────────────────────────────┘
```

---

## What You Are NOT Allowed To Do

| ❌ Forbidden | Why |
|---|---|
| Commit code without updating roadmap | Next agent won't know what's done |
| Mark `[x]` without tests passing | Lying about completeness breaks trust |
| Skip `cargo clippy` | Clippy catches real bugs, not just style |
| Commit "fix tests later" | Tests are not optional |
| Change public API without updating `architecture.md` | Architecture drift causes chaos |
| Write code that depends on a stage not yet completed | Breaks build for everyone |
| Add dependencies to `Cargo.toml` without justification in the PR description | Keeps the binary lean |
| Use `unwrap()` in library code without a comment explaining why it's safe | Panics in prod are unacceptable |

---

## Code Standards

### Doc comments are mandatory on all public items
```rust
/// Performs matrix multiplication of two 2D tensors.
///
/// # Panics
/// Panics if `self.shape[1] != other.shape[0]`.
///
/// # Example
/// ```
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
/// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
/// let c = a.matmul(&b);
/// ```
pub fn matmul(&self, other: &Tensor) -> Tensor { ... }
```

### No `unwrap()` in library code without justification
```rust
// BAD
let val = map.get("key").unwrap();

// GOOD — panics only if our own invariant is violated, which is a bug
let val = map.get("key").expect("key must exist: inserted in ::new()");
```

### Error types, not panics, for user-facing failures
```rust
// BAD — user gets a panic with no context
assert_eq!(self.shape[1], other.shape[0]);

// GOOD — user gets a typed error they can handle
if self.shape[1] != other.shape[0] {
    return Err(TensorError::ShapeMismatch {
        expected: self.shape[1],
        got: other.shape[0],
    });
}
```

### Tests must be deterministic
```rust
// BAD — random seed changes every run
let x = Tensor::randn(&[3, 4]);

// GOOD — fixed seed for reproducible tests
let x = Tensor::randn_seeded(&[3, 4], seed: 42);
```

---

## Status Legend

| Symbol | Meaning |
|---|---|
| 🔴 Not started | No code written, no branch exists |
| 🟡 In progress | Branch exists, actively being worked on |
| 🟢 Done | Merged to main, all tests passing, docs updated |
| ⏸️ Blocked | Cannot proceed until a dependency is completed |
| 🚫 Cancelled | Decided against — reason documented in roadmap |

---

## Architecture Data Flow

Understanding this is mandatory before touching autograd or nn layers:

```
User Code
    │
    ▼
┌──────────────────────┐
│     nn::Module       │  ← defines forward() and parameters()
│  (Linear, ReLU...)   │
└──────────┬───────────┘
           │ forward(input: &Variable) -> Variable
           ▼
┌──────────────────────┐
│      Variable        │  ← Tensor + grad + grad_fn + requires_grad
└──────────┬───────────┘
           │ every op creates a new node and records inputs
           ▼
┌──────────────────────┐
│    ComputeGraph      │  ← DAG of all ops since last zero_grad()
│    (implicit DAG)    │
└──────────┬───────────┘
           │ loss.backward() triggers topological sort + chain rule
           ▼
┌──────────────────────┐
│  Gradient Tensors    │  ← accumulated in variable.grad
└──────────┬───────────┘
           │ optimizer reads .grad from each parameter
           ▼
┌──────────────────────┐
│     Optimizer        │  ← SGD / Adam: updates param.data in place
│  (SGD, Adam, AdamW)  │
└──────────┬───────────┘
           │ zero_grad() clears all .grad for next iteration
           ▼
    Next Training Step
```

**Layer dependencies — build in this order, no skipping:**
```
Storage → Shape → Tensor → Variable → ComputeGraph → Backward → Module → Linear → Activations → Loss → Optimizer
```

---

## Communication

| Channel | Purpose |
|---|---|
| GitHub Issues | Bug reports, specific feature requests |
| GitHub Discussions | Architecture decisions, design questions |
| Pull Requests | Code + doc changes — always together |
| `agents.md` | The law — edit only with Jinuk's approval |
| `plan.md` | Living implementation plan — update freely to match reality |
| `roadmap.md` | Source of truth for what's done and what's next |
