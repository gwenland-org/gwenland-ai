---
name: stumman-naming
description: >
  Naming rules for the stumman crate (Stummañ training framework): the M1
  rename target mapping today's committed names (Tensor, Tape, GlProc,
  SisdBackend, ComputationNode, TensorMeta, TensorId, NodeId) onto their
  prefixed forms (TPTensor, AGTape, BEGlProc, BESisd, AGNode, VLTensorMeta,
  VLTensorId, VLNodeId), the Breton sub-system codenames required in every
  module header (Kevrin, Karg, Kevskrid, Gwellaer), and doc-comment style
  rules. Use when adding, renaming, or reviewing any type, module, or doc
  comment under stumman/src/, and when writing autograd, tensor, backend, or
  optimizer types. Read alongside gwenland-naming-convention for the full
  prefix table.
---

# Stummañ Naming Convention

> **Domain:** naming (`stumman/` only)
> **Applies to:** everything under `stumman/src/`
> **Status:** prefixes are a **rename target**; Breton headers are **live and enforced**
> **Last updated:** 2026-08-16

## BEFORE YOU START

- [ ] I have read [`../gwenland-naming-convention/SKILL.md`](../gwenland-naming-convention/SKILL.md) for the prefix table. This file only covers what is stumman-specific.
- [ ] I know the prefixed names below **do not exist in the crate yet**. Every type is still on its pre-rename name.
- [ ] If I'm creating a file, I know its module header must carry the right Breton codename.
- [ ] If I'm touching the autograd tape, I have read `stumman/KNOWN_ISSUES.md` KL-002 and KL-003 first. Both are naming-adjacent and both are unresolved.

## The rename has not happened

Verified against the working tree on 2026-08-16. Zero of the seven public
types carry a prefix. `stumman/src/lib.rs:18-21` still re-exports the old
names:

```rust
pub use autograd::{NodeId, Tape, TensorId};
pub use backend::{GlProc, SisdBackend};
pub use error::{GlTrainError, Result};
pub use tensor::{Backend, Tensor};
```

So the table below is a **target**, not a map. Writing `AGTape::new()` today
does not compile. Two rules follow:

- **New types** in stumman are born prefixed.
- **Existing types** keep their names until the rename lands as one dedicated
  commit. Do not half-migrate: a tree where `Tape` and `AGTape` both appear is
  worse than either end state.

## Rename Target

| Committed name | Where it lives now | Target | Prefix | Why |
|---|---|---|---|---|
| `Tensor<B>` | [tensor/tensor.rs:64](../../../stumman/src/tensor/tensor.rs#L64) | `TPTensor<B>` | `TP` | Generic over backend `B` |
| `Backend` (trait) | [tensor/backend.rs:51](../../../stumman/src/tensor/backend.rs#L51) | `Backend` | none | Traits take no prefix |
| `GlProc` | [backend/glproc.rs:15](../../../stumman/src/backend/glproc.rs#L15) | `BEGlProc` | `BE` | CPU/AVX2 backend |
| `SisdBackend` | [backend/sisd.rs:32](../../../stumman/src/backend/sisd.rs#L32) | `BESisd` | `BE` | Scalar reference backend |
| `Tape` | [autograd/tape.rs:66](../../../stumman/src/autograd/tape.rs#L66) | `AGTape` | `AG` | Autograd engine |
| `ComputationNode` | [autograd/node.rs:47](../../../stumman/src/autograd/node.rs#L47) | `AGNode` | `AG` | One recorded op |
| `TensorMeta` | [autograd/tape.rs:27](../../../stumman/src/autograd/tape.rs#L27) | `VLTensorMeta` | `VL` | Shape + grad flag, no storage |
| `TensorId` | [autograd/node.rs:15](../../../stumman/src/autograd/node.rs#L15) | `VLTensorId` | `VL` | Alias over `usize` |
| `NodeId` | [autograd/node.rs:18](../../../stumman/src/autograd/node.rs#L18) | `VLNodeId` | `VL` | Alias over `usize` |
| `BackwardFn` | [autograd/node.rs:33](../../../stumman/src/autograd/node.rs#L33) | `BackwardFn` | none | Fn alias, not a data type |
| `GlTrainError` | [error.rs:10](../../../stumman/src/error.rs#L10) | `GlTrainError` | none | Role is obvious |
| `Result<T>` | [error.rs:38](../../../stumman/src/error.rs#L38) | `Result<T>` | none | Mirrors `std`, must stay familiar |

Note `TensorMeta` lives in `tape.rs`, not `node.rs`. It is re-exported through
`autograd::mod`, which is what makes it look like a node type.

When the rename lands it must touch `lib.rs:18-21` in the same commit, or the
public API and the internal names disagree.

## Breton sub-system codenames (live, enforced)

Unlike the prefixes, this convention is already in the crate and every file
follows it. Each module belongs to a named sub-system, and the module header
states it:

| Sub-system | Breton | Module | Header form |
|---|---|---|---|
| Tensor | Kevrin | `tensor/` | `//! Stummañ Kevrin — the tensor sub-system.` |
| Backend | Karg | `backend/`, `tensor/backend.rs` | `//! Stummañ Karg — SISD backend (pure scalar reference).` |
| Autograd | Kevskrid | `autograd/` | `//! Stummañ Kevskrid — Autograd tape.` |
| Optimizer | Gwellaer | not written yet (M2) | `//! Stummañ Gwellaer — ...` |

A new file under `stumman/src/` opens with its sub-system line. `error.rs` and
`lib.rs` sit outside the four sub-systems and carry no codename.

Watch the boundary: `tensor/backend.rs` holds the `Backend` **trait** and is
tagged **Karg**, not Kevrin, even though it sits in the `tensor/` directory.
The codename tracks the sub-system, not the folder.

## Module Structure

```
stumman/src/
    lib.rs                -- re-exports, no codename
    error.rs              -- GlTrainError, Result<T>, no codename
    tensor/               -- Kevrin
        mod.rs
        tensor.rs         -- Tensor<B>        -> TPTensor<B>
        backend.rs        -- Backend trait     (Karg, not Kevrin)
        ops.rs            -- placeholder, Wave 3+
    backend/              -- Karg
        mod.rs
        glproc.rs         -- GlProc           -> BEGlProc
        sisd.rs           -- SisdBackend      -> BESisd
    autograd/             -- Kevskrid
        mod.rs
        tape.rs           -- Tape, TensorMeta -> AGTape, VLTensorMeta
        node.rs           -- ComputationNode, TensorId, NodeId, BackwardFn
```

## Rules

1. **Backends are `BE*`.** Anything implementing the `Backend` trait takes the
   prefix. `BECuda` (M4, via glcuda) and `BEVulkan` are reserved for that wave.
   Do not create them early as empty stubs.

2. **`AG` beats `VL` inside `autograd/`.** The repo-wide rule says pure data
   takes `VL`. `AGNode` overrides it: the type is fields-only, but it is
   meaningless outside the tape, so the domain prefix carries more information
   than the shape prefix does. `VLTensorMeta` stays `VL` because a shape and a
   bool are genuinely generic. When in doubt, ask whether the type would still
   make sense if you moved it out of `autograd/`. If no, use `AG`.

3. **`VL` for plain data.** Type aliases, metadata structs, shape descriptors,
   config bags. Derived traits only, no behaviour beyond that.

4. **`TP` for generics over a backend.** `TPTensor<B>` is the only one in Wave
   1-2. Future generic containers such as a grad buffer follow it.

5. **No prefix on `Result` or `GlTrainError`.** Both are read constantly and
   both mirror shapes every Rust reader already knows. A prefix here costs
   familiarity and buys nothing.

6. **No `BEAuto` or `BEDynamic` before M4.** See KL-001 below.

## Known Issues that constrain naming

### KL-001 — `Backend` is not dyn-compatible

`Box<dyn Backend>` does not compile. Two independent blockers, both verified
against rustc 1.95.0 rather than inferred: the `Clone` supertrait implies
`Self: Sized`, and every method is an associated function with no `self`
receiver. The plan's `fn auto_backend() -> Box<dyn Backend>` sketch (§3.6) will
not build as written.

**Naming consequence:** do not invent a `BEAuto`, `BEDynamic`, or `BEAny`
wrapper to route around it. Four resolution options are written up in
`stumman/KNOWN_ISSUES.md`, and option (d) would introduce an `AnyBackend`
**enum**, not a `BE*` struct. Picking a name now would prejudge a decision
that belongs to M4.

### KL-003 — the node input list needs a name, in Wave 3

Untracked operands currently leave dangling IDs on the tape, and
`KNOWN_ISSUES.md` names two candidate shapes: `inputs: Vec<(TensorId, bool)>`
or a separate `tracked_inputs` list. That is a naming decision as much as a
data-structure one, and it is still open. Do not settle it in passing while
doing unrelated Wave 3 work.

## Wave 3+ Naming Preview

These names are **reserved, not approved**. Reserving them stops two waves
inventing different names for the same thing; it does not authorise creating
the types early.

| Type | Prefix | Role |
|------|--------|------|
| Gradient buffer | `VLGradBuffer` | Pure value, holds grad data |
| Backward context | `AGBackwardCtx` | Tape snapshot for replay |
| Grad checker | `AGGradCheck` | Numerical gradient verification |
| Generic grad buffer | `TPGradBuffer<B>` | If it ends up generic over backend |

M2 (optimizer, sub-system Gwellaer) wants a prefix that **does not exist yet**:

| Type | Proposed prefix | Role |
|------|------|------|
| `OPAdamW`, `OPSGD`, `OPLion` | `OP` | Stateful update rule applied to model parameters |

`OP` is not in the GwenLand prefix table. Before M2 writes a line of optimizer
code, `OP` gets proposed through the process in
[`../gwenland-naming-convention/SKILL.md`](../gwenland-naming-convention/SKILL.md#adding-a-new-prefix)
and added to the table. The obvious objection to answer: why is an optimizer
not just an `AB` algorithm block? (Answer to give: it carries mutable state
across steps; an `AB` is pure.)

## ✅ Correct Pattern

```rust
//! Stummañ Kevskrid — Autograd tape.

/// Records ops during the forward pass. Wave 3 replays them backwards.
pub struct AGTape {
    nodes: Vec<AGNode>,
    tensors: HashMap<VLTensorId, VLTensorMeta>,
}

const TOL_ELEM:   f32 = 1e-6;   // elementwise ops, no accumulation
const TOL_MATMUL: f32 = 1e-4;   // matmul, accumulation over K dimension

assert!((got - want).abs() < TOL_MATMUL);
```

## ❌ Anti-Pattern (Never Do This)

```rust
//! Autograd tape.
//                  ❌ missing the Kevskrid codename

/// AGTape records ops during the forward pass — Wave 3 replays them backwards.
//                                             ❌ em dash in a doc comment

pub struct BEAuto;              // ❌ blocked by KL-001, decide at M4
pub struct OPAdamW;             // ❌ OP is not an approved prefix yet
pub struct VLGlTrainError;      // ❌ error types take no prefix

assert!((got - want).abs() < 1e-4);   // ❌ bare tolerance literal
```

## Style Rules

**No em dashes in Rust comments or doc strings.** Use a period or a colon.

⚠️ **This rule conflicts with the committed crate.** 48 doc-comment lines under
`stumman/src/` currently use `—`, including all 8 module headers (`//! Stummañ
Kevskrid — Autograd engine.`). Treat the rule as applying to lines you write
or edit; do not open a PR that rewrites all 48. If the crate should be swept,
that is JinXSuper's call and it rides along with the M1 rename commit.

**Doc comments stay casual and concrete.** One person writes this crate, so it
should not read like a committee wrote it. Say what the type does and why it
exists:

```rust
/// Scalar reference backend. Storage is Vec<f32>, everything is a plain loop.
/// Exists so gradient checks have an independent oracle to compare against.
```

not:

```rust
/// The BESisd struct provides a scalar, non-vectorized reference implementation
/// of the Backend trait, serving as an independent oracle for numerical
/// gradient verification purposes.
```

**Tolerances always get named constants.** Never a bare float literal in a test
assertion. The name records *why* the number is what it is, which a literal
cannot.

## Related Skills

- [../gwenland-naming-convention/SKILL.md](../gwenland-naming-convention/SKILL.md) — the prefix table this file inherits
- [../rust-skills/trait-design.md](../../../gl-agent-skills/rust-skills/trait-design.md) — object safety, which is what KL-001 is about
- [../rust-skills/testing-standards.md](../../../gl-agent-skills/rust-skills/testing-standards.md) — where the tolerance constants get used
- [../before-coding/wave-confirmation-gates.md](../../../gl-agent-skills/before-coding/wave-confirmation-gates.md) — why M4 names wait for M4
