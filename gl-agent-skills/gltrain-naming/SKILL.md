---
name: gltrain-naming
description: >
  Naming rules for the gltrain crate (Stummañ training framework): the M1
  rename target mapping today's committed names (Tensor, Tape, GlProc,
  SisdBackend, ComputationNode, TensorMeta, TensorId, NodeId) onto their
  prefixed forms (TPTensor, AGTape, BEGlProc, BESisd, AGNode, VLTensorMeta,
  VLTensorId, VLNodeId), the Breton sub-system codenames required in every
  module header (Kevrin, Karg, Kevskrid, Gwellaer), and doc-comment style
  rules. Use when adding, renaming, or reviewing any type, module, or doc
  comment under gltrain/src/, and when writing autograd, tensor, backend, or
  optimizer types. Read alongside gwenland-naming-convention for the full
  prefix table.
---

# Stummañ Naming Convention

> **Domain:** naming (`gltrain/` only)
> **Applies to:** everything under `gltrain/src/`
> **Status:** prefixes are a **rename target**; Breton headers are **live and enforced**
> **Last updated:** 2026-08-16

## BEFORE YOU START

- [ ] I have read [`../gwenland-naming-convention/SKILL.md`](../gwenland-naming-convention/SKILL.md) for the prefix table. This file only covers what is gltrain-specific.
- [ ] I know the prefixed names below **do not exist in the crate yet**. Every type is still on its pre-rename name.
- [ ] If I'm creating a file, I know its module header must carry the right Breton codename.
- [ ] If I'm touching the autograd tape, I have read `gltrain/KNOWN_ISSUES.md` first. KL-002 and KL-003 are the naming-adjacent ones and **both were resolved in 47ab498**, each with a regression test. Read them for the settled semantics, not as open questions. KL-004 through KL-006 landed with Wave 3.

## The rename has not happened

Verified against the working tree on 2026-08-16. Zero of the seven public
types carry a prefix. `gltrain/src/lib.rs:18-21` still re-exports the old
names:

```rust
pub use autograd::{NodeId, Tape, TensorId};
pub use backend::{GlProc, SisdBackend};
pub use error::{GlTrainError, Result};
pub use tensor::{Backend, Tensor};
```

So the table below is a **target**, not a map. Writing `AGTape::new()` today
does not compile. Two rules follow:

- **New types** in gltrain are born prefixed.
- **Existing types** keep their names until the rename lands as one dedicated
  commit. Do not half-migrate: a tree where `Tape` and `AGTape` both appear is
  worse than either end state.

## Rename Target

| Committed name | Where it lives now | Target | Prefix | Why |
|---|---|---|---|---|
| `Tensor<B>` | [tensor/tensor.rs:64](../../gltrain/src/tensor/tensor.rs#L64) | `TPTensor<B>` | `TP` | Generic over backend `B` |
| `Backend` (trait) | [tensor/backend.rs:51](../../gltrain/src/tensor/backend.rs#L51) | `Backend` | none | Traits take no prefix |
| `GlProc` | [backend/glproc.rs:15](../../gltrain/src/backend/glproc.rs#L15) | `BEGlProc` | `BE` | CPU/AVX2 backend |
| `SisdBackend` | [backend/sisd.rs:32](../../gltrain/src/backend/sisd.rs#L32) | `BESisd` | `BE` | Scalar reference backend |
| `Tape` | [autograd/tape.rs:66](../../gltrain/src/autograd/tape.rs#L66) | `AGTape` | `AG` | Autograd engine |
| `ComputationNode` | [autograd/node.rs:47](../../gltrain/src/autograd/node.rs#L47) | `AGNode` | `AG` | One recorded op |
| `TensorMeta` | [autograd/tape.rs:27](../../gltrain/src/autograd/tape.rs#L27) | `VLTensorMeta` | `VL` | Shape + grad flag, no storage |
| `TensorId` | [autograd/node.rs:15](../../gltrain/src/autograd/node.rs#L15) | `VLTensorId` | `VL` | Alias over `usize` |
| `NodeId` | [autograd/node.rs:18](../../gltrain/src/autograd/node.rs#L18) | `VLNodeId` | `VL` | Alias over `usize` |
| `BackwardFn` | [autograd/node.rs:33](../../gltrain/src/autograd/node.rs#L33) | `BackwardFn` | none | Fn alias, not a data type |
| `GlTrainError` | [error.rs:10](../../gltrain/src/error.rs#L10) | `GlTrainError` | none | Role is obvious |
| `Result<T>` | [error.rs:38](../../gltrain/src/error.rs#L38) | `Result<T>` | none | Mirrors `std`, must stay familiar |

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
| Op library | Oberour | `autograd/ops.rs` | `//! Stummañ Oberour: pure math helpers for the backward pass.` |
| Optimizer | Gwellaer | `optim/` | `//! Stummañ Gwellaer: AdamW optimizer.` |
| Model DSL | Gwiskadur | `nn/` | `//! Stummañ Gwiskadur: LoRA adapter.` |
| Checkpoint | Pik | `checkpoint/` | `//! Stummañ Pik: adapter checkpoint.` |

A new file under `gltrain/src/` opens with its sub-system line. `error.rs` and
`lib.rs` sit outside the five sub-systems and carry no codename.

Watch the boundary: `tensor/backend.rs` holds the `Backend` **trait** and is
tagged **Karg**, not Kevrin, even though it sits in the `tensor/` directory.
The codename tracks the sub-system, not the folder.

Same for `autograd/ops.rs`: it sits under `autograd/` but is **Oberour**, not
Kevskrid. Kevskrid is the recorder and replayer, the tape and its nodes;
Oberour is the arithmetic those backward closures call. GLTRAIN_PLAN.md Part 7
assigns Oberour to this path, and by the precedence rule in
[`../README.md`](../README.md) an architecture doc being implemented outranks
this skill, so the plan wins and this table follows it.

Do not confuse the two files named `ops.rs`. `tensor/ops.rs` is Kevrin and is
still an empty placeholder; `autograd/ops.rs` is Oberour and holds
`matmul_f32` and `transpose_2d`.

## Module Structure

```
gltrain/src/
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
        node.rs           -- ComputationNode, TensorId, NodeId, BackwardFn,
                          --   InputGrad
        grad_store.rs     -- VLGradStore       (already prefixed, born Wave 3)
        ops.rs            -- matmul_f32, transpose_2d   (Oberour, not Kevskrid)
```

`VLGradStore` is the first type in the crate to be born with its prefix, which
is the rule for new types. It is not in the rename table below because there is
nothing to rename.

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
`gltrain/KNOWN_ISSUES.md`, and option (d) would introduce an `AnyBackend`
**enum**, not a `BE*` struct. Picking a name now would prejudge a decision
that belongs to M4.

### KL-002 — one tape per op (RESOLVED in 47ab498)

Operands carrying different tapes are rejected with
`InvalidOp("operands must share the same tape")`. Every input ID on a node
therefore belongs either to that node's tape or to no tape at all.

**Naming consequence:** none outstanding. Do not invent a tape-merging or
multi-tape type to route around the restriction; mixing tapes is disallowed by
design, not pending a better name.

### KL-003 — the node input list keeps its shape (RESOLVED in 47ab498)

This entry used to say the input list needed a name and warned against settling
it in passing. It was settled deliberately, in its own commit, with a test.

`ComputationNode::inputs` stays `Vec<TensorId>`. Both candidate shapes that were
under consideration, `Vec<(TensorId, bool)>` and a separate `tracked_inputs`
list, were **rejected**. The distinction they encoded is carried by the tape
instead: an ID that does not resolve via `Tape::get_tensor_meta` is a frozen
operand.

The rule, stated on `Tensor::record_op` and on `Tape`:

> A `None` input ID means a frozen/untracked operand: no gradient is computed
> for it, and this is not an error.

Wave 3 honours it. `BackwardFn` returns `InputGrad = Option<(Vec<f32>,
Vec<usize>)>` per input, and a `None` there is skipped rather than treated as a
failure.

**Naming consequence:** `InputGrad` takes no prefix. It is a type alias over an
`Option` of plain data, in the same family as `BackwardFn`, and a prefix would
buy nothing.

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

## M2 names (approved 2026-08-17)

`OP` **is now in the** [GwenLand prefix table](../gwenland-naming-convention/SKILL.md#prefix-table),
and `CP` was reassigned from Compiler to Checkpoint in the same edit. Both were
signed off by JinXSuper before any M2 code was written, which is the order the
parent skill requires.

| Type | Prefix | Role |
|---|---|---|
| `OPAdamW` | `OP` | AdamW. Carries first/second moments across steps. |
| `OPLion`, `OPAdafactor`, `OPAdamW8bit` | `OP` | Stub optimizers, real interfaces. |
| `CPLora` | `CP` | Adapter-only checkpoint. |
| `CPFull`, `CPSharded`, `CPIncremental` | `CP` | Stub checkpoint layouts. |
| `PLGgufMerge` | `PL` | **Not** a checkpoint: a one-way export pipeline. See below. |
| `LRLora` | `LR` | Canonical LoRA adapter. |
| `LRDora`, `LRQLora`, `LRLoHa`, `LRVeRA`, `LRLoCon` | `LR` | Stub adapters. |
| `ABLinear` | `AB` | Plain linear layer, a reusable building block. |
| `TPParameter<B>` | `TP` | Named trainable tensor, generic over backend. |
| `VLLoraConfig`, `VLAdamWConfig`, `VLParamGroup`, `VLAdapterCapability` | `VL` | Config and metadata bags. |
| `Adapter`, `Optimizer`, `CheckpointStore`, `Exporter`, `Module` | none | Traits take no prefix. |
| `AdapterRegistry`, `OptimizerRegistry`, `CheckpointRegistry` | none | Match `PluginRegistry` in glictus-caliburni, the pattern they copy. |

Two of these encode a research finding rather than a taste:

- **`PLGgufMerge` is `PL`, not `CP`.** Merging an adapter into a GGUF is
  lossy and one way: it requantizes, and nothing can resume from its output. A
  `CP` name would imply a `load()` that cannot exist. `PL` (pipeline) says what
  it is: read adapter, read base, merge, requantize, write.
- **There is no `LRLoraPlus`.** LoRA+ changes only the learning-rate ratio
  between the A and B parameter groups, so it lives on the optimizer as a
  policy, not in `nn/adapter/`. Naming it as an adapter would create a type
  structurally identical to `LRLora`.

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
pub struct LRLoraPlus;          // ❌ LoRA+ is an optimizer policy, not an adapter
pub struct VLGlTrainError;      // ❌ error types take no prefix

assert!((got - want).abs() < 1e-4);   // ❌ bare tolerance literal
```

## Style Rules

**No em dashes in Rust comments or doc strings.** Use a period or a colon.

⚠️ **This rule conflicts with the committed crate.** 48 doc-comment lines under
`gltrain/src/` currently use `—`, including all 8 module headers (`//! Stummañ
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
- [../rust-skills/trait-design.md](../rust-skills/trait-design.md) — object safety, which is what KL-001 is about
- [../rust-skills/testing-standards.md](../rust-skills/testing-standards.md) — where the tolerance constants get used
- [../before-coding/wave-confirmation-gates.md](../before-coding/wave-confirmation-gates.md) — why M4 names wait for M4
