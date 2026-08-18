---
name: gwenland-naming-convention
description: >
  GwenLand's two-character semantic prefix convention for Rust type names (BE
  backend, KL kernel, AG autograd, TP generic, VL value, plus 12 more). Covers
  the full prefix table, which types take a prefix, and which deliberately do
  not. Use whenever a struct, enum, or type alias is introduced, renamed, or
  reviewed in glproc, glcore, glcuda, gljax, glserve, stumman, or any future
  gl* crate. Trigger on any new type name even when the request is about
  something else, because a name is cheapest to fix before it has callers.
  Read stumman-naming alongside this for the stumman crate specifically.
---

# GwenLand Semantic Naming Convention

> **Domain:** naming (repo-wide)
> **Applies to:** every `gl*` crate and `stumman`
> **Status:** **TARGET STATE.** Adoption today is 0 of 224 public types.
> **Last updated:** 2026-08-16

## BEFORE YOU START

- [ ] I know this convention describes where the repo is **going**, not how it reads now. Grepping for `BE`/`KL`/`AG` returns nothing.
- [ ] I am introducing or renaming a type. If I'm only editing a function body, this skill does not apply.
- [ ] If the type lives in `stumman/`, I have also read [`../stumman-naming/SKILL.md`](../stumman-naming/SKILL.md).
- [ ] I am not mass-renaming existing types as a drive-by. Renames land as their own commit, never bundled into a feature change.

## Status: read this before you trust the table

This convention is **prescriptive, not descriptive**. Measured 2026-08-16:

| Crate | `pub struct`/`pub enum` | Using a prefix |
|---|---|---|
| gljax | 90 | 0 |
| glcore | 51 | 0 |
| glproc | 30 | 0 |
| glcuda | 26 | 0 |
| glserve | 20 | 0 |
| stumman | 7 | 0 |
| **Total** | **224** | **0** |

Real names in the tree right now: `Arena`, `BackendKind`, `BlockQ4K`, `DType`,
`Dispatcher`, `ExecutionPolicy`, `GgufFile`, `GllmTokenizer`, `GlprocConfig`.

Two consequences you must not get wrong:

1. **New types follow this convention.** That is the point of the skill.
2. **Existing types are not broken.** Do not "fix" `ExecutionPolicy` into
   `VLExecutionPolicy` because you happened to open the file. Renaming a
   public type in glcore ripples into four backends and the runtime. Migration
   is per-crate, scheduled, and needs JinXSuper's sign-off.

If a task needs both a new prefixed type and an old unprefixed one in the same
file, that mix is expected and correct. It is not a bug to report.

## Format

`<Prefix><Name>` — two uppercase characters, then PascalCase. No underscore
between them.

```rust
MDQwen2_5     // ✅
BECuda        // ✅
TPBuffer<T>   // ✅
BE_Cuda       // ❌ underscore after prefix
BeCuda        // ❌ prefix must be uppercase
BCuda         // ❌ single-character prefix
```

## Prefix Table

| Prefix | Meaning | What goes here | Examples |
|--------|---------|----------------|---------|
| `MD` | Model | Primary ML model or architecture | `MDQwen2_5`, `MDLlama3` |
| `LR` | LoRA | LoRA adapter or variant | `LRQwen2_5Code` |
| `DT` | Data | Dataset or data artifact | `DTTrainingSet`, `DTExportGrades` |
| `VL` | Value | Value type without independent identity | `VLShape`, `VLTensorMeta`, `VLTensorId` |
| `EN` | Enum | Finite set of named variants | `ENDataType`, `ENBackend` |
| `GP` | Graph | Computation or dataflow graph (IR level) | `GPComputeGraph`, `GPInferenceGraph` |
| `CP` | Checkpoint | Persisted training artifact with a format version | `CPLora`, `CPFull`, `CPSharded` |
| `CM` | Compiler | Compiler-level abstractions | `CMCompiler`, `CMModule` |
| `SV` | Service | Service or orchestrator | `SVInference`, `SVModelRegistry` |
| `BE` | Backend | Execution platform implementation | `BECuda`, `BEVulkan`, `BEGlProc`, `BESisd` |
| `KL` | Kernel | Executable compute kernel | `KLMatMul`, `KLAttention` |
| `RT` | Runtime | Execution environment for workloads | `RTInference`, `RTLocal` |
| `RS` | Resource | Managed resource with ownership/lifecycle | `RSDeviceBuffer`, `RSModelWeights` |
| `HW` | Hardware | Physical hardware platform or microarch | `HWTigerLake`, `HWRTX4090` |
| `PL` | Pipeline | Ordered multi-stage workflow | `PLInference`, `PLQuantization` |
| `AB` | Algorithm Block | Reusable ML algorithmic building block | `ABGELU`, `ABRMSNorm`, `ABAttention` |
| `TP` | Template/Generic | Generic abstraction over types or values | `TPBuffer<T>`, `TPArray<T>`, `TPTensor<B>` |
| `AG` | Autograd | Autograd engine components | `AGTape`, `AGNode` |
| `OP` | Optimizer | Update rule carrying mutable state across steps | `OPAdamW`, `OPLion`, `OPAdafactor` |

### Picking between the ones that overlap

Four pairs cause almost every wrong pick:

- **`AB` vs `KL`** — role, not identity. `ABGELU` is GELU as an algorithm you
  could implement anywhere. `KLGELU` is a specific compiled kernel you can
  launch. The same math appears twice under two prefixes, and that is correct.
- **`VL` vs a domain prefix** — `VL` is the fallback for plain data with no
  home. If the type only makes sense inside one sub-system, use that
  sub-system's prefix instead (see `AGNode` in stumman-naming).
- **`RS` vs `TP`** — `RS` owns something and has a `Drop` story. `TP` is a
  shape over a type parameter. `TPBuffer<T>` is generic; `RSDeviceBuffer` owns
  VRAM.
- **`EN` vs everything** — do **not** reach for `EN` just because the type is
  an `enum`. `ENBackend` is a list of backend names; `BECuda` is a backend.
  Use `EN` when "a closed set of variants" is the type's whole job.

## Rules

1. **Types only, never bindings.** The prefix is on the type. Locals, fields,
   and parameters use plain snake_case. A prefixed variable name is redundant
   with the type it was just annotated with.

2. **Traits get no prefix.** `trait Backend`, not `trait BEBackend` and never a
   Hungarian `ITBackend`. The `trait` keyword already said it. This also keeps
   the trait name free for its implementors to qualify: `BEGlProc: Backend`
   reads correctly, `BEGlProc: BEBackend` does not.

3. **One prefix, one meaning, forever.** `AG` is Autograd everywhere. Not
   "Aggregate", not "Agent". If you need a second meaning, you need a second
   prefix.

4. **Not every type gets a prefix.** Prefix a type when the category tells a
   reader something they could not infer. Error types, `Result` aliases, config
   structs local to one module, and test fixtures stay plain. Forcing a prefix
   onto a type that fits no category is worse than no prefix.

5. **The prefix names the architectural role, not the implementation.** If a
   type moves layers, its prefix moves with it. Do not encode "this one uses
   AVX2" or "this one is fast" into the name.

6. **Never rename across a feature commit.** A rename touches every call site,
   so it buries the actual diff. Separate commit, separate review.

## Where each prefix sits in the stack

Read this as "which layer am I writing in", not as a dependency graph.

| Layer | Prefix | Example |
|---|---|---|
| Model / architecture | `MD` | `MDQwen2_5` |
| Algorithm blocks it decomposes into | `AB` | `ABRMSNorm`, `ABRoPE`, `ABAttention`, `ABSwiGLU` |
| IR the blocks lower to | `GP` | `GPComputeGraph` |
| Thing that lowers it | `CM` | `CMCompiler` |
| Launchable compute | `KL` | `KLMatMul` |
| Platform that runs it | `BE` | `BECuda`, `BEGlProc` |
| Execution environment | `RT` | `RTInference` |
| Memory it owns | `RS` | `RSDeviceBuffer` |
| Silicon underneath | `HW` | `HWTigerLake` |

Training inserts one layer: the model records into `AGTape`, which holds
`TPTensor<B>`, which dispatches to a `BE*`. See stumman-naming.

## ✅ Correct Pattern

```rust
/// Records every op in the forward pass so Wave 3 can walk it backwards.
pub struct AGTape { /* ... */ }

pub struct BEGlProc;
impl Backend for BEGlProc { /* ... */ }

// Bindings stay plain. The type annotation already carries the prefix.
let tape = AGTape::new();
let backend = BEGlProc::default();
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ prefix leaking onto a binding
let ag_tape = AGTape::new();

// ❌ Hungarian trait prefix
trait ITOptimizer { /* ... */ }

// ❌ implementation detail baked into the name
struct BEGlProcAvx2Fast;

// ❌ prefix forced onto a type with no category
struct VLGlTrainError { /* ... */ }   // just call it GlTrainError

// ❌ drive-by rename of a committed public type inside a feature commit
- pub struct ExecutionPolicy { /* ... */ }
+ pub struct VLExecutionPolicy { /* ... */ }
```

## Adding a New Prefix

Adding one is a repo-wide decision, so it needs JinXSuper's sign-off before
any code uses it. Order matters:

1. Show the category is genuinely distinct, not a sub-type of an existing
   entry. "Optimizers are a kind of algorithm block" would kill a proposed
   `OP`, so make the case explicitly.
2. Pick two characters that collide with nothing in the table above.
3. Update this table **first**, then any crate-level naming skill that
   inherits from it.
4. Only then write the type.

Two characters, always. No single-character prefixes, no three-character ones.

### Decided 2026-08-17 (stumman M2)

Two table changes landed together, both signed off by JinXSuper:

- **`OP` (Optimizer) is approved.** The category survives the "optimizers are a
  kind of algorithm block" objection in step 1: an `AB` is *pure* — it maps
  inputs to outputs — while an optimizer carries **mutable state across training
  steps** (`OPAdamW` holds first and second moments between calls). That is a
  different lifetime, not a different flavour of the same thing.
- **`CP` now means Checkpoint, and Compiler moved to `CM`.** `CP` reads as
  "CheckPoint" to anyone who has not memorised the table, which is the whole
  point of a mnemonic prefix. The reassignment cost nothing: measured on
  2026-08-17, **zero types in the repo used `CP`** — no `CPCompiler`, no
  `CPModule`, and no compiler crate exists to hold them. Rule 3 ("one prefix,
  one meaning, forever") starts from this entry, not the unused reservation.

## GwenLand-Specific Notes

- **A prefix is not a substitute for a doc comment.** `KLMatMul` tells a reader
  the layer, not the contract. Layer plus one line of prose, always.
- **Doc comments near prefixed types stay casual and short.** No em dashes in
  Rust doc comments; use a period or a colon. Write "AGTape records every op
  during the forward pass so Wave 3 can replay it backwards", not "The AGTape
  component serves as the primary mechanism for recording computational
  operations during the forward propagation phase of training."
- **Precedence.** Per [`../README.md`](../../../gl-agent-skills/README.md): measured production
  numbers beat `architecture/` specs, which beat these skills. If an
  `architecture/` doc names types differently and that doc is being
  implemented, the doc wins and this skill gets fixed in the same PR.

## Related Skills

- [../stumman-naming/SKILL.md](../stumman-naming/SKILL.md) — stumman's type map and rename target
- [../rust-skills/trait-design.md](../../../gl-agent-skills/rust-skills/trait-design.md) — why traits stay unprefixed and object-safe
- [../architecture-skills/backend-independence.md](../../../gl-agent-skills/architecture-skills/backend-independence.md) — what a `BE*` type may expose
- [../before-coding/branch-strategy.md](../../../gl-agent-skills/before-coding/branch-strategy.md) — landing a rename as its own commit
