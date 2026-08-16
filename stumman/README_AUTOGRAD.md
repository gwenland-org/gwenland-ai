# Stummañ M1: the autograd engine

## What this is

Stummañ is GwenLand's pure-Rust training framework, and M1 is its smallest
useful piece: a define-by-run autograd engine. You build tensors, run ops on
them, and the tape records what happened so `backward()` can replay it and hand
you gradients.

What M1 is **not**, yet: there is no optimizer, no LoRA layer, no dataset
loader, no GPU. Ops are 2D only, there is no broadcasting, and the backward
math runs on scalar loops rather than the AVX2 kernels the forward pass uses.
Those are M2 and later. See [STUMMAN_PLAN.md](STUMMAN_PLAN.md) for where this
is going, and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the constraints that are
deliberate rather than missing.

A note on platforms: stumman builds on x86-64 Linux and Windows. It does not
build on Apple Silicon, because it depends on glproc and glproc's SIMD kernels
import `std::arch::x86_64` with no architecture gate. CI keeps a macOS leg
running as experimental so the day that changes, it says so.

A note on names: the crate is mid-convention. New types are born with a
semantic prefix (`VLGradStore`), older ones still carry their original names
(`Tensor`, `Tape`, `GlProc`, `SisdBackend`) and get renamed in one dedicated
commit later. See `gl-agent-skills/stumman-naming/SKILL.md`. This document uses
the names that compile today.

## Quick start

```bash
cd stumman
cargo run --example minimal_autograd
```

Real output, not a sketch:

<!-- Output last verified: commit 9875516 -->

```
Forward
  x @ w        [-0.550, -0.200, -0.100,  1.400]
  + b          [-0.450, -0.100, -0.000,  1.500]
  relu         [ 0.000,  0.000,  0.000,  1.500]
  loss         [ 0.375]
  recorded:    ["Matmul", "Add", "Relu", "Mean"]

Backward
  grad x shape [2, 3]
               [ 0.000,  0.000,  0.000,  0.050,  0.100,  0.150]
  grad w shape [3, 2]
               [ 0.000,  0.000,  0.000,  0.500,  0.000,  0.250]
  grad b shape [2, 2]
               [ 0.000,  0.000,  0.000,  0.250]

Frozen base weight
  out          [ 1.000,  2.000,  3.000,  4.000]
  adapter grad: yes
  base grad:    no   (never tracked, so never computed)

M1 autograd engine: forward recorded, backward replayed.
```

Worth reading those numbers rather than skimming them. ReLU zeroed three of the
four outputs, so only one path survives to the gradient, and that is why most of
`grad w` is zero. If everything came back non-zero, something would be wrong.

## Core types

| Type | Role |
|---|---|
| `Tensor<B>` | Generic tensor over a compute backend |
| `Backend` (trait) | Contract every backend implements |
| `GlProc` | CPU backend, calls glproc's SIMD kernels |
| `SisdBackend` | Plain scalar backend, the oracle for gradient checks |
| `Tape` | Records forward ops, then drives the backward pass |
| `VLGradStore` | Holds gradients after `backward()`, keyed by tensor id |

### `Tensor<B>`

Shape metadata plus reference-counted backend storage. Cloning shares storage;
every op allocates a new tensor rather than mutating in place. The backend is a
type parameter, so `Tensor<GlProc>` and `Tensor<SisdBackend>` are different
types and cannot be mixed by accident.

Tensors are born untracked. `with_grad(tape)` opts one in.

### `Backend`

Allocation, matmul, transpose, elementwise ops, reductions. `GlProc` routes
matmul to `glproc::kernels::matmul`, which dispatches AVX-512, AVX2 or scalar
from a cached CPUID probe. `SisdBackend` is a deliberate duplicate written in
plain loops: an oracle that shared code with the thing it checks could not catch
a bug in the shared part.

### `Tape`

The recorder. Ops append a node; `backward()` walks the nodes in reverse, which
is a valid topological order because define-by-run appends in execution order.

```rust
use std::sync::{Arc, Mutex};
use stumman::{GlProc, Tape, Tensor};

let tape = Arc::new(Mutex::new(Tape::new()));
let x = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?
    .with_grad(tape.clone());
let y = x.matmul(&x)?;

let mut guard = Tape::lock(&tape);
assert_eq!(guard.op_names(), vec!["Matmul"]);
guard.backward()?;
```

All operands of one op must share the same tape. Mixing two tapes returns
`InvalidOp` rather than silently splitting the graph. See KL-002.

### `VLGradStore`

A map from tensor id to `(gradient data, shape)`. It **accumulates**: a tensor
reached along two paths gets the sum, which is what shared weights need. Reach
it through `Tape::grad(id)` or `Tape::grad_store()`.

```rust
guard.backward()?;
match guard.grad(x.id()) {
    Some((data, shape)) => println!("grad {shape:?}: {data:?}"),
    None => println!("no gradient, this tensor is frozen"),
}
guard.zero_grad();
```

Gradients are `Vec<f32>`, not `B::Storage`. That is what keeps `Tape`
non-generic, so one tape can eventually span a mixed-backend graph.

## Forward pass ops

Every op below records a node when at least one operand is tracked. Ops on
untracked tensors compute normally and record nothing.

| Method | Shapes | Notes |
|---|---|---|
| `matmul(&other)` | `[M,K] @ [K,N]` to `[M,N]` | 2D only in M1 |
| `add(&other)` | same shape to same shape | no broadcasting |
| `sub(&other)` | same shape to same shape | |
| `mul(&other)` | same shape to same shape | elementwise, not matmul |
| `mul_scalar(f32)` | any to same | |
| `relu()` | any to same | |
| `transpose()` | `[M,N]` to `[N,M]` | 2D only |
| `sum()` | any to `[1]` | |
| `mean()` | any to `[1]` | |

Two reductions return a raw number instead and record nothing, for when you
want the value rather than a graph node: `sum_scalar()` and `mean_scalar()`.

Shape and data access: `shape()`, `n_elems()`, `ndim()`, `to_vec()`, `item()`
(one-element tensors only). Autograd access: `id()`, `requires_grad()`,
`tape()`, `with_grad(tape)`, `detach()`.

## Backward pass

Lock the tape, call `backward()`, read gradients by tensor id.

```rust
let mut guard = Tape::lock(&tape);
guard.backward()?;
let (grad, shape) = guard.grad(w.id()).expect("w should have a gradient");
guard.zero_grad();
```

**What gets differentiated.** `backward()` seeds the last recorded node's output
with ones, so it computes the gradient of `sum(output)`. End your graph in
`sum()` or `mean()` and that seed is exactly `[1.0]`, the usual dL/dL = 1.

**One backward per store.** Calling `backward()` twice without `zero_grad()`
returns `InvalidOp("call zero_grad() before backward()")`. This is not
pedantry: the seed accumulates onto itself and then propagates, so a second
call produced `[3,3,3,3]` where the first gave `[1,1,1,1]`. Tripled, not
doubled, and silently. Accumulating across mini-batches is a real need and gets
its own entry point in a later wave. See KL-005.

**Frozen operands.** A tensor that never joined a tape gets no gradient, and
that is correct rather than an error. `Tape::grad` returns `None` for it. This
is the LoRA shape: a frozen base weight consumed by a trainable adapter, and it
means backward skips work nobody will read. See KL-003.

**Gradient checking.** `SisdBackend` plus central finite differences covers all
nine ops. There is also one exact test that compares a non-square matmul
gradient against hand-computed row and column sums, which pins the math far
harder than a finite difference can.

## Known limitations in M1

Full write-ups with measurements are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**KL-001, `Backend` is not dyn-compatible.** `Box<dyn Backend>` does not
compile, for two independent reasons verified against rustc: the `Clone`
supertrait implies `Self: Sized`, and every method is an associated function
with no `self` receiver. Dispatch is static and that is deliberate. It only
bites at M4, when runtime backend selection ships, and the plan's
`auto_backend() -> Box<dyn Backend>` sketch will need one of four documented
alternatives.

**KL-002, one tape per op (resolved).** Operands carrying different tapes used
to split the graph silently: the node landed on one tape referencing an input
only the other knew about, and the second tape never learned the op happened.
Now rejected with `InvalidOp`.

**KL-003, unresolvable input ids (resolved).** A node lists every operand,
including untracked ones, so looking an input id up in the tape can return
`None`. That means frozen, no gradient wanted, not an error. Backward skips it.

**KL-004, no 0-d tensors (resolved by decision).** A scalar is a rank-1 tensor
of shape `[1]`. There is no 0-d tensor and there will not be one, because an
empty shape makes `product()` return 1 for a shape carrying no information,
which is a silent-wrong waiting to happen.

**KL-005, one backward per gradient store (resolved).** Covered above. The
guard exists because the failure was silent and produced a plausible number.

**KL-006, captured forward values.** `matmul`, `mul` and `relu` capture their
forward inputs when the node is recorded, because their gradients depend on
operand data. Harmless today since nothing mutates tensor storage in place, and
that is verified rather than assumed. It becomes a real problem the moment M2's
optimizer writes weights in place: a tape holding stale captures would compute
gradients from weights that no longer exist, with no error and a plausible loss
curve. It must be resolved in the same change that introduces in-place updates.

Beyond the numbered entries: backward math runs on the scalar helpers in
`autograd/ops.rs`, never on AVX2, which is the price of keeping the tape
backend-agnostic. And each recorded `matmul` or `mul` holds a copy of its
operands, so tape memory grows with graph size. Neither is measured against a
real training loop yet.

## What comes next

M2 is the optimizer and the first real training run: AdamW and SGD under the
Gwellaer sub-system, a `LoraLayer` wrapping a frozen base with trainable
adapters, safetensors checkpointing, and a loss that actually decreases on a
micro-dataset. That is where KL-006 has to be answered and where tape memory
gets measured for real. See
[STUMMAN_PLAN.md](STUMMAN_PLAN.md), Milestone 2.
