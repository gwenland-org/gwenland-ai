# Stummañ — Known Issues

Tracked limitations in the `stumman` crate.

These are **not TODOs**. Each entry is a deliberate, documented consequence of a
design decision, recorded here so a later wave does not rediscover it as a bug
or "fix" it speculatively. Every entry names the milestone that owns its
resolution.

**Status key** — `KNOWN LIMITATION`: accepted, resolution scheduled ·
`ACCEPTED`: will not change · `OPEN QUESTION`: undecided, blocks a named wave ·
`RESOLVED`: fixed; entry kept so the failure mode stays on the record.

---

## KL-001 — `Backend` is not dyn-compatible

| | |
|---|---|
| **Status** | KNOWN LIMITATION |
| **Introduced** | M1 Wave 1 |
| **Resolution owned by** | **M4** (GPU backend + backend selection CLI) |
| **Affects** | `stumman/src/tensor/backend.rs` — the `Backend` trait |
| **Severity** | Low today; blocks one specific plan sketch at M4 |

### What

The `Backend` trait cannot be used as a trait object. Both of these fail to
compile:

```rust
let b: Box<dyn Backend> = Box::new(GlProc);   // error
fn f(b: &dyn Backend) { /* ... */ }           // error
```

Two blockers, each **verified sufficient on its own** (checked against rustc
1.95.0, not inferred):

1. **`Clone` supertrait.** `Backend: Clone` and `Clone: Sized`, so the trait is
   excluded from dyn-compatibility regardless of its methods. This is the one
   rustc reports:

   ```
   error[E0038]: the trait `stumman::Backend` is not dyn compatible
     = note: the trait is not dyn compatible because it requires `Self: Sized`
   ```

2. **No `self` receiver.** Every method is an associated function
   (`fn matmul(a: &Self::Storage, ...)`, not `fn matmul(&self, ...)`). A vtable
   needs a receiver to dispatch on. Confirmed in isolation on a stripped-down
   trait with no `Clone` supertrait:

   ```
   error[E0038]: ...because associated function `zeros` has no `self` parameter
   ```

   rustc stops at the first blocker, so fixing #1 alone would surface #2.

Note what is **not** a blocker: binding the associated type. Both errors above
were produced with `dyn Backend<Storage = Vec<f32>>` already spelled out. The
associated type is still a reason not to *want* trait objects here — a GPU
backend's `Storage` is a device buffer, not a `Vec<f32>`, so no single erased
type spans the backends — but it is a design objection, not a compile error.

### Why it is this way

Dispatch is static, resolved at compile time. This is what STUMMAN_PLAN.md §3.6
specifies under *Static Backend Selection*: zero runtime overhead, and each
backend chooses its own `Storage` type — the whole reason `Storage` is an
associated type rather than a fixed `Vec<f32>`.

### The conflict

The **same** plan section (§3.6, *GATE Integration*) sketches:

```rust
fn auto_backend() -> Box<dyn Backend> {
    let policy = ExecutionPolicy::auto();
    match policy.best_device() {
        Device::Cuda(dev) => Box::new(GlCuda::new(dev)),
        Device::Cpu       => Box::new(GlProc::new()),
        Device::Tpu(dev)  => Box::new(GlJax::new(dev)),
    }
}
```

**This sketch will not compile against the trait as written.** It is the only
place in the plan that assumes dyn-compatibility.

The plan's other dispatch form, in the same section, works fine and needs no
trait objects:

```rust
match backend {
    "cpu"  => train_model::<GlProc>()?,
    "cuda" => train_model::<GlCuda>()?,
    "tpu"  => train_model::<GlJax>()?,
    _ => bail!("unknown backend: {backend}"),
}
```

### Resolution options (decide at M4, not before)

- **(a) Keep static dispatch.** Ship the `match` dispatcher above. Costs one
  monomorphised copy of the training loop per backend — larger binary, zero
  runtime overhead. Cheapest, and enough for `gwen train --backend cuda`.
- **(b) Object-safe facade.** Add a separate `dyn`-facing trait
  (`&self` methods, `Storage` erased behind an opaque handle) implemented
  in terms of the static one. Keeps this trait untouched; adds a layer.
- **(c) Restructure `Backend`.** Take `&self`, drop `Clone`, box the storage.
  Most invasive; gives up the zero-cost property the plan asked for.
- **(d) Enum wrapper.** rustc's own suggestion on the E0038: define
  `enum AnyBackend { GlProc(GlProc), GlCuda(GlCuda), GlJax(GlJax) }`, implement
  the dispatch on it, and use that where a runtime choice is needed. Closed set
  of backends, no vtable, no trait change — a good fit given the backend list
  is fixed and known at compile time.

### Do not act on this before M4

Waves 2–4 (autograd tape, matmul backward, gradient check) need only static
dispatch. Widening the trait now would be speculative work against a contract
no caller exercises yet.

---

## KL-002 — Ops across two different tapes corrupted the graph silently

| | |
|---|---|
| **Status** | **RESOLVED** (M1 Wave 2, same wave it was found) |
| **Introduced** | M1 Wave 2 |
| **Fixed in** | `Tensor::record_op` — `Arc::ptr_eq` guard |
| **Affects** | `stumman/src/tensor/tensor.rs`, via `matmul` / `add` |
| **Regression test** | `ops_across_two_different_tapes_are_rejected`, `ops_on_the_same_tape_are_accepted` |

### What it did

When two operands carried *different* tapes, the op picked one and discarded
the other with no error:

```rust
let a = Tensor::<GlProc>::zeros(&[2,2])?.with_grad(tape1.clone());
let b = Tensor::<GlProc>::zeros(&[2,2])?.with_grad(tape2.clone());
let c = a.matmul(&b)?;   // no error, no warning
```

Measured before the fix (probed 2026-08-16, not inferred):

```
tape1: nodes=1  tensors=2   ← node landed here
tape2: nodes=0  tensors=1   ← never told the op happened
node inputs recorded: [0, 1]
tape1.get_tensor_meta(b.id) -> None      ← DANGLING input reference
c.tape() is tape1
```

The node on `tape1` listed `b` as an input while `tape1` had no metadata for
it; `tape2` never learned the op happened. The graph was split in half and
neither half knew.

### Root cause

`record_op` resolved the tape with
`self.tape.clone().or_else(|| other.tape.clone())` — first-operand-wins — and
registered only the *output* tensor. Inputs were assumed already registered,
which holds only when both operands belong to the tape that was chosen.

### The fix

`record_op` now takes both operands and resolves the tape itself. When both
carry a tape and `!Arc::ptr_eq(t1, t2)`, it returns
`GlTrainError::InvalidOp("operands must share the same tape")`. There is no
sensible merge — node IDs from two tapes would collide — so mixing is rejected
outright, turning a silent wrong answer into a loud one.

Both call sites (`matmul`, `add`) propagate it with `?`. Neither tape is
mutated on the error path, which the regression test asserts.

The check runs *after* the forward compute, which is deliberate: mismatched
tapes are a programming error that never occurs on a correct path, so the
wasted arithmetic costs nothing real and the tape logic stays in one place.

### Not chosen

- **Thread-local tape** (plan §3.6/Q2) would make the situation
  unrepresentable rather than merely rejected. Still the better long-term
  answer, but it is a much larger change and Q2 is not settled — leaving it to
  whichever wave takes up that question.
- **Merging tapes** would require renumbering node IDs. Not worth it.

---

## KL-003 — Untracked operands leave input IDs that do not resolve

| | |
|---|---|
| **Status** | **RESOLVED** — semantics specified and locked by a test |
| **Introduced** | M1 Wave 2 |
| **Documented in** | `Tensor::record_op` and `autograd::Tape` doc comments |
| **Affects** | `stumman/src/tensor/tensor.rs`, `stumman/src/autograd/tape.rs` |
| **Regression test** | `untracked_operand_records_node_with_partial_inputs` |

### What

When one operand is tracked and the other is a plain tensor, the recorded node
lists both as inputs, and the untracked one is registered nowhere:

```
x tracked, w plain:  x.matmul(&w)
node inputs = [3, 4]
tape.get_tensor_meta(w.id = 4) -> None
```

### The specified semantics

> **A `None` input ID means a frozen/untracked operand: no gradient is computed
> for it, and this is not an error.**

This is the ordinary LoRA shape — a frozen base weight consumed by a trainable
activation — so it is the primary path, not an edge case. The behaviour was
always intended; what was missing was anyone saying so. It is now stated on
both `Tensor::record_op` and `Tape`, and pinned by
`untracked_operand_records_node_with_partial_inputs`, which asserts the
asymmetry directly: the tracked operand resolves, the frozen one does not, and
the op still succeeds.

### Obligation this places on Wave 3

`backward()` **must skip unresolvable input IDs** — treat them as a place to
stop propagating, never as a failure. Erroring on a `None` lookup would break
LoRA training. Together with KL-002 the invariant is now tight: every input ID
on a node either belongs to this tape or belongs to no tape at all, never to a
different one.
