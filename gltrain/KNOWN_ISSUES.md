# Stummañ — Known Issues

Tracked limitations in the `gltrain` crate.

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
| **Affects** | `gltrain/src/tensor/backend.rs` — the `Backend` trait |
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
   error[E0038]: the trait `gltrain::Backend` is not dyn compatible
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

Dispatch is static, resolved at compile time. This is what GLTRAIN_PLAN.md §3.6
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
| **Affects** | `gltrain/src/tensor/tensor.rs`, via `matmul` / `add` |
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
| **Affects** | `gltrain/src/tensor/tensor.rs`, `gltrain/src/autograd/tape.rs` |
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

---

## KL-004 - Reductions return shape `[1]`, there is no 0-d tensor

| | |
|---|---|
| **Status** | **RESOLVED** by decision (M1 Wave 3) |
| **Raised** | M1 Wave 2 gate, open risk 4 |
| **Affects** | `Tensor::sum`, `Tensor::mean`, `Tensor::item` |
| **Regression test** | `sum_returns_rank_one_tensor_holding_the_total`, `mean_returns_rank_one_tensor_holding_the_average` |

### The problem

Wave 2 flagged it: `sum()` and `mean()` returned a bare `f32`, so a loss value
could not sit on the tape and `loss.backward()` had nothing to attach to. But
`shape_to_n_elems` rejects an empty shape, so a genuine 0-d tensor could not be
built either. `item()` already claimed to serve "0-d or 1-elem tensors" when
only the second was reachable.

### The decision

**A scalar is a rank-1 tensor of shape `[1]`.** Stummañ has no 0-d tensor and
is not getting one.

- `sum()` and `mean()` now return `Tensor<B>` with shape `[1]` and record a
  tape node, so they can end a graph and be differentiated.
- `sum_scalar()` and `mean_scalar()` return the raw `f32` and record nothing,
  for callers that just want the number.
- `item()` keeps working, since a `[1]` tensor has exactly one element.

### Why not 0-d

Allowing an empty shape would make `shape.iter().product()` return 1 for a
shape carrying no information, which is a silent-wrong waiting to happen in any
code that reasons about rank. Shape `[1]` costs nothing, keeps every existing
shape check meaningful, and is what the backward seed already assumes: the
final node's gradient buffer is sized from its shape, and `[1]` makes that seed
exactly `[1.0]`.

### Cost

`sum()` changed its return type, which is a breaking change to a Wave 1 API.
One call site existed, in `sum_of_ones_tensor_equals_n_elems`; it moved to
`sum_scalar()` and kept its name.

---

## KL-005 - `backward()` requires a clean gradient store

| | |
|---|---|
| **Status** | **RESOLVED** by guard (M1 Wave 3) |
| **Introduced** | M1 Wave 3 |
| **Fixed in** | `Tape::backward` precondition check |
| **Regression test** | `backward_twice_without_zero_grad_is_rejected` |

### What it did

Calling `backward()` twice in a row did not double the gradient, it **tripled**
it. Measured on `x @ w` with `x` tracked:

```
after 1st backward: [1.0, 1.0, 1.0, 1.0]
after 2nd backward: [3.0, 3.0, 3.0, 3.0]
```

The seed is accumulated into the store like any other gradient, so the second
pass starts from a seed of 2.0 and adds its propagation on top of the 1.0
already there. Nothing warned.

### The guard

`backward()` now returns
`GlTrainError::InvalidOp("call zero_grad() before backward()")` when the
gradient store is non-empty. The rejected call mutates nothing, and after
`zero_grad()` the same pass reproduces the original gradient exactly. Both are
asserted by the regression test.

### Obligation this places on Wave 4

Accumulating gradients across mini-batches is a real requirement (plan section
3.5, `grad_accum`). It must get its **own entry point**, something like
`backward_accumulate()`, that seeds without clearing and is opted into
deliberately. It must not come back as "just call `backward()` twice", which is
the behaviour this guard exists to forbid.

---

## KL-006 - Backward closures capture forward values, which an in-place weight update would make stale

| | |
|---|---|
| **Status** | **RESOLVED** by construction (M2 Wave 1) |
| **Introduced** | M1 Wave 3 |
| **Fixed in** | `Tape::finish_step` + the `Optimizer::step` signature |
| **Affects** | `gltrain/src/tensor/tensor.rs` - the `matmul`, `mul`, `div`, `sqrt` and `relu` backward closures |
| **Regression test** | `finish_step_clears_the_tape_and_returns_the_gradients`, `finish_step_returns_an_empty_store_when_backward_was_never_called`, `finish_step_resets_the_node_id_counter` |

### What

Several ops capture the *values* of their forward operands at record time,
because their gradients depend on the operand data rather than only on its
shape:

| Op | Captured | Needed for |
|---|---|---|
| `matmul` | `a_data`, `b_data` | `dA = dC @ B^T`, `dB = A^T @ dC` |
| `mul` | `a_data`, `b_data` | `dA = dC * B`, `dB = dC * A` |
| `div` | `b_data`, `a_data` | `dA = dC / B`, `dB = -dC * A / B^2` |
| `sqrt` | the forward **output** | `dA = dC / (2*y)` |
| `relu` | `a_data` | the positive-input mask |

The capture is a `Vec<f32>` snapshot taken when the node is recorded. It is a
copy, not a view of the tensor's storage.

### When it bit

M2's optimizer writes updated weights back through
`TPParameter::set_data` -> `Tensor::replace_data`, which mutates storage in
place so the parameter keeps its `TensorId` across steps (optimizer state is
keyed on that ID). Any tape still holding a capture of that weight would keep
the **pre-update** values, and the next backward pass would compute gradients
from weights that no longer exist.

There is no error and no crash. The loss curve stays plausible and is quietly
wrong, which is the most expensive failure shape this crate has.

### The resolution: option (a), as a type-level guarantee

`KNOWN_ISSUES.md` recommended option (a), "require `tape.clear()` before
`optimizer.step()`, enforced with a guard". It shipped in that spirit but in a
stronger form, because the literal reading does not work: the gradients the
optimizer needs *live in* `grad_store`, so a guard that merely refused a
non-empty tape would make it impossible to ever read them.

`Tape::finish_step()` does both halves atomically instead:

```rust
pub fn finish_step(&mut self) -> VLGradStore {
    let grads = std::mem::take(&mut self.grad_store);
    self.nodes.clear();
    self.tensors.clear();
    self.next_node_id = 0;
    grads
}
```

and `Optimizer::step` takes the returned `VLGradStore`, never a `&mut Tape`:

```rust
fn step(&mut self, params: &mut [&mut TPParameter<B>], grads: &VLGradStore) -> Result<()>;
```

**A `VLGradStore` cannot be obtained without the tape already being cleared.**
There is no ordering for a caller to get wrong and no boolean anyone has to
remember to check: holding the gradients *is* the proof that no live closure
can observe the weights the step is about to overwrite. A runtime `if
tape.is_empty()` check would have been strictly weaker, since it could only
have run after the grads were already gone.

### What this does not cover

Nothing else in the crate mutates tensor storage. `replace_data` is
`pub(crate)` and `TPParameter::set_data` is its only caller, so the mutation
surface is exactly one function and it is on the path this guard protects.
A future in-place backend kernel would reopen the question and would need
re-checking against this entry, which is why the entry stays on the record.

### Not chosen

- **(b) Optimizer produces a new tensor.** Every parameter would get a fresh
  `TensorId` each step, and optimizer state keyed on that ID would need
  rekeying every step. Rejected: it moves the cost from a guard to a permanent
  per-step rekey.
- **(c) Capture `Arc<B::Storage>` instead of a `Vec<f32>` snapshot.** Removes
  the class of problem entirely and costs nothing when nobody mutates, but it
  puts a backend type inside a captured closure and leaks `B` into the tape's
  own types. Still the better answer if the tape ever has to survive a step;
  it does not today.
