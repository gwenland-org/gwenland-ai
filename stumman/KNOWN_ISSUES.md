# Stummañ — Known Issues

Tracked limitations in the `stumman` crate.

These are **not TODOs**. Each entry is a deliberate, documented consequence of a
design decision, recorded here so a later wave does not rediscover it as a bug
or "fix" it speculatively. Every entry names the milestone that owns its
resolution.

**Status key** — `KNOWN LIMITATION`: accepted, resolution scheduled ·
`ACCEPTED`: will not change · `OPEN QUESTION`: undecided, blocks a named wave.

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
