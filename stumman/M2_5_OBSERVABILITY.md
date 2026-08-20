# Stummañ M2.5 — Observability (Deskiñ)

**Status:** complete
**Branch:** `stumman-m2.5-observability` (off `stumman-m2`)
**Depends on:** Stummañ M2 (`f88b97d`)
**Consumed by:** glbench v3, Wave 4 (`training::collector`)
**Landed:** 2026-08-19

## Scope

Adds an observer hook to `Trainer` without changing training semantics or any
existing test's behaviour.

## Deliverables

- `src/train/observe.rs` — `VLTrainingStep`, `trait StepObserver`
- `src/train/trainer.rs` — `set_observer`, `clear_observer`, `current_epoch`,
  phase timing, epoch index, `emit_observation`
- `src/autograd/grad_store.rs` — `iter()`
- `src/train/mod.rs`, `src/lib.rs` — re-exports
- `examples/observed_training.rs` — the overhead measurement
- `tests/observer_boundary.rs` — the external-consumer proof (added
  2026-08-20 during the glbench Wave 3 audit; see below)

## What is deliberately not here

- `VLStepCollector` — lives in glbench `training::collector` (Wave 4)
- Gradient bit profiling — driven by glbench when `wants_tensors()` is true
- Peak RSS — needs a platform read; not required by anything M2.5 unblocks
- Any change to an optimizer implementation
- Any new external dependency. `Instant`, `HashMap`, `Rc`, `RefCell` are std

## Constraints respected

- **KL-001.** `StepObserver` is dyn-compatible: no type parameter, no
  `Self`-by-value receiver, no `Clone` supertrait. `Box<dyn StepObserver>`
  compiles, and a test asserts it so a later edit cannot erase it silently.
- **KL-006.** The observer runs after `optimizer.step()`, and the tape has been
  empty since `finish_step()`. It receives `&VLGradStore` for the duration of
  one call and can neither mutate nor retain it. Every existing KL-006
  regression test passes **unmodified**.
- **Zero cost when unobserved.** `observer.is_some()` is read once at the top of
  `train_step`; when false, no clock is read and no gradient is walked.
  Verified by loss comparison, not by inspection.

## Findings

Two things the design brief assumed that the code did not agree with. Both were
found by running it.

### The observer must run after the update, not before it

The brief placed the observer window between `finish_step()` and
`optimizer.step()`. That cannot work: `VLTrainingStep` carries `optimizer_ns`,
and a callback that fires before the optimizer runs has no value to put there.
The brief's own doc comment for `on_step` said "after the parameter update",
contradicting its own flow diagram.

Running late is safe. `grads` is a local owned by `train_step`, and
`Optimizer::step` takes it by shared reference (`adamw.rs:193` reads it through
`grads.get`), so it is fully alive afterwards. `VLGradStore::take` is documented
as "the optimizer uses this" but no optimizer calls it — only tests do.

Placing the window after the update also keeps `total_ns` honest: it is measured
before the observer window opens, so installing an observer does not inflate the
number the observer reports.

### `VLGradStore` holds activations, not just parameters

`Tape::finish_step` returns a gradient for **every tensor the tape touched**. On
the M2 trainer that is **9 entries for 2 parameters** — the rest are
intermediate activations.

The first implementation walked the whole store, and reported `grad_count = 9`
with an L2 norm mixing activation gradients into it. That number is not the
gradient norm anyone means by the phrase: every framework's `clip_grad_norm_`
computes it over parameters, and gradient-health analysis reads it that way.

`VLTrainingStep`'s gradient fields therefore cover **trainable parameters
only**, looked up by `TPParameter::id`. The full store still reaches an observer
that wants it, through `on_tensors`. A test pins both halves so a later edit
cannot re-merge them.

## Measured overhead

`cargo run --release --example observed_training`, 1000 steps, best of 5,
interleaved. Measured 2026-08-19.

| Layer | no observer | observer, no tensors | observer, with tensors |
|---|---|---|---|
| 64×64 | 14,857 ns/step | 15,499 (**+4.3%**) | 16,303 (**+9.7%**) |
| 256×256 | 35,542 ns/step | 39,640 (**+11.5%**) | 42,810 (**+20.4%**) |
| 512×512 | 78,905 ns/step | 90,103 (**+14.2%**) | 90,646 (**+14.9%**) |

Final loss bit-identical across all three configurations at every size.

**The overhead grows with layer width; it does not shrink.** The example's first
draft claimed the opposite — that the cost was a fixed per-step charge whose
ratio would fall as the step grew. The sweep disproved it. The observer walks
the parameter gradients, a LoRA adapter has `2·r·d` parameters, and on this
backend the step itself does not grow as fast as `d²`, so the ratio rises.

Do not extrapolate these numbers to M3's model. Re-run the example there.

### The `state_tensors` delta shrinks with size, and that is real

At 512×512 the gap between `with tensors` and `no tensors` looked suspiciously
small, so it was repeated rather than quoted:

| 512×512 | no observer | no tensors | with tensors | delta |
|---|---|---|---|---|
| repeat 1 | 78,905 | 90,103 (+14.2%) | 90,646 (+14.9%) | 0.6 pt |
| repeat 2 | 77,778 | 87,412 (+12.4%) | 89,355 (+14.9%) | 2.5 pt |

So part of repeat 1's 0.6 was noise, but the direction survives: the
`state_tensors` delta is far smaller at 512×512 than the 9 points measured at
256×256. Per copied element it is 3× cheaper — roughly 0.39 ns/float at 256
against 0.12 ns/float at 512 — which is consistent with per-allocation overhead
dominating at the smaller size rather than with anything about the optimizer.

Recorded as an observation, not a conclusion. Nothing in M2.5 depends on it, and
one plausible mechanism is not a measurement of that mechanism.

## Test count

| | |
|---|---|
| M2 baseline | 302 |
| M2.5 additions (unit) | 15 |
| **Unit total** | **317** |
| `tests/observer_boundary.rs` (integration) | 9 |
| doc-test | 1 |
| **Suite total** | **327** |

All 302 M2 tests pass unmodified. `cargo clippy --all-targets` reports zero new
warnings from `stumman` (two pre-existing `unused_mut` warnings come from
`glproc/src/loader.rs` and are untouched).

New tests:

| Test | Guards |
|---|---|
| `step_observer_is_object_safe` | KL-001 — `Box<dyn StepObserver>` compiles |
| `nan_and_inf_are_separate_fields_on_the_record` | the two counters do not collide |
| `iter_yields_every_entry_exactly_once` | `iter` reaches every id, data and shape intact |
| `iter_on_an_empty_store_yields_nothing` | a gradient-free step does not panic |
| `unobserved_train_step_is_deterministic` | the determinism baseline |
| `installing_an_observer_does_not_change_the_loss` | **the load-bearing test** |
| `requesting_tensors_does_not_change_the_loss` | same, with `state_tensors` on |
| `observer_receives_step_index_and_epoch` | global monotonic index, epoch from the loop |
| `a_bare_train_step_reports_epoch_zero` | `train` resets the counter |
| `observer_loss_matches_the_returned_value` | observed loss is the returned loss |
| `observed_gradient_statistics_describe_the_step` | counts, norm, lr source |
| `the_store_holds_activations_but_the_record_counts_parameters` | the §Findings distinction |
| `tensor_payload_is_opt_in` | `wants_tensors` gates `state_tensors` |
| `phase_timings_are_attributed_within_the_total` | phases fit inside the total |
| `clearing_the_observer_stops_observation` | `clear_observer` really detaches |

## glbench v3 Wave 3 reconciliation (2026-08-20)

This milestone was written against its own brief, not against
`architecture/glbench-v3/DESIGN.md`. Audited against that document's Wave 3 and
§7.1, every deliverable is present — `observe.rs`, the `Trainer` plumbing with
phase timing and epoch index, `VLGradStore::iter`, and the re-exports. Two
things the audit changed or found worth stating.

### Declared deviation: the observer runs after `optimizer.step()`

§7.1 states the constraint as *"the observer reads gradients **between**
`finish_step()` and `optimizer.step()`"*. This implementation calls it after the
update instead, for the reason `observe.rs`'s module docs give: `optimizer_ns`
cannot be reported by a callback that runs before the optimizer does, so an
early record would carry a zero or a lie in that field.

The deviation is safe because §7.1's *reason* for that placement — KL-006 — does
not depend on it. KL-006's guarantee is that the tape is empty before any weight
write, and `finish_step()` empties it well before either candidate position. The
observer never sees a tape. Reading `grads` late is likewise safe: it is a local
`train_step` owns, and `Optimizer::step` borrows it shared.

**§7.1's bullet is over-specified and should be corrected to name the invariant
(KL-006, tape emptied) rather than a position that cannot carry the timing.**

### Added: `tests/observer_boundary.rs`

Every M2.5 test lives inside the crate, where private paths are reachable. None
of them could prove the thing Wave 4 depends on most: that a consumer **outside**
`stumman` can name every type in the trait signature, implement `StepObserver`,
install it, and read back what it collected. An integration test links the crate
externally, so it can.

It also settles a question Wave 4 would otherwise have hit at implementation
time. `clear_observer` hands back `Box<dyn StepObserver>`, which cannot be
downcast without `Any` — so **the returned box is not how a caller reads the
results**. The working pattern is shared ownership (`Rc<RefCell<…>>`), and the
test demonstrates it under the name `VLStepCollectorProxy` so Wave 4's real
`VLStepCollector` can copy it.

What the nine tests pin, beyond compilation: global gap-free `index` across
epoch boundaries, `epoch` tracking `train()`, D-19 sampling being expressible
from `index` alone (including the endpoint rule, with a last step that is not a
multiple of N), `grad_count` counting parameters while `on_tensors` still
receives the whole store, optimizer state arriving non-empty, phase timings
fitting inside the total, and bit-identical loss sequences observed vs not.

## Note on the naming skill

`gl-agent-skills/stumman-naming/SKILL.md`'s Breton table lists seven sub-systems
and omits **Deskiñ** (`train/`), although `src/lib.rs:8` declares it and every
file under `train/` follows it. Not fixed here, because the skill is outside
`stumman/` and this milestone's brief said to stop at the crate boundary. It is
a one-row addition and should land with whoever next edits that skill.
