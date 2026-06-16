# Design Document — GWEN-223: Checkpoint Resume — Serialize AdamW Optimizer State (True Resume)

## Overview

GWEN-222 restored LoRA adapter weights on `--resume` but left AdamW starting from scratch — fresh
`m1`/`m2` moment vectors and step counter at zero. For hardware that rarely throttles, the
momentum warm-up over a handful of steps is negligible. On GwenLand's target i3 (no GPU,
200–320 s/step), thermal throttling turns that "handful of steps" into 10–20 minutes of
degraded training per resume event. Compounded across multiple pauses, the cumulative loss in
momentum signal and the bias-correction drift are real.

GWEN-223 closes that gap by persisting the full AdamW state — first moments `m1`, second moments
`m2`, and the global step counter `t` — to a parallel safetensors file
`checkpoint_{step:06}_adamw.safetensors`. On resume, the state is reloaded and injected back
into the parallel `HashMap` that tracks moments manually (Option A). Missing or corrupt state
files degrade gracefully to GWEN-222 behaviour; save failures never abort the training run.

The feature is purely additive: no public API surfaces change shape, no existing test breaks,
and the `_adamw` file is always optional from the reader's perspective.

---

## Architecture

### Component Map

```
packages/tui/src/commands/train.rs
  (no new flags — GWEN-222's --resume is reused as-is)

packages/core/src/train/
  layered_training_loop.rs      ← primary change site
    LayeredTrainingLoop         ← +moment_store, +step_t fields
    run()                       ← +update_moments() call after each adamw.step()
    save_checkpoint()           ← promoted to method; +save_adamw_state() call
    load_checkpoint()           ← +load_adamw_state() call (graceful fallback)

  adamw_state.rs                ← NEW module
    MomentStore                 ← type alias HashMap<String, (Tensor, Tensor)>
    save_adamw_state()          ← writes checkpoint_{step}_adamw.safetensors
    load_adamw_state()          ← reads it back; returns Ok(None) on missing file
    adamw_state_path()          ← derives _adamw path from weight checkpoint path

  checkpoint_resumer.rs         ← no change (GWEN-222)
  config.rs                     ← no change
  error.rs                      ← no change
```

The design deliberately keeps the new logic in one new module (`adamw_state.rs`) and one
promotion (`save_checkpoint` becomes a method). The `LayeredTrainingLoop` struct gains two
fields; all other struct members are untouched.

### Data-Flow: Save Path

```
run() — after every optimizer step
  │
  ├─ update_moments(grads, &vars, step_t)
  │     └─ for each trainable Var:
  │           key = varmap_key_for(var)
  │           m1' = β1·m1 + (1-β1)·g
  │           m2' = β2·m2 + (1-β2)·g²
  │           moment_store.insert(key, (m1', m2'))
  │
  ├─ step_t += 1
  │
  └─ [every 500 steps]
        save_checkpoint(&varmap, &config, step_t)  ← GWEN-222 weight file
        save_adamw_state(&moment_store, step_t, &config.output_path)
              ├─ build flat tensor map:
              │     "layer_N.proj.lora_a.m1" → Tensor
              │     "layer_N.proj.lora_a.m2" → Tensor
              │     "layer_N.proj.lora_b.m1" → Tensor
              │     "layer_N.proj.lora_b.m2" → Tensor
              │     "step"                   → Tensor([step_t as u64])
              ├─ write checkpoint_{step:06}_adamw.safetensors via safetensors_write()
              └─ [on Err] log warning, do NOT propagate — training continues
```

### Data-Flow: Load / Resume Path

```
native_runner.rs: run_native_local()
  │
  ├─ checkpoint_resumer::resolve_checkpoint(mode, output_path)
  │     → (Some(weight_ckpt_path), step)          ← GWEN-222
  │
  ├─ LayeredTrainingLoop::new(config, gguf_path, batches, varmap, tx, step)
  │     └─ moment_store = HashMap::new()          ← empty; populated on load
  │        step_t = step                          ← seeded from resumed step
  │
  ├─ training_loop.load_checkpoint(&weight_ckpt_path)    ← GWEN-222 (weights)
  │
  └─ training_loop.load_adamw_state(&weight_ckpt_path)   ← GWEN-223 (moments)
        ├─ adamw_state_path(weight_ckpt_path)
        │     → checkpoint_{step:06}_adamw.safetensors
        ├─ [file missing] → log warning, return Ok(())  ← GWEN-222 fallback
        ├─ [file present] → deserialize flat tensor map
        │     for each "layer_N.proj.lora_{a|b}.{m1|m2}" key:
        │       moment_store.insert(varmap_key, (m1_tensor, m2_tensor))
        │     step_t = tensor["step"].to_scalar::<u64>()
        └─ [deserialize Err] → log warning, clear moment_store, return Ok(())
```

### System Diagram

```mermaid
graph TD
    subgraph Training Loop ["LayeredTrainingLoop"]
        OPT[AdamW optimizer]
        MS[moment_store\nHashMap<String,(Tensor,Tensor)>]
        ST[step_t: usize]
        VM[VarMap\nlora weights]
    end

    subgraph Checkpoint Save ["Every 500 steps"]
        WF[checkpoint_{step}.safetensors\nLoRA weights only]
        AF[checkpoint_{step}_adamw.safetensors\nm1/m2 tensors + step scalar]
    end

    subgraph Checkpoint Load ["On --resume"]
        WL[load_checkpoint_into_varmap\nGWEN-222]
        AL[load_adamw_state\nGWEN-223]
        FB[Fallback: fresh AdamW\nif _adamw file missing]
    end

    OPT -->|step| MS
    OPT -->|step| ST
    MS -->|save_adamw_state| AF
    VM -->|varmap.save| WF
    WF -->|resolve_checkpoint| WL
    AF -->|adamw_state_path| AL
    AL -->|missing/corrupt| FB
    WL -->|restored lora_a/b| VM
    AL -->|restored m1/m2| MS
    AL -->|restored step_t| ST
```

### Key Invariants

1. **Shape invariant** — every `(m1, m2)` pair in `moment_store` has the identical shape as
   the corresponding LoRA weight tensor in `VarMap`.
2. **Step invariant** — after a true resume, `step_t` equals the step at which the checkpoint
   was saved. Bias correction `1 - βᵢᵗ` thus continues on the correct axis.
3. **Count invariant** — for a non-fallback run with `N` layers and 7 projections,
   `moment_store.len()` equals `N × 7 × 2` (lora_a + lora_b per projection per layer).
   In fallback mode (single-tensor fixtures) it equals `N × 2`.
4. **Non-fatal save** — a write error on the `_adamw` file must not propagate; it is logged
   and swallowed. The weight checkpoint proceeds regardless.
5. **Non-fatal load** — a missing or corrupt `_adamw` file must not propagate; it is logged
   and the training run continues from fresh moments (GWEN-222 behaviour).
6. **Round-trip fidelity** — for any `(m1, m2)` pair saved at step `S`, loading the `_adamw`
   file produces tensors numerically identical to the originals (bitwise for F32 round-trip
   through safetensors).

---

## Components and Interfaces

### `LayeredTrainingLoop` — struct additions

Two new fields are appended to the existing struct (all other fields unchanged):

```rust
pub struct LayeredTrainingLoop {
    // ... all existing fields from GWEN-222 ...

    /// GWEN-223: manually maintained AdamW moment state.
    /// Key: VarMap key string (e.g. "l0.attn_q.lora_a").
    /// Value: (m1, m2) tensors with the same shape as the weight tensor.
    /// Empty at construction; populated after each optimizer step.
    moment_store: HashMap<String, (Tensor, Tensor)>,

    /// GWEN-223: global AdamW step counter, separate from optimizer_steps
    /// in run() to keep bias correction aligned with the checkpoint axis.
    /// Seeded from initial_step on construction; incremented in update_moments().
    step_t: usize,
}
```

The existing `global_step` field (GWEN-222) tracks the checkpoint naming axis. `step_t` is
the AdamW bias-correction counter, initialized from `initial_step` just like `global_step`.
For a fresh run both are 0; after resume both equal the restored step.

### `adamw_state.rs` — new module

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use candle_core::Tensor;
use anyhow::Result;

/// Map from VarMap key to (m1, m2) moment tensors.
pub(crate) type MomentStore = HashMap<String, (Tensor, Tensor)>;

/// Derive the AdamW state file path from the weight checkpoint path.
///
/// "checkpoint_000500.safetensors" → "checkpoint_000500_adamw.safetensors"
pub(crate) fn adamw_state_path(weight_ckpt_path: &Path) -> PathBuf

/// Write moment_store + step_t to checkpoint_{step:06}_adamw.safetensors.
///
/// Flat key layout in the file:
///   "layer_{i}.{proj}.lora_a.m1" / ".m2"
///   "layer_{i}.{proj}.lora_b.m1" / ".m2"
///   "step" → [step_t as u64, 1] tensor
///
/// Keys are derived from the VarMap key ("l{i}.{proj}.lora_{a|b}") via
/// varmap_key_to_adamw_prefix().
///
/// Returns Ok(()) even on write failure — caller must log and continue.
pub(crate) fn save_adamw_state(
    store: &MomentStore,
    step_t: usize,
    output_path: &Path,
    step: usize,
) -> Result<()>

/// Load the AdamW state file paired with weight_ckpt_path.
///
/// Returns Ok(None) if the _adamw file does not exist.
/// Returns Ok(None) and logs a warning on any deserialization error.
/// Returns Ok(Some((store, step_t))) on success.
pub(crate) fn load_adamw_state(
    weight_ckpt_path: &Path,
) -> Result<Option<(MomentStore, usize)>>

/// Convert a VarMap key to the flat AdamW state key prefix used in the file.
///
/// "l0.attn_q.lora_a"  → "layer_0.attn_q.lora_a"
/// "l12.ffn_down.lora_b" → "layer_12.ffn_down.lora_b"
///
/// Appends ".m1" or ".m2" at the call site.
pub(crate) fn varmap_key_to_adamw_prefix(varmap_key: &str) -> Option<String>

/// Convert an AdamW state key prefix back to a VarMap key.
///
/// "layer_0.attn_q.lora_a" → "l0.attn_q.lora_a"
///
/// Used when loading the state file to populate moment_store.
pub(crate) fn adamw_prefix_to_varmap_key(prefix: &str) -> Option<String>
```

---

## Data Models

### `MomentStore` key layout

The `MomentStore` uses VarMap key strings as its keys. This lets the update loop iterate
`varmap.all_vars()`, resolve each Var's name, and write directly — no translation needed
during training. Translation only occurs at save/load time.

| VarMap key pattern | Example | m1 shape | m2 shape |
|---|---|---|---|
| `l{n}.{proj}.lora_a` | `l0.attn_q.lora_a` | `[rank, d_in]` | `[rank, d_in]` |
| `l{n}.{proj}.lora_b` | `l0.attn_q.lora_b` | `[d_out, rank]` | `[d_out, rank]` |
| `l{n}.lora_a` (fallback) | `l0.lora_a` | `[r, hidden]` | `[r, hidden]` |
| `l{n}.lora_b` (fallback) | `l0.lora_b` | `[hidden, r]` | `[hidden, r]` |

### `checkpoint_{step:06}_adamw.safetensors` — flat key layout

```
"layer_0.attn_q.lora_a.m1"   → Tensor F32 [rank, d_in]
"layer_0.attn_q.lora_a.m2"   → Tensor F32 [rank, d_in]
"layer_0.attn_q.lora_b.m1"   → Tensor F32 [d_out, rank]
"layer_0.attn_q.lora_b.m2"   → Tensor F32 [d_out, rank]
"layer_0.attn_k.lora_a.m1"   → Tensor F32 [rank_k, d_in_k]
...  (7 projections × N layers × 2 sides × 2 moments = 28N entries)
"step"                        → Tensor U64 [1]
```

The `"layer_{i}"` prefix mirrors GWEN-222's `checkpoint_{step:06}.safetensors` naming:
both use the same layer numbering. The difference is the `_adamw` filename suffix and the
`.m1` / `.m2` key suffixes — zero new naming convention to learn.

### `step` scalar

Stored as a `U64` tensor of shape `[1]` (not a bare scalar) because the safetensors format
requires a shape. Loaded with `.to_vec1::<u64>()` and read at index `[0]`.

### `NewTrainConfig` — no change

No new fields are needed. The AdamW hyperparameters `β1=0.9`, `β2=0.999`, `ε=1e-8`, `weight_decay=0.01`
are already in `ParamsAdamW` and are reconstructed identically on every run from `config.lr`.
They do not need to be serialized.

---

## Low-Level Design

### AdamW Manual Update Formula (Option A)

Because `candle_nn::AdamW` does not expose `m1`/`m2` as public fields, we maintain a parallel
`MomentStore` and apply the standard AdamW moment update manually after each
`self.adamw.step(&grads)` call. The optimizer step and the manual update must see the same
gradients, so we clone the gradients before the step (or retain the pre-clipped grads that
were passed to `adamw.step`):

```
// After: self.adamw.step(&grads)
// grads here are the scaled + clipped gradients

fn update_moments(
    &mut self,
    grads: &GradStore,
    vars: &[Var],
    varmap_data: &HashMap<String, Var>,
) -> Result<()> {
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;

    for var in vars {
        // Resolve this Var back to its VarMap key.
        let key = varmap_key_for(var, varmap_data);  // see Key Derivation below
        let Some(key) = key else { continue };
        let Some(grad) = grads.get(var.as_tensor()) else { continue };

        let (m1_prev, m2_prev) = self.moment_store
            .entry(key.clone())
            .or_insert_with(|| {
                let shape = var.as_tensor().shape().clone();
                let dev = var.as_tensor().device().clone();
                (
                    Tensor::zeros(shape.clone(), DType::F32, &dev).unwrap(),
                    Tensor::zeros(shape,          DType::F32, &dev).unwrap(),
                )
            });

        // m1 = β1·m1_prev + (1-β1)·g
        let m1 = m1_prev.affine(beta1, 0.0)?.add(&grad.affine(1.0 - beta1, 0.0)?)?;
        // m2 = β2·m2_prev + (1-β2)·g²
        let g_sq = grad.sqr()?;
        let m2 = m2_prev.affine(beta2, 0.0)?.add(&g_sq.affine(1.0 - beta2, 0.0)?)?;

        *m1_prev = m1;
        *m2_prev = m2;
    }
    self.step_t += 1;
    Ok(())
}
```

The moment tensors are kept as plain `Tensor` (not `Var`) — they are state, not trainable
parameters. They live only on CPU (same device as all LoRA tensors).

**Numerical identity guarantee:** Because the `candle_nn::AdamW` implementation applies the
same formula internally, the moments we track in `MomentStore` will equal the optimizer's
internal state at every step. This is verified by the property test in Wave 5.

### Key Derivation — VarMap key ↔ AdamW state key

The VarMap stores variables by string key. Candle's `VarMap::data()` returns a
`Arc<Mutex<HashMap<String, Var>>>`. We can resolve a `Var` back to its key by locking the
map and scanning for pointer equality on the underlying tensor storage:

```rust
fn varmap_key_for(var: &Var, data: &HashMap<String, Var>) -> Option<String> {
    // Tensor storage identity: compare device location + data pointer via elem_count + id.
    // Candle exposes Tensor::id() (a monotonic u64) for this purpose.
    let target_id = var.as_tensor().id();
    data.iter()
        .find(|(_, v)| v.as_tensor().id() == target_id)
        .map(|(k, _)| k.clone())
}
```

This scan is O(vars) per step but `vars` is small (7 projections × N layers × 2 ≈ 28–56
tensors for typical models). The loop runs once per optimizer step (every 500 steps a
checkpoint is saved on top).

**VarMap key → AdamW state key:**

```
VarMap key:      "l{n}.{proj}.lora_{a|b}"
AdamW state key: "layer_{n}.{proj}.lora_{a|b}.{m1|m2}"

Transformation:
  1. Strip leading 'l' → parse layer index n
  2. Prefix with "layer_" + n + "."
  3. Append ".m1" or ".m2"
```

The `varmap_key_to_adamw_prefix` function handles step 1–2; the caller appends the suffix.

**Reverse (load path):**

```
AdamW state key: "layer_{n}.{proj}.lora_{a|b}.{m1|m2}"
VarMap key:      "l{n}.{proj}.lora_{a|b}"

  1. Strip "layer_" prefix
  2. Read n until first '.'
  3. Replace "layer_{n}." with "l{n}."
  4. Strip trailing ".m1" or ".m2"
```

Fallback keys (`l{n}.lora_a` / `l{n}.lora_b`) follow the same rule with no `{proj}` segment.

### `save_checkpoint` — promoted to method

Currently `save_checkpoint` is a module-level free function that takes `&VarMap`. To also
write the `_adamw` file without changing the outer `run()` call structure, it is promoted to
a method on `LayeredTrainingLoop`:

```rust
fn save_checkpoint_and_adamw_state(&self, step: usize) -> () {
    // 1. Save LoRA weights (GWEN-222 path)
    let weight_path = self.config.output_path.join(
        format!("checkpoint_{:06}.safetensors", step)
    );
    if let Err(e) = std::fs::create_dir_all(&self.config.output_path)
        .and_then(|_| self.varmap.save(&weight_path))
    {
        eprintln!("[checkpoint] WARNING: failed to save weights: {e}");
        // Do not propagate — training continues.
        return;
    }
    eprintln!("[checkpoint] saved → {}", weight_path.display());

    // 2. Save AdamW state (GWEN-223 path — failure is non-fatal)
    if let Err(e) = crate::train::adamw_state::save_adamw_state(
        &self.moment_store,
        self.step_t,
        &self.config.output_path,
        step,
    ) {
        eprintln!(
            "[resume] WARNING: failed to save AdamW state for checkpoint {step}: {e}"
        );
        // Non-fatal — weight checkpoint was already written.
    }
}
```

The call site in `run()` changes from:
```rust
if optimizer_steps % 500 == 0 {
    save_checkpoint(&self.varmap, &self.config, optimizer_steps)?;
}
```
to:
```rust
if optimizer_steps % 500 == 0 {
    self.save_checkpoint_and_adamw_state(optimizer_steps);
}
```

Note the `?` is removed: both saves are now best-effort. A warning is sufficient because
the next checkpoint interval will retry.

### `load_adamw_state` — full call sequence

In `native_runner.rs`, after the GWEN-222 weight load:

```rust
if let Some(ref path) = ckpt_path {
    training_loop.load_checkpoint(path)?;                // GWEN-222
    training_loop.load_adamw_state(path);                // GWEN-223 (infallible)
}
```

`load_adamw_state` is a method on `LayeredTrainingLoop`, not on the module:

```rust
pub fn load_adamw_state(&mut self, weight_ckpt_path: &Path) {
    match crate::train::adamw_state::load_adamw_state(weight_ckpt_path) {
        Ok(Some((store, step_t))) => {
            self.moment_store = store;
            self.step_t = step_t;
            eprintln!(
                "[resume] AdamW state restored: {} moment pairs, step_t={}",
                self.moment_store.len(), self.step_t
            );
        }
        Ok(None) => {
            eprintln!(
                "[resume] AdamW state not found for checkpoint {}, \
                 resuming with fresh optimizer (GWEN-222 behavior)",
                weight_ckpt_path.display()
            );
            // moment_store stays empty; step_t keeps initial_step value.
        }
        Err(e) => {
            eprintln!(
                "[resume] WARNING: failed to load AdamW state: {e}. \
                 Resuming with fresh optimizer."
            );
            self.moment_store.clear();
            // step_t keeps initial_step — bias correction stays correct even
            // without moments (bias correction at wrong step is still better
            // than step_t=0 on a warm run).
        }
    }
}
```

### `save_adamw_state` — file write implementation

The `_adamw.safetensors` file is written using `candle_core::safetensors::save` (the standard
candle safetensors write path), which serializes a `HashMap<String, Tensor>` directly:

```rust
pub(crate) fn save_adamw_state(
    store: &MomentStore,
    step_t: usize,
    output_path: &Path,
    step: usize,
) -> Result<()> {
    std::fs::create_dir_all(output_path)?;
    let path = output_path.join(format!("checkpoint_{:06}_adamw.safetensors", step));

    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    for (varmap_key, (m1, m2)) in store {
        let Some(prefix) = varmap_key_to_adamw_prefix(varmap_key) else {
            continue;  // skip unrecognised keys silently
        };
        tensors.insert(format!("{prefix}.m1"), m1.clone());
        tensors.insert(format!("{prefix}.m2"), m2.clone());
    }

    // Store step_t as a U64 scalar tensor (shape [1]).
    let step_tensor = Tensor::from_vec(
        vec![step_t as u64],
        (1,),
        &candle_core::Device::Cpu,
    )?;
    tensors.insert("step".to_string(), step_tensor);

    candle_core::safetensors::save(&tensors, &path)?;
    eprintln!("[checkpoint] AdamW state saved → {}", path.display());
    Ok(())
}
```

### `load_adamw_state` — file read implementation

```rust
pub(crate) fn load_adamw_state(
    weight_ckpt_path: &Path,
) -> Result<Option<(MomentStore, usize)>> {
    let adamw_path = adamw_state_path(weight_ckpt_path);

    if !adamw_path.exists() {
        return Ok(None);
    }

    // Load all tensors from the safetensors file.
    let tensors = candle_core::safetensors::load(&adamw_path, &candle_core::Device::Cpu)
        .map_err(|e| anyhow::anyhow!("failed to read AdamW state: {e}"))?;

    // Extract step_t from the "step" key.
    let step_t = tensors
        .get("step")
        .and_then(|t| t.to_vec1::<u64>().ok())
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| anyhow::anyhow!("AdamW state file missing 'step' key"))?;

    // Re-pair m1/m2 back into MomentStore (VarMap key space).
    let mut m1_map: HashMap<String, Tensor> = HashMap::new();
    let mut m2_map: HashMap<String, Tensor> = HashMap::new();

    for (key, tensor) in &tensors {
        if key == "step" { continue; }
        if let Some(prefix) = key.strip_suffix(".m1") {
            if let Some(vk) = adamw_prefix_to_varmap_key(prefix) {
                m1_map.insert(vk, tensor.clone());
            }
        } else if let Some(prefix) = key.strip_suffix(".m2") {
            if let Some(vk) = adamw_prefix_to_varmap_key(prefix) {
                m2_map.insert(vk, tensor.clone());
            }
        }
    }

    let mut store = MomentStore::new();
    for (vk, m1) in m1_map {
        if let Some(m2) = m2_map.remove(&vk) {
            store.insert(vk, (m1, m2));
        }
        // m1 without m2: silently skip (partial write scenario).
    }

    Ok(Some((store, step_t as usize)))
}
```

### `adamw_state_path` — path derivation

```rust
pub(crate) fn adamw_state_path(weight_ckpt_path: &Path) -> PathBuf {
    let stem = weight_ckpt_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    let parent = weight_ckpt_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}_adamw.safetensors"))
}
```

Examples:
- `./gwen-output/checkpoint_000500.safetensors` → `./gwen-output/checkpoint_000500_adamw.safetensors`
- `/abs/path/checkpoint_001000.safetensors` → `/abs/path/checkpoint_001000_adamw.safetensors`

---

## Error Handling

| Scenario | Handling | User-visible output | Training aborts? |
|---|---|---|---|
| `_adamw` file missing on resume | `Ok(None)` from `load_adamw_state` | `[resume] AdamW state not found for checkpoint {step}, resuming with fresh optimizer (GWEN-222 behavior)` | No |
| `_adamw` file corrupt / bad JSON header | `Err` caught in `load_adamw_state` method | `[resume] WARNING: failed to load AdamW state: {e}. Resuming with fresh optimizer.` | No |
| `_adamw` file has unpaired m1 (no m2) | silently skip that key in load | no output | No |
| `_adamw` file missing `step` key | `Err` from `load_adamw_state`, treated as missing | warning + fresh optimizer | No |
| AdamW state write fails (disk full, permissions) | `Err` caught in `save_checkpoint_and_adamw_state` | `[resume] WARNING: failed to save AdamW state for checkpoint {step}: {e}` | No |
| Weight checkpoint write fails | `eprintln!` warning in method | `[checkpoint] WARNING: failed to save weights: {e}` | No |
| `varmap_key_to_adamw_prefix` returns `None` for unknown key | key silently skipped in save | none | No |
| Tensor shape mismatch on load (m1 shape ≠ weight shape) | discovered at first step's `update_moments` when the key is re-inserted | shapes diverge; next checkpoint overwrites with correct shapes | No — incorrect until next checkpoint |
| VarMap `all_vars()` returns empty list | `update_moments` is a no-op; `moment_store` stays empty | none at step time; `_adamw` file will contain only `step` | No |

**Shape mismatch mitigation:** When `load_adamw_state` populates `moment_store`, the shapes of the
loaded tensors are not validated against the current VarMap. Adding that validation in Wave 4 is
recommended: for each `(varmap_key, (m1, m2))` in the loaded store, check
`m1.shape() == var.as_tensor().shape()` and drop the entry with a warning if they differ.

---

## Testing Strategy

### Wave 1 — Verify Candle AdamW internals (no code changes)

- Read `candle_nn::AdamW` source to confirm `m1`/`m2` are not exposed.
- Verify `candle_core::safetensors::save` accepts `HashMap<String, Tensor>` with `U64` dtype.
- Confirm `Tensor::id()` is available for key-derivation scan.
- Write `audit.md` noting exact versions and any API surface concerns.

### Wave 2 — In-memory bookkeeping (`moment_store` + `step_t`)

Unit tests in `layered_training_loop.rs` / `adamw_state.rs`:

- `test_update_moments_initial_zero` — after 1 step on a zero-gradient, moments equal zero.
- `test_update_moments_formula` — with known gradient `g`, verify
  `m1 = (1-β1)·g` and `m2 = (1-β2)·g²` after step 1 (previous moments were zero).
- `test_update_moments_step_t_increments` — after `N` steps, `step_t == N`.
- `test_moment_store_count_all_projections` — after construction + 1 step with a
  7-projection GGUF fixture, `moment_store.len() == num_layers × 7 × 2`.
- `test_moment_store_count_fallback` — single-tensor fixture gives `num_layers × 2` entries.
- `test_varmap_key_for_resolves` — insert a Var into VarMap, call `varmap_key_for`, assert key returned.
- `test_varmap_key_to_adamw_prefix` — table of (input, expected_output) pairs for the
  standard and fallback patterns.
- `test_adamw_prefix_to_varmap_key` — round-trip property: `adamw_prefix_to_varmap_key(varmap_key_to_adamw_prefix(k)?) == k`.

### Wave 3 — Save AdamW state to safetensors

Unit tests in `adamw_state.rs`:

- `test_save_adamw_state_creates_file` — after calling `save_adamw_state`, file exists.
- `test_save_adamw_state_filename_pattern` — filename matches `checkpoint_{step:06}_adamw.safetensors`.
- `test_save_adamw_state_contains_step_key` — loaded header contains `"step"` key.
- `test_save_adamw_state_key_count` — for N entries in moment_store, file contains `2N + 1` tensor keys.
- `test_save_adamw_state_non_fatal` — if output dir is read-only, function returns `Err` (caller should swallow).
- `test_adamw_state_path_derivation` — parametric test: verify the `_adamw.safetensors` suffix is correct for standard and non-standard checkpoint names.

### Wave 4 — Load AdamW state on resume

Unit tests in `adamw_state.rs`:

- `test_load_adamw_state_missing_file` — returns `Ok(None)`, no panic.
- `test_load_adamw_state_roundtrip` — save then load; assert all (m1, m2) tensors are numerically equal (element-wise F32 comparison within tolerance 1e-7).
- `test_load_adamw_state_step_roundtrip` — saved step `S` is restored as `S`.
- `test_load_adamw_state_corrupt_file` — write garbage bytes; `load_adamw_state` returns `Err`.
- `test_load_adamw_state_missing_step_key` — write a valid safetensors with no `"step"` key; returns `Err`.
- `test_load_adamw_state_shape_validation` — (Wave 4 enhancement) mismatched shape entries are dropped with warning.

Integration test in `native_runner.rs` test suite:

- `test_e2e_true_resume` — run N steps on a micro-GGUF, force-checkpoint, resume, assert `moment_store` is non-empty and `step_t == N`.

### Wave 5 — Validation (loss curve smoothness + regression)

- `test_moment_values_match_adamw_internal` — run the training loop for K steps, then compare the moments in `moment_store` against a reference AdamW applied to the same gradients. The reference is computed independently using the update formula with the same β1/β2 and the captured gradients. Assert element-wise max error < 1e-5.
- `test_loss_curve_no_step_back` — resume mid-run; assert loss at step N+1 after resume ≤ loss at step N before pause + small ε (no catastrophic spike).
- `test_no_regression_fresh_run` — a fresh (no-resume) training run produces identical losses with GWEN-223 code as without it (moment_store is empty; behaviour is identical to GWEN-222).

**Property-based test library**: `quickcheck` (already in `[dev-dependencies]`).

---

## Correctness Properties

*A property is a universal statement that must hold for all valid inputs and execution
histories. Each property below is a candidate for a `quickcheck` `#[quickcheck]` test.*

---

### Property 1: m1/m2 shape invariant

*For any* trainable Var with shape `S`, after any number of `update_moments` calls, the
tensors `moment_store[key].0` (m1) and `moment_store[key].1` (m2) shall have shape `S`.

**Validates: Requirements 3.5**

---

### Property 2: step_t after restore equals saved step

*For any* step count `S` where `S % 500 == 0`, saving the AdamW state at step `S` and
loading it into a fresh `LayeredTrainingLoop` shall produce `step_t == S`.

**Validates: Requirements 2.2, 6.2**

---

### Property 3: missing `_adamw` file does not abort training

*For any* weight checkpoint path `P` where `P` exists but
`P.stem() + "_adamw.safetensors"` does not exist, calling `load_adamw_state(P)` shall
return `Ok(None)` and the training loop shall proceed without error.

**Validates: Requirements 2.3, 7.2, 7.4**

---

### Property 4: save failure does not abort training

*For any* call to `save_checkpoint_and_adamw_state` where the `_adamw` write fails
(simulated by a read-only directory), the method shall return without panicking and the
weight checkpoint shall still exist.

**Validates: Requirements 1.4, 1.5, 7.3**

---

### Property 5: round-trip fidelity

*For any* `MomentStore` containing F32 tensors with arbitrary values in `(-1e6, 1e6)`,
calling `save_adamw_state` followed by `load_adamw_state` shall produce a `MomentStore`
where every element of every tensor is within `1e-7` of the original.

**Validates: Requirements 6.1**

---

### Property 6: moment_store entry count invariant

*For any* `LayeredTrainingLoop` initialized with an N-layer GGUF (7 projections, non-fallback),
after `K ≥ 1` optimizer steps, `moment_store.len()` shall equal `N × 7 × 2`.

*For any* `LayeredTrainingLoop` initialized in fallback mode, `moment_store.len()` shall
equal `N × 2`.

**Validates: Requirements 5.1, 5.2**

---

### Property 7: AdamW state key round-trip

*For any* valid VarMap key `k` matching the pattern `l{n}.{proj}.lora_{a|b}` (where `n`
is a non-negative integer, `proj` is one of the 7 known projection var_keys, and the side
is `lora_a` or `lora_b`):

```
adamw_prefix_to_varmap_key(varmap_key_to_adamw_prefix(k)?) == Some(k)
```

**Validates: Requirements 4.5**

---

### Property 8: adamw_state_path is a pure function of weight_ckpt_path

*For any* weight checkpoint path `P`, `adamw_state_path(P)` shall:
- Have the same parent directory as `P`.
- Have a filename ending in `_adamw.safetensors`.
- Have a filename that starts with `P.file_stem()`.

**Validates: Requirements 4.3, 4.4**

---

### Property 9: step tensor is parseable from any valid _adamw file

*For any* step value `S` in `[0, usize::MAX]`, a file written by `save_adamw_state`
with step `S` shall have a `"step"` key whose `to_vec1::<u64>()` returns `[S as u64]`.

**Validates: Requirements 1.3, 6.2**

---

### Property 10: moment update formula correctness

*For any* initial moments `(m1_0, m2_0)` and gradient `g`, after one `update_moments` call:

```
m1_expected = β1 · m1_0 + (1 - β1) · g
m2_expected = β2 · m2_0 + (1 - β2) · g²
```

The tensors in `moment_store` shall equal `(m1_expected, m2_expected)` element-wise within
`1e-6` tolerance.

**Validates: Requirements 3.3, 3.4**

---

## Implementation Sequence (Wave Structure)

### Wave 1 — Verify Candle internals + audit (no code changes)

1. Read `candle_nn/src/optim.rs` source (or Cargo cache) to confirm:
   - `AdamW.step()` is the only public method; no `moments()` accessor exists.
   - `candle_core::safetensors::save` accepts `U64` dtype.
   - `Tensor::id()` is stable and usable for VarMap key resolution.
2. Write `audit.md` in `.kiro/specs/gwen-223-adamw-optimizer-state-serialization/` with findings.
3. No `.rs` file changes in Wave 1.

### Wave 2 — In-memory bookkeeping

Touch points:
- `layered_training_loop.rs`:
  - Add `moment_store: HashMap<String, (Tensor, Tensor)>` and `step_t: usize` to struct.
  - Initialize both in `new()` from `initial_step`.
  - Add `update_moments()` method (private).
  - Call `update_moments()` in `run()` immediately after `self.adamw.step(&grads)`.
- `adamw_state.rs` (new file):
  - Implement `varmap_key_to_adamw_prefix` and `adamw_prefix_to_varmap_key`.
  - Implement `varmap_key_for` helper.
  - Declare `MomentStore` type alias.
- Tests: all Wave 2 unit tests listed in Testing Strategy.

### Wave 3 — Save AdamW state

Touch points:
- `adamw_state.rs`:
  - Implement `save_adamw_state` and `adamw_state_path`.
- `layered_training_loop.rs`:
  - Promote `save_checkpoint` free function to `save_checkpoint_and_adamw_state` method.
  - Update the `% 500` call site in `run()`.
- Tests: all Wave 3 unit tests.

### Wave 4 — Load AdamW state on resume

Touch points:
- `adamw_state.rs`:
  - Implement `load_adamw_state`.
- `layered_training_loop.rs`:
  - Add `load_adamw_state` method (the public-facing method that calls the module function and handles `Ok(None)` / `Err`).
- `native_runner.rs`:
  - Add `training_loop.load_adamw_state(&path)` call after `load_checkpoint`.
- Add shape-validation pass in `load_adamw_state` (drop entries where loaded shape ≠ VarMap shape).
- Tests: all Wave 4 unit tests + integration test.

### Wave 5 — Validation

- Implement `test_moment_values_match_adamw_internal` using captured gradients.
- Implement `test_loss_curve_no_step_back` using micro-GGUF fixture.
- Implement `test_no_regression_fresh_run`.
- Run full test suite: `cargo test -p gwen-core -- train`.

---

## Dependencies

No new external crates required.

| Dependency | Already present | Usage |
|---|---|---|
| `candle_core::safetensors` | Yes (GWEN-222) | `save`, `load` for `_adamw` file |
| `candle_core::Tensor` | Yes | moment tensors, step scalar |
| `std::collections::HashMap` | Yes | `MomentStore` |
| `anyhow` | Yes | error propagation |
| `quickcheck` | Yes (`dev-dependencies`) | property-based tests |
| `tempfile` | Yes (`dev-dependencies`) | test file I/O |

The only new Rust surface is `candle_core::safetensors::save` with a `U64`-dtype tensor.
This should be verified in Wave 1 (some versions of candle's safetensors writer only support
`F32`/`BF16`/`F16`; if `U64` is unsupported the `step` tensor can be stored as a `F32`
scalar and cast on load with a precision note in the audit).

