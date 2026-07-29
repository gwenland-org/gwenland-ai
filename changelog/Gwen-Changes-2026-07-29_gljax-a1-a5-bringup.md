**Type:** Implementation — gljax ARTX01–05 bring-up, Waves A1–A5.
**Status:** ~10,200 lines, 188 tests green, clippy `-D warnings` clean.
⛔ **Nothing has executed.** The sprint gate ("one token out, matching glproc")
is not met and could not be met on this machine.

---

## Executive Summary

`gljax` went from a design series with zero code to a workspace member with a
PJRT FFI layer, a StableHLO emitter, a graph builder, an ops layer, a Qwen2
forward pass, a runtime and a checkpoint binder. Every one of those compiles,
is unit-tested, and — for everything that produces IR — is verified by a real
MLIR parser.

**None of the device code has ever run.** There is no PJRT plugin for Windows,
which is the development machine. A CI workflow now runs the plugin tests on
`ubuntu-latest` against a pinned plugin; the first push is when
`gljax::sys`'s hand-written 138-slot vtable finally gets exercised.

## The blocker, measured before any code was written

| Source | Result |
|---|---|
| `zml/pjrt-artifacts` latest (`manual-2026-07-28T12-29-00Z`) | 9 assets, all `linux`/`darwin`. **Zero Windows, zero `.dll`** |
| local `jaxlib` 0.10.2 (`jax_common.dll`, 235 MB) | loads via ctypes, exports **no `GetPjrtApi`** |
| WSL | 2.7.10.0 installed and healthy, **zero distros** — confirmed via `wsl --list --all`, `HKCU:\…\Lxss`, `Get-AppxPackage`, and a `*.vhdx` search |

⚠️ ARTX01 §1.4 and §5.1 both list the CPU plugin as available for Windows.
**That row is wrong.**

⚠️ This also corrects a premise recorded at the end of the previous session:
"jaxlib 0.10.2 is already confirmed as a working PJRT CPU plugin on this
machine". It is a working *JAX* backend; the ARTX10 probes never went through
the C API, which is why they worked and Rust cannot.

## ⭐ The technique that made the sprint worth doing anyway

`gljax/tools/verify_mlir.py` traces 12 modules and parses each through jaxlib's
MLIR + StableHLO dialect. Not a compiler — but a real parser and verifier,
which is the one thing the Rust tests cannot be.

**It caught two bugs that green structural tests had missed**, both hitting
every model gljax could trace:

1. **`array<i64: >`** — the empty dense-array attribute. MLIR spells it
   `array<i64>`; the colon form fails to lex. Emitted by *any* rank-0
   broadcast, i.e. every scalar constant in RMSNorm and softmax.
2. **`dense<1e-6>`** — MLIR's float token requires a decimal point in the
   mantissa, so `1e-6` lexes as integer `1` followed by garbage. That is the
   RMSNorm epsilon of every Llama-family model. Rust's `{:?}` produces exactly
   that string. So **every model gljax could trace emitted an unparseable
   module**, while the test asserting `dense<1e-6>` was present passed.

Both are now regression-tested. The generalisable lesson is the same shape as
the tokenizer's: assert against what a reference implementation *does*, not
against what you told the emitter to produce.

## ⛔ Ten places the ARTX docs are wrong, found by implementing them

Each is documented at the code that corrects it, and summarised in
`gljax/README.md`.

| Doc | Correction |
|---|---|
| ARTX03 §3 | **NeoX RoPE is the half-split `(i, i+D/2)`**, not adjacent `(2i, 2i+1)`. ARTX01 §7.2 says halves, ARTX03 says adjacent, the sprint brief says both in the same paragraph. ⭐ `glproc/src/runner.rs:161` (`RopeStyle::Neox`, validated on Qwen2.5-0.5B) settles it. |
| ARTX03 (all) | **Qwen2 has q/k/v biases.** Never mentioned; `glproc/src/loader.rs:638` loads all three. |
| ARTX02 §7 | `matmul` derives batch dims from `self.rank()`, so `[B,S,D] @ [D,F]` batches the weight over its contraction axis. ARTX02 §9's own expected output contradicts it. Fixed: batch count from the lower-rank operand. |
| ARTX02 §5 | `dot_general` passes mismatched operand dtypes through; StableHLO rejects that. Fixed: reconcile by **widening** — never narrow (P5). |
| ARTX02 §6 | `TraceCx::finish` does `Rc::try_unwrap`, which panics whenever a Tensor is alive — including the outputs being passed in. ARTX02 §9's own example would panic every time. Fixed: borrow. |
| ARTX02 §3 | `emit_reduce` omits the braces around the region. `({ … })` is required; `reduce` is inside RMSNorm *and* softmax. |
| ARTX01 | **`compile_options` is a serialized `CompileOptionsProto`** — never mentioned anywhere. Hand-encoded, 6 bytes. ⚠️ `num_replicas`/`num_partitions` must be explicit: proto3 omits defaults and absent parses back as `0`, not `1`. |
| ARTX01 §1.4/§5.1 | The plugin file is **`libpjrt_cpu.so`**, not `pjrt_c_api_cpu_plugin.so`, and it is **274 MB**, not ~80 MB. Verified by download. |
| ARTX01 §9.1 | "Refuse on any minor divergence" would reject every plugin that is not this exact build. Fixed: major must match; then check `struct_size` reaches past the last slot gljax calls (71 of 138). Also: §9.1 predicted API minor ~58; the header is at **0.114**. |
| sprint brief | `glcore::GllmCheckpoint` and `SafetensorsCheckpoint` **do not exist**. The real type is `glcore::format::SafetensorsFile`. |

Also: ARTX03 §4 states Qwen2-0.5B is `n_heads=16, n_kv_heads=8` (MHA); the
published config is `14/2` (GQA repeat 7), which `hidden 896 = 14 × 64`
confirms. The GQA expansion path is therefore on the critical path, not a
dormant branch.

## What is verified, and what is not

| Layer | Status |
|---|---|
| `stablehlo/` emitter, types, ops | ✅ unit-tested; 12 modules parse+verify |
| `graph/` shape inference, `TraceCx` | ✅ unit-tested |
| `ops/` rms_norm, rope, attention, ffn, softmax, embedding, kv_cache | ✅ structure; ⚠️ **numerics unexecuted** |
| `model/qwen2` full forward | ✅ traces + verifies at every bucket ≤ 1024 |
| `runtime/` digest, cache, plan, bucket, sample | ✅ against real files / published vectors |
| `checkpoint/` safetensors binding | ✅ with synthetic sources |
| `sys/` PJRT C API bindings | ⛔ **never called** |
| `pjrt/` plugin, client, buffer, execute | ⛔ **never called** |
| `runtime/session` | ⛔ **never constructed** |

⚠️ "Structure" means the graph has the right ops in the right order with the
right shapes. It does **not** mean the graph computes the right numbers. That
is ARTX12 Part B and needs a device.

## Dependency budget held at three (glcore, libloading, log)

* **SHA-256 hand-written** (`runtime/digest.rs`, ~90 lines) against the
  FIPS-180-4 vectors including the one-million-`a` case. `sha2` is already in
  the workspace lockfile via glictus-caliburni, so swapping to
  `glictus_caliburni::sha256_bytes` is a one-line change if a dep becomes
  acceptable.
* **`CompileOptionsProto` hand-encoded** rather than taking `prost`.
* **`.gllm` not wired** — needs glictus-caliburni. `checkpoint::WeightSource`
  is a trait, so adding it is a new impl rather than a rewrite.

## Known gaps at hand-off

* ⛔ **The 2048 bucket does not trace.** The causal mask is a dense O(S²)
  constant; 2048² = 4.2 M elements ≈ 34 MB of MLIR text, over the 1 Mi-element
  cap. It refuses with a message pointing at "pass it as a runtime weight".
  Buckets ≤ 1024 trace fine. ARTX03 calls a 512-wide mask "1 MB, acceptable for
  v1" without noting the quadratic.
* ⛔ **No KV cache in the model.** `ops/kv_cache.rs` has the
  `dynamic_slice`/`dynamic_update_slice` primitives, parse-verified, but
  `Session::generate` uses ARTX03 §4's **full-recomputation** path — O(n·S) for
  n tokens. Correct by construction, slow. Wiring the cache needs buffer
  donation (`input_output_alias`), which gljax does not set; without it every
  step copies the whole cache.
* `PJRT_Executable_Serialize` / `DeserializeAndLoad` are bound but not wrapped
  — the compile cache is written, never read back.
* MoE is `unimplemented!()` on purpose; sampling is argmax only (ARTX14 unbuilt).

## CI

`.github/workflows/gljax-pjrt.yml` runs the plugin tests on `ubuntu-latest`.

The plugin is **pinned by tag and checked by SHA-256** — ARTX01 §1.4 says to
treat plugins like a compiler dependency. Pinning the tag says which artifact
you meant; checking the digest says you got it.

⭐ **The workflow fails if any PJRT test SKIPs.** A green run that skipped them
would look like coverage and be worse than no run at all — those three tests
are the only thing that exercises the hand-written vtable.

`verify_mlir.py` runs there too: it needs no plugin, and a parse failure is a
far clearer signal than the compile error PJRT reports for the same module.
