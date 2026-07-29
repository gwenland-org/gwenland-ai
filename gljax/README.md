# gljax — pure-Rust XLA/PJRT client

gljax emits StableHLO MLIR text and hands it to a dynamically loaded PJRT
plugin. It owns no kernels. Design series: [`architecture/`](architecture/),
starting at [`Overall-Architecture.md`](architecture/Overall-Architecture.md).

**Status:** ARTX01–05 bring-up, Waves A1–A5. Everything that can be verified
without a device is verified; nothing has ever executed. See below.

---

## ⛔ There is no PJRT plugin for Windows

Measured 2026-07-29 on the reference machine (Windows 11, i3-1115G4):

| Source | Result |
|---|---|
| `zml/pjrt-artifacts` latest release (`manual-2026-07-28T12-29-00Z`) | 9 assets: `{cpu,cuda,oneapi,rocm}` × `{linux,darwin}` × `{amd64,arm64}`. **Zero Windows assets, zero `.dll`.** |
| `jaxlib` 0.10.2 installed locally (`jax_common.dll`, 235 MB) | Loads, but exports **no `GetPjrtApi`**. jaxlib's Windows build links the CPU backend into the extension module; it is not a PJRT C API plugin. |
| WSL2 | Component enabled, default version 2, **no distribution installed**. |

⚠️ This contradicts ARTX01 §1.4 and §5.1, both of which list the CPU plugin as
available for "Linux, macOS, Windows". That row is wrong as of today's release.

Building a Windows plugin from XLA source needs Bazel + MSVC against the full
XLA tree — not a realistic path from here.

Checked again 2026-07-29: WSL itself **is** installed and current
(`WSL 2.7.10.0`, kernel 6.18.33.2-2, WSLg) but has **no distribution** —
confirmed four ways: `wsl --list --all`, the `HKCU:\…\Lxss` registry, `Get-AppxPackage`,
and a `*.vhdx` search all come back empty. The `bash.exe` on PATH is Git Bash.

### What this means

The **sprint gate — "one token out, matching glproc" — is unreachable on this
machine.** `Session` has never been constructed and `generate()` has never run.

**The chosen path is CI** (2026-07-29):
[`.github/workflows/gljax-pjrt.yml`](../.github/workflows/gljax-pjrt.yml) runs
the plugin tests on `ubuntu-latest` against a pinned plugin. Windows stays the
editing environment. This is ARTX01 §5.5's shape.

The alternative, still open: install a WSL distro
(`wsl --install Ubuntu-24.04 --location D:\wsl\ubuntu` — **to D:**, since C: has
only ~10 GB free and a Linux `target/` plus the 274 MB plugin will not fit).

---

## Verifying what gljax emits, without a plugin

`jaxlib` ships MLIR and the StableHLO dialect. That is not a compiler, but it
*is* a real parser and verifier — the one thing the Rust tests cannot be:

```bash
python gljax/tools/verify_mlir.py           # traces and verifies 12 modules
cargo run -p gljax --example dump_mlir -- block bf16 > module.mlir
python gljax/tools/verify_mlir.py module.mlir
```

⭐ **This caught two real bugs that green structural tests had missed:**

- `array<i64: >` — the empty dense-array attribute. MLIR spells it `array<i64>`;
  the colon form fails to lex. Broadcasting any rank-0 tensor emitted it, which
  is every scalar constant in RMSNorm and softmax.
- `dense<1e-6>` — MLIR's float token requires a decimal point in the mantissa,
  so `1e-6` lexes as an integer followed by garbage. That is the RMSNorm epsilon
  of every Llama-family model, so **every model gljax could trace emitted an
  unparseable module** while the tests asserting `dense<1e-6>` passed.

Both are now regression-tested. The lesson is P2's, exactly: assert on what a
real implementation does, not on what the emitter was told to produce.

### Running the Rust tests

```bash
cargo test -p gljax          # 188 tests
```

Host-only tests always run. The three PJRT tests SKIP loudly when no plugin is
configured — **a SKIP is not a pass.** Use `-- --nocapture` to see them. CI
fails the build if any of them skips, precisely so a green run cannot be
mistaken for coverage.

To run them locally on Linux:

```bash
TAG=manual-2026-07-28T12-29-00Z
curl -sSfL -O "https://github.com/zml/pjrt-artifacts/releases/download/$TAG/pjrt-cpu_linux-amd64.tar.gz"
echo "a00125365f1fb04c164d4dc63941e7b002d29f376be0bdc45557cc62efd19d60  pjrt-cpu_linux-amd64.tar.gz" | sha256sum -c -
mkdir -p ~/pjrt && tar -xzf pjrt-cpu_linux-amd64.tar.gz -C ~/pjrt

export PJRT_PLUGIN_CPU=~/pjrt/libpjrt_cpu.so   # ARTX01 §5.4
export PJRT_CPU_PLUGIN_PATH=~/pjrt/...         # sprint-brief alias, also honoured
cargo test -p gljax -- --nocapture
```

⚠️ The file is **`libpjrt_cpu.so`**. ARTX01 §1.4 and §5.1 call it
`pjrt_c_api_cpu_plugin.so`; zml does not ship that name. ARTX01's "~80 MB" is
also stale — 64 MB compressed, **274 MB** extracted. Verified by download
2026-07-29: ELF 64-bit LSB, exports `GetPjrtApi`, tarball contains exactly one
file.

`GLJAX_DUMP_MLIR=1` logs every module before compiling — the first thing to
reach for when PJRT rejects a program, since the compile call *is* the
validation step (ARTX01 §2.4).

---

## What is verified, and what is not

| Layer | Status |
|---|---|
| `stablehlo/` emitter, types, ops | ✅ unit-tested; 12 modules parse+verify through jaxlib's MLIR |
| `graph/` shape inference, `TraceCx` | ✅ unit-tested |
| `ops/` rms_norm, rope, attention, ffn, softmax, embedding | ✅ structure tested; ⚠️ **numerics unexecuted** |
| `model/qwen2` full forward | ✅ traces + verifies at every bucket ≤ 1024 |
| `runtime/` digest, cache, plan, bucket, sample | ✅ tested against real files / published vectors |
| `checkpoint/` safetensors binding | ✅ tested with synthetic sources |
| `sys/` PJRT C API bindings | ⛔ **never called** |
| `pjrt/` plugin, client, buffer, execute | ⛔ **never called** |
| `runtime/session` | ⛔ **never constructed** |

⚠️ "Structure tested" means the graph has the right ops in the right order with
the right shapes. It does **not** mean the graph computes the right numbers.
That is ARTX12 Part B and needs a device.

---

## Layout

```
src/
  sys/          raw PJRT C API bindings — hand-written, no bindgen, no build.rs
  pjrt/         safe wrappers: plugin → client → {executable, buffer, device}
  stablehlo/    MLIR text emission: emitter, types, ops
  graph/        SsaValue, FuncBuilder (shape inference), TraceCx (scope stack)
  tensor/       the public tracing handle
  precision/    PrecisionPolicy + thread-local scope
  ops/          rms_norm, rope, attention, ffn, softmax, embedding, kv_cache
  model/        Qwen2 forward pass
  runtime/      digest, compile cache, plan, buckets, sampling, Session
  checkpoint/   safetensors → traced signature binding
tools/
  verify_mlir.py   parse gljax output through jaxlib's MLIR
```

Dependencies: `glcore`, `libloading`, `log`. Adding a fourth is a wave-gate
decision (ARTX01 §5.4) — which is why SHA-256 is hand-written in
`runtime/digest.rs` and the `CompileOptionsProto` is hand-encoded in
`pjrt/compile.rs`.

---

## Corrections to the design series

Points where the implementation deviates because the spec was wrong or
self-contradictory. Each is documented at the code that makes the choice.

| Where | The correction |
|---|---|
| `ops/rope.rs` | NeoX pairs `(i, i+D/2)` — the **half-split**. ARTX03 §3 says adjacent `(2i, 2i+1)`; ARTX01 §7.2 says halves; the brief says both. `glproc/src/runner.rs:161`, validated on Qwen2.5-0.5B, settles it. |
| `model/qwen2.rs` | Qwen2 has **q/k/v biases**. ARTX03 never mentions them; glproc loads all three. |
| `graph/builder.rs` | `matmul` takes batch dims from the **lower-rank** operand. ARTX02 §7's version batches a rank-2 weight over its contraction axis; ARTX02 §9's own expected output disagrees with it. |
| `graph/builder.rs` | `dot_general` reconciles operand dtypes by **widening**. ARTX02 emits mismatched operand types, which the verifier rejects. |
| `graph/{builder,trace}.rs` | `finish` borrows instead of `Rc::try_unwrap`. ARTX02 §6's version panics whenever an output tensor is alive — i.e. always. |
| `stablehlo/ops.rs` | `reduce` wraps its region in `({ … })`. ARTX02 §3 omits the braces; the module does not parse. |
| `pjrt/compile.rs` | `compile_options` is a serialized `CompileOptionsProto`. ARTX01 never mentions it. |
| `checkpoint/` | `glcore::GllmCheckpoint` and `SafetensorsCheckpoint` do not exist; the real type is `glcore::format::SafetensorsFile`. |
| `pjrt/plugin.rs` | Version check is `struct_size` reachability, not ARTX01 §9.1's exact-minor match, which would reject every plugin that is not this build. |
| `.github/workflows/gljax-pjrt.yml` | The plugin file is `libpjrt_cpu.so`, not ARTX01 §1.4/§5.1's `pjrt_c_api_cpu_plugin.so`, and it is 274 MB, not ~80 MB. |
