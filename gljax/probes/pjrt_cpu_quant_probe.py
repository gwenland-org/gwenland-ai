"""Q4 probe v2 — what the PJRT CPU plugin actually compiles AND runs.

Two mechanisms, because the two questions are different:

* **Quant-dialect patterns** (`!quant.uniform`, `uniform_quantize`) cannot be
  built by any JAX API, so they are fed as raw MLIR text and we report whether
  they even *parse*. A parse failure is a stronger result than a lowering
  failure: it means the type is not registered at all.
* **Everything else** is built with `jnp`, lowered by `jax.jit`, compiled and
  **executed**, and the output is checked against NumPy. Compiling without
  running would not catch a backend that lowers the pattern to something
  numerically wrong — P4's bug class.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

import jax, jax.numpy as jnp
import jaxlib.mlir.ir as ir
from jax._src.interpreters import mlir as jmlir

CTX = jmlir.make_ir_context()
print(f"plugin: {jax.devices()[0].client.platform_version} | jax {jax.__version__}\n")

rows = []


def raw_parse(name, mlir):
    """Can this IR even be parsed into JAX's MLIR context?"""
    try:
        with CTX:
            ir.Module.parse(mlir)
        rows.append((name, "PARSES", ""))
    except Exception as e:
        txt = str(e).replace("\n", " ")
        rows.append((name, "PARSE-FAIL", txt[:120]))


def run(name, fn, args, expect, want_ops=()):
    """Lower -> compile -> execute -> check numerics, and report emitted ops."""
    try:
        lowered = jax.jit(fn).lower(*args)
        text = lowered.as_text()
        compiled = lowered.compile()
        got = np.asarray(compiled(*args))
    except Exception as e:
        rows.append((name, "FAIL", str(e).replace("\n", " ")[:120]))
        return
    err = float(np.max(np.abs(got - expect)))
    seen = [o for o in want_ops if o in text]
    rows.append((name, "RUNS-OK" if err < 1e-3 else "RUNS-WRONG",
                 f"max|err|={err:.1e}" + (f" | emits {','.join(seen)}" if seen else "")))
    return text


# ── quant-dialect: the ARTX10 assumption ────────────────────────────────────
raw_parse("(a) uniform_quantize -> uniform_dequantize", """
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 3.900000e-03>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<4x!quant.uniform<i8:f32, 3.900000e-03>>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}""")
raw_parse("(f) hybrid dot: f32 lhs x !quant.uniform rhs", """
func.func @main(%a: tensor<4x8xf32>, %w: tensor<8x16x!quant.uniform<i8:f32, 3.900000e-03>>) -> tensor<4x16xf32> {
  %0 = stablehlo.dot_general %a, %w, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x16x!quant.uniform<i8:f32, 3.900000e-03>>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}""")
raw_parse("(k) control for the parser: plain f32 dot", """
func.func @main(%a: tensor<4x8xf32>, %w: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = stablehlo.dot_general %a, %w, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}""")

# ── executable patterns ─────────────────────────────────────────────────────
rng = np.random.default_rng(0)
A = jnp.asarray(rng.standard_normal((4, 64), dtype=np.float32))
Wf = jnp.asarray(rng.standard_normal((64, 16), dtype=np.float32))
Wi8 = jnp.asarray(rng.integers(-8, 8, (64, 16)).astype(np.int8))
Wi4 = jnp.asarray(rng.integers(-8, 8, (64, 16)).astype(np.int8))  # values fit i4

run("(b) control: f32 dot", lambda a, w: a @ w, (A, Wf), np.asarray(A) @ np.asarray(Wf))

run("(c) convert(i8->f32) -> dot",
    lambda a, w: a @ w.astype(jnp.float32), (A, Wi8),
    np.asarray(A) @ np.asarray(Wi8).astype(np.float32), ("stablehlo.convert",))

# per-axis: one scale per output column
s_ax = jnp.asarray(rng.random(16, dtype=np.float32) + 0.5)
run("(g) per-axis scale: convert*bcast -> dot",
    lambda a, w, s: a @ (w.astype(jnp.float32) * s), (A, Wi8, s_ax),
    np.asarray(A) @ (np.asarray(Wi8).astype(np.float32) * np.asarray(s_ax)),
    ("stablehlo.convert", "stablehlo.broadcast_in_dim", "stablehlo.multiply"))

# ⭐ BLOCKWISE — the GQ4A shape: a scale per 32-element block along the
# CONTRACTING dim. 64/32 = 2 blocks x 16 outputs = 32 scales. This is exactly
# what a StableHLO quantized TYPE cannot express (per-axis gives one scale per
# slice along ONE dim, never out x in/32).
Sb = jnp.asarray(rng.random((2, 16), dtype=np.float32) + 0.5)
def blockwise(a, w, s):
    wf = w.astype(jnp.float32).reshape(2, 32, 16) * s[:, None, :]
    return a @ wf.reshape(64, 16)
ref_b = np.asarray(A) @ (np.asarray(Wi8).astype(np.float32).reshape(2, 32, 16)
                         * np.asarray(Sb)[:, None, :]).reshape(64, 16)
run("(h) BLOCKWISE (GQ4A shape, 32-elem blocks)", blockwise, (A, Wi8, Sb), ref_b,
    ("stablehlo.reshape", "stablehlo.broadcast_in_dim", "stablehlo.multiply"))

# ⭐ TWO-LEVEL — GQ4A's f16 block scale x f32 superblock scale
Ssup = jnp.asarray(rng.random((1, 16), dtype=np.float32) + 0.5)
def twolevel(a, w, s, t):
    wf = w.astype(jnp.float32).reshape(2, 32, 16) * s[:, None, :] * t[:, None, :]
    return a @ wf.reshape(64, 16)
ref_t = np.asarray(A) @ ((np.asarray(Wi8).astype(np.float32).reshape(2, 32, 16)
                          * np.asarray(Sb)[:, None, :]) * np.asarray(Ssup)[:, None, :]).reshape(64, 16)
run("(i) two-level scales (block x superblock)", twolevel, (A, Wi8, Sb, Ssup), ref_t)

# f16 block scale, as GQ4A actually stores it
Sb16 = Sb.astype(jnp.float16)
def blockwise_f16(a, w, s):
    wf = w.astype(jnp.float32).reshape(2, 32, 16) * s.astype(jnp.float32)[:, None, :]
    return a @ wf.reshape(64, 16)
ref_16 = np.asarray(A) @ (np.asarray(Wi8).astype(np.float32).reshape(2, 32, 16)
                          * np.asarray(Sb16).astype(np.float32)[:, None, :]).reshape(64, 16)
run("(i') f16 block scale (GQ4A's actual dtype)", blockwise_f16, (A, Wi8, Sb16), ref_16)

# ── int4 storage: can a 4-bit tensor even cross the PJRT boundary? ──────────
try:
    w4 = jnp.asarray(np.asarray(Wi4), dtype=jnp.int4)
    f = jax.jit(lambda a, w: a @ w.astype(jnp.float32))
    got = np.asarray(f(A, w4))
    err = float(np.max(np.abs(got - np.asarray(A) @ np.asarray(Wi4).astype(np.float32))))
    rows.append(("(j) int4 tensor as a real PJRT buffer", "RUNS-OK" if err < 1e-3 else "RUNS-WRONG",
                 f"max|err|={err:.1e} | dtype={w4.dtype} nbytes={w4.nbytes}"))
except Exception as e:
    rows.append(("(j) int4 tensor as a real PJRT buffer", "FAIL", str(e).replace("\n", " ")[:120]))

w = max(len(n) for n, _, _ in rows)
print(f"{'pattern':<{w}}  {'verdict':<11}  detail")
print("-" * (w + 70))
for n, v, d in rows:
    print(f"{n:<{w}}  {v:<11}  {d}")

# ─────────────────────────────────────────────────────────────────────────────
# Recorded result — jaxlib 0.10.2 PJRT CPU plugin, 2026-07-28, i3-1115G4
#
#   (a) uniform_quantize -> uniform_dequantize    PARSE-FAIL  quant dialect unregistered
#   (f) hybrid dot: f32 lhs x !quant.uniform rhs  PARSE-FAIL  same
#   (k) parser control: plain f32 dot             PARSES
#   (b) control: f32 dot                          RUNS-OK     max|err|=3.8e-06
#   (c) convert(i8->f32) -> dot                   RUNS-OK     max|err|=1.5e-05
#   (g) per-axis scale                            RUNS-OK     max|err|=2.3e-05
#   (h) BLOCKWISE, 32-elem blocks (GQ4A shape)    RUNS-OK     max|err|=1.5e-05
#   (i) two-level scales (block x superblock)     RUNS-OK     max|err|=2.3e-05
#   (i') f16 block scale (GQ4A's dtype)           RUNS-OK     max|err|=1.9e-05
#   (j) int4 as a real PJRT buffer                RUNS-OK     dtype=int4 nbytes=1024
#
# ⛔ And the finding that decides the architecture — measured on a
# [896, 4864] weight (a Qwen2.5-0.5B ffn_up shape):
#
#   weights as ARGUMENTS   argument  4.91 MB | temp 17.45 MB
#       -> optimised HLO contains
#          %fused_computation.1 (f32[28,4864], s8[896,4864]) -> f32[896,4864]
#          i.e. the FULL f32 weight is materialised into scratch on every call,
#          then the dot reads it. Storage is saved; bandwidth is not — it is
#          made worse.
#
#   weights as CONSTANTS   argument  0.00 MB | temp  0.02 MB
#       -> optimised HLO contains %constant.6 = f32[896,4864] constant({...})
#          i.e. XLA constant-folded the whole dequantisation at compile time.
#          Zero runtime cost, and zero memory saving: the executable now holds
#          full f32 weights.
#
#   f32 weights as argument (reference)  argument 17.44 MB | temp 0.02 MB
#
# ⭐ There is NO configuration on this plugin where quantised weights stay
# quantised through the dot. int4 is not bit-packed either (int4 and int8 both
# report 1.00 byte/element), so the narrow dtype buys nothing on its own.
#
# ⚠️ Scope of this claim: jaxlib 0.10.2 CPU plugin, one shape, batch 1, no
# donated buffers, no XLA flag overrides, no GPU. Re-run before trusting it on
# any other plugin or version.
