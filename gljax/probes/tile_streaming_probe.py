"""Does tile-streaming avoid the materialisation? (Solusi 1)

The measured problem: emitting `convert -> broadcast -> multiply -> dot` over a
whole [896, 4864] weight makes XLA build the full f32 matrix in scratch
(17.45 MB) before the dot reads it.

The proposal: never dequantise the whole weight. Slice it, dequantise one tile,
dot the tile, accumulate. If XLA honours that structure, peak scratch should
fall to roughly one tile instead of the whole matrix.

This measures `temp_size_in_bytes` from the compiled executable, which is the
same number that exposed the problem — so the comparison is like for like.
Correctness is checked against NumPy every time: a tiling that is fast and
wrong is the P4 bug class.
"""
import sys, numpy as np, functools
sys.stdout.reconfigure(encoding="utf-8")
import jax, jax.numpy as jnp
from jax import lax

rng = np.random.default_rng(0)
K, N, BLK = 896, 4864, 32
NB = K // BLK
DENSE = K * N * 4

A = jnp.asarray(rng.standard_normal((1, K), dtype=np.float32))
W = jnp.asarray(rng.integers(-8, 8, (K, N)).astype(np.int8))
S = jnp.asarray(rng.random((NB, N), dtype=np.float32) + 0.5)

REF = np.asarray(A) @ (np.asarray(W).astype(np.float32).reshape(NB, BLK, N)
                       * np.asarray(S)[:, None, :]).reshape(K, N)

rows = []


def measure(name, fn, args):
    try:
        c = jax.jit(fn).lower(*args).compile()
        got = np.asarray(c(*args))
        ma = c.memory_analysis()
        err = float(np.max(np.abs(got - REF)))
        rows.append((name, ma.temp_size_in_bytes, ma.argument_size_in_bytes, err, ""))
    except Exception as e:
        rows.append((name, -1, -1, float("nan"), str(e).replace("\n", " ")[:90]))


# ── baseline: whole-weight dequant (the measured problem) ───────────────────
def whole(a, w, s):
    return a @ (w.astype(jnp.float32).reshape(NB, BLK, N) * s[:, None, :]).reshape(K, N)
measure("baseline: whole-weight dequant", whole, (A, W, S))

# ── Solusi 1a: tile the OUTPUT dim, unrolled, concatenate ──────────────────
def out_tiles(tile_n):
    def f(a, w, s):
        outs = []
        for j in range(0, N, tile_n):
            wt = lax.slice(w, (0, j), (K, j + tile_n))
            st = lax.slice(s, (0, j), (NB, j + tile_n))
            wf = (wt.astype(jnp.float32).reshape(NB, BLK, tile_n)
                  * st[:, None, :]).reshape(K, tile_n)
            outs.append(a @ wf)
        return jnp.concatenate(outs, axis=1)
    return f
for t in (512, 128):
    measure(f"S1a: output tiles of {t}, unrolled", out_tiles(t), (A, W, S))

# ── Solusi 1b: tile the REDUCTION dim with lax.scan, accumulate ─────────────
def red_scan(tile_k):
    nt = K // tile_k
    bpt = tile_k // BLK
    def f(a, w, s):
        wr = w.reshape(nt, tile_k, N)
        sr = s.reshape(nt, bpt, N)
        ar = a.reshape(1, nt, tile_k).transpose(1, 0, 2)   # [nt, 1, tile_k]
        def step(acc, xs):
            wt, st, at = xs
            wf = (wt.astype(jnp.float32).reshape(bpt, BLK, N) * st[:, None, :]).reshape(tile_k, N)
            return acc + at @ wf, None
        acc, _ = lax.scan(step, jnp.zeros((1, N), jnp.float32), (wr, sr, ar))
        return acc
    return f
for t in (128, 32):
    measure(f"S1b: reduction tiles of {t}, lax.scan", red_scan(t), (A, W, S))

# ── Solusi 1c: output tiles via lax.scan (no unroll -> smaller program) ─────
def out_scan(tile_n):
    nt = N // tile_n
    def f(a, w, s):
        wr = w.reshape(K, nt, tile_n).transpose(1, 0, 2)       # [nt, K, tile_n]
        sr = s.reshape(NB, nt, tile_n).transpose(1, 0, 2)      # [nt, NB, tile_n]
        def step(carry, xs):
            wt, st = xs
            wf = (wt.astype(jnp.float32).reshape(NB, BLK, tile_n) * st[:, None, :]).reshape(K, tile_n)
            return carry, (a @ wf)[0]
        _, out = lax.scan(step, None, (wr, sr))
        return out.reshape(1, N)
    return f
for t in (512, 128):
    measure(f"S1c: output tiles of {t}, lax.scan", out_scan(t), (A, W, S))

print(f"weight dense f32 = {DENSE/1e6:.2f} MB | int8 = {K*N/1e6:.2f} MB | "
      f"scales = {NB*N*4/1e6:.2f} MB\n")
w = max(len(r[0]) for r in rows)
print(f"{'variant':<{w}}  {'temp MB':>9} {'vs dense':>9}  {'arg MB':>7}  {'max err':>9}  note")
print("-" * (w + 55))
for n, t, a, e, note in rows:
    if t < 0:
        print(f"{n:<{w}}  {'FAIL':>9} {'':>9}  {'':>7}  {'':>9}  {note}")
    else:
        print(f"{n:<{w}}  {t/1e6:>9.2f} {t/DENSE:>8.2f}x  {a/1e6:>7.2f}  {e:>9.1e}  {note}")

# ─────────────────────────────────────────────────────────────────────────────
# Recorded result — jaxlib 0.10.2 PJRT CPU plugin, 2026-07-28, i3-1115G4
# Decode-shaped matvec [1,896] x [896,4864], block=32, best-of-30.
#
#   variant                        time      vs f32   temp      argument
#   f32 weights (reference)      848.0 us     1.00x    0.02 MB   17.44 MB
#   quant, whole-weight dequant 3932.1 us     4.64x   17.45 MB    4.91 MB
#   quant, reduction tiles 128  1107.7 us     1.31x    2.51 MB    4.91 MB
#   quant, reduction tiles 32   1099.8 us     1.30x    1.19 MB    4.91 MB
#
# ⭐ Tile-streaming the REDUCTION dimension through `lax.scan` is 3.58x faster
# than dequantising the whole weight, and cuts scratch 17.45 -> 1.19 MB.
# Quantization goes from catastrophic (4.64x slower than f32) to a ~30% tax.
#
# ⭐⭐ The resulting trade is real and has a sign that depends on the model:
#       f32          17.46 MB working set,  848 us
#       tiled quant   6.10 MB working set, 1100 us
#     2.86x less memory for 1.30x the time. A win when the model does not fit,
#     a loss when it does. That is a COST MODEL decision, not a capability flag.
#
# ⛔ THE UNROLLED LOOP DOES NOT WORK. `S1a` (a Python for-loop emitting one dot
# per tile) measured 17.43 MB — XLA fuses it straight back into a whole-weight
# dequant. The loop must survive into the compiled program as a real
# `stablehlo.while`, which is what `lax.scan` produces. Any Rust emitter must
# emit a while-loop, not unrolled slices.
#
# ⛔ Output-dim tiling is much weaker than reduction-dim tiling (9.26 MB vs
# 1.19 MB): the accumulator stays [1, N] under reduction tiling, so only
# [tile_k, N] is ever live.
#
# ⚠️ Scope: batch 1 (decode shape), one weight shape, CPU plugin only. Prefill
# (batch >> 1) is NOT measured and may invert the conclusion, because a
# materialised weight amortises across tokens.
