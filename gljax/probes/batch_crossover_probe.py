"""Where does tile streaming stop winning? (ARTX10 §9 question 3)

Decode measured at batch 1: tile streaming beat whole-weight dequant 3.58x.
The open question was whether prefill inverts that — a materialised weight is
built once and then reused by every token in the batch, so its cost amortises
while tile streaming pays its loop overhead per call regardless.

This sweeps M from 1 (decode) to 512 (a prefill chunk) on the same weight and
reports the crossover, if there is one. That crossover IS the cost model's
input: it is the batch size above which `Materialise` stops being a memory
trade and becomes strictly better.

Correctness is re-checked at every M — a tiling that is fast and wrong is P4.
"""
import sys, time, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import jax, jax.numpy as jnp
from jax import lax

rng = np.random.default_rng(0)
K, N, BLK = 896, 4864, 32
NB = K // BLK

W = jnp.asarray(rng.integers(-8, 8, (K, N)).astype(np.int8))
S = jnp.asarray(rng.random((NB, N), dtype=np.float32) + 0.5)
Wf_np = (np.asarray(W).astype(np.float32).reshape(NB, BLK, N)
         * np.asarray(S)[:, None, :]).reshape(K, N)
Wf = jnp.asarray(Wf_np)


def whole(a, w, s):
    return a @ (w.astype(jnp.float32).reshape(NB, BLK, N) * s[:, None, :]).reshape(K, N)


def red(tile_k):
    nt = K // tile_k
    bpt = tile_k // BLK
    def f(a, w, s):
        m = a.shape[0]
        wr = w.reshape(nt, tile_k, N)
        sr = s.reshape(nt, bpt, N)
        # [m, K] -> [nt, m, tile_k] so each step sees its own slice of the
        # contracting dim for every row of the batch.
        ar = a.reshape(m, nt, tile_k).transpose(1, 0, 2)
        def step(acc, xs):
            wt, st, at = xs
            wf = (wt.astype(jnp.float32).reshape(bpt, BLK, N) * st[:, None, :]).reshape(tile_k, N)
            return acc + at @ wf, None
        acc, _ = lax.scan(step, jnp.zeros((m, N), jnp.float32), (wr, sr, ar))
        return acc
    return f


def bench(fn, args, ref):
    c = jax.jit(fn).lower(*args).compile()
    got = np.asarray(c(*args))
    err = float(np.max(np.abs(got - ref)))
    for _ in range(2):
        c(*args).block_until_ready()
    best = 1e9
    reps = 20 if args[0].shape[0] < 128 else 6
    for _ in range(reps):
        t = time.perf_counter()
        c(*args).block_until_ready()
        best = min(best, time.perf_counter() - t)
    return best, c.memory_analysis().temp_size_in_bytes, err


print(f"weight [{K},{N}] block={BLK} | dense f32 {K*N*4/1e6:.2f} MB | "
      f"int8+scales {(K*N + NB*N*4)/1e6:.2f} MB\n")
print(f"{'M':>5} {'f32 us':>10} {'whole us':>10} {'tile32 us':>10} {'tile128 us':>11}   "
      f"{'best quant':>11}  {'vs f32':>7}  {'tile temp MB':>12}")
print("-" * 92)

for M in (1, 4, 16, 64, 256, 512):
    A = jnp.asarray(rng.standard_normal((M, K), dtype=np.float32))
    ref = np.asarray(A) @ Wf_np
    t_f32, _, _ = bench(lambda a, w: a @ w, (A, Wf), ref)
    t_who, tp_who, e_who = bench(whole, (A, W, S), ref)
    t_32, tp_32, e_32 = bench(red(32), (A, W, S), ref)
    t_128, tp_128, e_128 = bench(red(128), (A, W, S), ref)
    assert max(e_who, e_32, e_128) < 5e-2, f"numerics diverged at M={M}"
    best_q = min(t_who, t_32, t_128)
    which = {t_who: "whole", t_32: "tile32", t_128: "tile128"}[best_q]
    print(f"{M:>5} {t_f32*1e6:>10.1f} {t_who*1e6:>10.1f} {t_32*1e6:>10.1f} {t_128*1e6:>11.1f}   "
          f"{which:>11}  {best_q/t_f32:>6.2f}x  {tp_32/1e6:>12.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Recorded result — jaxlib 0.10.2 PJRT CPU plugin, 2026-07-28, i3-1115G4
# weight [896, 4864], block 32. Times in microseconds, best-of-N.
#
#     M     f32     whole   tile32   tile128   best quant   vs f32   tile temp
#     1    831.6   3557.4   1095.5    1085.6     tile128     1.31x     1.19 MB
#     4    956.1   3898.4   1551.2    1359.7     tile128     1.42x     1.26 MB
#    16   1341.2   4258.7   2428.9    2038.2     tile128     1.52x     1.54 MB
#    64   3611.8   6902.4   7891.7    6169.8     tile128     1.71x     2.65 MB
#   256  11573.9  15072.8  44932.2   21886.4     whole       1.30x     7.10 MB
#   512  23760.0  27223.5  84303.0   44079.3     whole       1.15x    13.03 MB
#
# ⭐⭐ THE RANKING INVERTS. Crossover between M=64 and M=256.
#
#   * M <= 64  (decode, small batches): tile streaming wins, up to 3.3x over
#     whole-weight dequant. The materialised weight cannot amortise.
#   * M >= 256 (prefill): whole-weight dequant wins, by up to 2x over tiling —
#     and at M=512 it costs only 1.15x an f32 baseline, because one 17.4 MB
#     materialisation is reused by 512 tokens.
#
# ⛔ Tile streaming degrades badly with M: 84303 us at M=512, i.e. 3.55x SLOWER
# than f32. Each scan step becomes a [M, tile_k] x [tile_k, N] dot, and the
# loop overhead plus poor blocking dominate. tile128 beats tile32 at every
# M > 1 — fewer, larger steps.
#
# ⭐ Consequence for ARTX10: `WeightDelivery` is a PER-PHASE choice, not a
# per-model one. Decode wants EmitTileStream(tile_k=128); prefill wants
# EmitBlockwise. Quantization is nearly free in prefill (1.15x) and a ~30% tax
# in decode.
#
# ⚠️ The crossover is a property of (weight shape, block, machine), not a
# constant. A cost model must measure it per deployment, which is what this
# probe does.
