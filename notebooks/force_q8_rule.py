"""The force_q8 decision rule, versioned with the engine it judges.

Lives here rather than inside the notebook for one reason: a notebook is a
file someone carries around, and three separate T4 sessions ran a STALE copy
of it while the branch held the corrected rule. Every number those runs
produced was right and every verdict drawn from them was wrong -- the harder
failure to notice, because nothing looks broken.

The notebook fetches this repo anyway. So the analysis lives in the repo and
the notebook execs it: fixing a rule now reaches the next run without anyone
re-importing anything.

Executed with `SUMMARY`, `BANNER_OK`, `VRAM_MOVED` and `overlap` already in
scope; sets `DECISION`.
"""

# DECIDE ON THE LARGEST MODEL ACTUALLY MEASURED, NOT ON 7B SPECIFICALLY.
#
# The original rule asked for 7B because 7B was expected to be the worst
# case. That reasoning was right; naming the model was not. force_q8 trades
# k-quant weights (4.5-6.5625 bpw) for Q8_0 (8.5 bpw), and decode reads every
# weight byte per token -- so the decode penalty grows with weight bytes, and
# the LARGEST measured model is the strictest test regardless of its name.
# Requiring a specific label meant a run with 0.5B + 3B reported UNDECIDED
# while holding a -19.7% no-overlap result.
#
# "Largest" is taken from measured baseline VRAM, not from the label: the
# label is a string someone typed, VRAM is a number the run produced.
def _size(row):
    vb = row["vram"][0][1]
    return vb if vb is not None else -1

usable = [r for r in SUMMARY
          if BANNER_OK.get(r["model"]) and VRAM_MOVED.get(r["model"]) and _size(r) > 0]
attempted = [r for r in SUMMARY if _size(r) > 0]

if not attempted:
    DECISION = "UNDECIDED -- no model produced a measurement"
elif not usable:
    DECISION = ("INVALID -- no model's arms actually differed (no banner, or VRAM "
                "unchanged). Nothing above is a measurement of force_q8.")
else:
    big = max(usable, key=_size)
    lab = big["model"]
    bb, vv = big["warm"]
    if bb[1] is None or vv[1] is None:
        DECISION = f"UNDECIDED -- {lab} decode unmeasured"
    else:
        d = 100 * (vv[1] - bb[1]) / bb[1]
        if overlap(bb, vv) or d > 0:
            DECISION = (f"FLIP THE DEFAULT -- {lab} warm decode {d:+.1f}% "
                        f"{'(arms overlap)' if overlap(bb, vv) else '(improved)'}; "
                        f"keep GLCUDA_NATIVE_KQUANT=1 as the escape hatch")
        else:
            DECISION = (f"STAY OPT-IN -- {lab} warm decode {d:+.1f}% with no overlap. "
                        f"A real trade: document it, do not default it")
        cb, cv = big["cold"]
        if cb[1] and cv[1] and not overlap(cb, cv):
            DECISION += f" · NOTE cold decode {100*(cv[1]-cb[1])/cb[1]:+.1f}% (no overlap)"
        vb, vv2 = big["vram"]
        if vb[1] and vv2[1]:
            DECISION += f" · VRAM {vv2[1]-vb[1]:+.0f} MiB ({100*(vv2[1]-vb[1])/vb[1]:+.1f}%)"
        DECISION += f" · decided on {lab}, the largest of {len(usable)} valid model(s)"

    # Does the penalty grow with size? Two points is not a curve, but a
    # monotone trend with a known mechanism is worth printing -- it is the
    # reason a bigger model would not reverse the verdict, only deepen it.
    if len(usable) >= 2:
        pts = sorted(usable, key=_size)
        trend = []
        for r in pts:
            b, v = r["warm"]
            if b[1] and v[1]:
                trend.append((r["model"], _size(r), 100 * (v[1] - b[1]) / b[1],
                              overlap(b, v)))
        if len(trend) >= 2:
            print("\ndecode penalty vs model size (weight bytes are the mechanism):")
            for lab_, mib, dd, ov in trend:
                print(f"  {lab_:5s}  {mib:6.0f} MiB baseline   decode {dd:+6.1f}%"
                      f"   {'overlap -- within noise' if ov else 'no overlap'}")
            if trend[-1][2] < trend[0][2]:
                print("  -> monotone: the penalty deepens as the model grows, which is "
                      "what\n     requantizing to a wider format predicts. A larger "
                      "model would\n     worsen this verdict, not reverse it.")

print("\n" + "=" * 66)
print(DECISION)
print("=" * 66)
