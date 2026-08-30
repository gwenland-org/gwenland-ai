# glcuda Wave 15A production A/B - four independent QK chains per warp

- notebook: wave15a-production-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- baseline revision: 3bce8dd7b8aaa2765855ab927c611b54981f9241
- Wave 15A patch: 063886dfdcb163c6c55252d1bde98d1c16f8aba3f9a08afbe4fa7fbf1148b3e9
- model SHA-256: 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
- lib tests: test result: ok. 60 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.94s
- parity gate: test result: ok. 28 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.26s
- repeats: 2, orders [['rows', 'gqa7', 'qk4'], ['qk4', 'gqa7', 'rows']]

## What this A/B is and is not

All three arms are the **same binary**. They differ only by environment, so
there is no build variance, no patch difference, and each arm is audited by
the path the engine itself announces:

```
{
  "rows": {
    "path": "rows",
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "ntok": 244
  },
  "gqa7": {
    "path": "gqa7",
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "ntok": 244
  },
  "qk4": {
    "path": "qk4",
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "ntok": 244
  }
}
```

The screening compared qk4 against the **row** kernel and measured 1.35x.
Production ships **gqa7**, where seven query heads share one KV tile, so the
retention question is qk4 against gqa7 - a different and harder comparison.
Both are reported.

## Sessions

| arm | P50 tok/s | mean | P95 latency ms | decode P50 | positions |
|---|---:|---:|---:|---:|---|
| rows | 8641.7 | 8649.4 | 28.53 | 200.3 | [0, 2] |
| gqa7 | 9053.4 | 8943.4 | 29.47 | 205.0 | [1, 1] |
| qk4 | 9384.3 | 9083.0 | 31.95 | 200.2 | [2, 0] |

## Paired comparisons

| comparison | median delta | worst paired | all repeats positive |
|---|---:|---:|---|
| qk4_vs_gqa7 | +3.65% | +3.25% | True |
| qk4_vs_rows | +8.59% | +8.41% | True |
| gqa7_vs_rows | +4.76% | +4.18% | True |

- **REJECT - a gate failed (tail latency, decode, or oracle)**

## Does the isolated screening transfer?

- isolated, qk4 vs rows: **1.349x** on the attention kernel alone
- production, qk4 vs rows: **1.086x** end to end

Attention is 34.6% of prefill, so a 1.349x attention kernel projects to
about +9.8% end to end. Comparing that projection with the production
number above is the whole point of running this: two previous waves in this
sprint (row-tile, VNNI-512) measured large isolated wins that went flat in
production, and the only way to tell the difference is to run it.

## Caveats that are part of the result

- Two repeats with three arms cannot be a full Latin square. Position SUM is
  balanced (every arm totals 2), but `gqa7` never sits at an extreme
  position, so its two numbers carry slightly less drift protection than the
  other two.
- Session drift on this machine has reached 5-19% between sessions. Only the
  paired, same-session deltas above are evidence; the absolute tok/s column
  is context.