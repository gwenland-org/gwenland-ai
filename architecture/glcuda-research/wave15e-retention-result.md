# glcuda Wave 15E - validating the Wave 15A retention as it ships

- notebook: wave15e-retention-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- patch: e66031c63277c27ec4470a90c1ca0b9b1a2555cc2bcf968d55cccac92cbac234
- model SHA-256: 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
- lib tests: test result: ok. 61 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.15s
- parity gate: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.50s
- repeats: 4, position-balanced, cold/warmup 5/5

## What is being validated

Qk4 is now the DEFAULT, so the candidate arm sets **no attention
environment at all** - that is the thing under test. The other two arms opt
out explicitly. Every arm is audited by the path the engine announces:

```
{
  "qk4_default": {
    "path": "qk4",
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "ntok": 244
  },
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
  }
}
```

## Sessions

| arm | P50 tok/s | median ms | MAD ms | max ms | decode | positions |
|---|---:|---:|---:|---:|---:|---|
| `qk4_default` | 8956.6 | 27.25 | 0.24 | 27.99 | 193.2 | [0, 2, 2, 0] |
| `rows` | 8346.1 | 29.24 | 0.23 | 29.85 | 193.1 | [1, 1, 0, 2] |
| `gqa7` | 8800.2 | 27.73 | 0.20 | 28.40 | 199.7 | [2, 0, 1, 1] |

## Paired comparisons, four repeats

| comparison | median delta | worst paired | all positive |
|---|---:|---:|---|
| qk4_default_vs_gqa7 | +1.78% | +1.50% | True |
| qk4_default_vs_rows | +7.32% | +6.83% | True |
| gqa7_vs_rows | +5.44% | +4.57% | True |

- tail, MEDIAN ACROSS SESSIONS: -1.4% (bar 5%)
- first A/B at two repeats measured +3.65% over gqa7 and +8.59% over rows
- **WEAK - +1.8%, positive but inside noise; the retention is not supported**

## Why the tail is measured this way

The first A/B rejected this change on P95 latency. Re-reading the raw ten
samples showed two sessions with slow LEADING iterations that settled
immediately after, and they landed on a different arm in each repeat - the
paired P95 delta was +37% in one and -16% in the other. P95 at n=10
interpolates to nearly the maximum, so one warm-up straggler owned the
verdict. This run raises cold and warmup to 5/5, reports median and MAD,
and judges the tail by the median across sessions.