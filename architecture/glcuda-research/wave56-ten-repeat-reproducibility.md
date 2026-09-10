# glcuda Wave 56 - ten-repeat reproducibility gate

- notebook: wave56-repro10-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- Wave 50 patch: 915ea7ede79038970018ae10d2c7851d0e01e64f3a73d38e1c0eef764b6dc6c9
- model SHA-256: 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
- lib tests: test result: ok. 66 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.76s
- full CUDA parity: test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.83s
- repeats: 10; 30 sessions; near-balanced positions {'pre_n16': [3, 3, 4], 'n16_current': [3, 4, 3], 'n16_regq': [4, 3, 3]}; cold/warmup/measure 5/5/10

## Compile, resource, and direct gate

```json
{
  "ptxas": {
    "mma4_retained": {
      "entry": "gl_attn_mma4_fused_f32",
      "found": true,
      "registers": 48,
      "static_shared_bytes": 4096,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    },
    "mma4_regq": {
      "entry": "gl_attn_mma4_regq_fused_f32",
      "found": true,
      "registers": 64,
      "static_shared_bytes": 0,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    },
    "n16_bstage": {
      "entry": "gl_gemm_mma_q8_bstage_n16",
      "found": true,
      "registers": 72,
      "static_shared_bytes": 9728,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    },
    "n16_m32": {
      "entry": "gl_gemm_mma_q8_bstage_n16_m32",
      "found": true,
      "registers": 64,
      "static_shared_bytes": 9728,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    }
  },
  "driver": {
    "retained": {
      "entry": "gl_attn_mma4_fused_f32",
      "found": true,
      "registers": 48,
      "static_shared_bytes": 4096,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    },
    "candidate": {
      "entry": "gl_attn_mma4_regq_fused_f32",
      "found": true,
      "registers": 64,
      "static_shared_bytes": 0,
      "spill_store_bytes": 0,
      "spill_load_bytes": 0,
      "stack_frame_bytes": 0
    },
    "retained_blocks_per_sm": 3,
    "candidate_blocks_per_sm": 4
  },
  "direct": {
    "ntok": 244,
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "warmup": 20,
    "iters": 200,
    "rounds": 7,
    "retained_us": 190.554,
    "candidate_us": 147.14,
    "speedup": 1.2951,
    "bit_exact": true,
    "retained_dynamic_shared": 15616,
    "candidate_dynamic_shared": 15616,
    "retained_blocks_per_sm": 3,
    "candidate_blocks_per_sm": 4
  }
}
```

## Production sessions

| arm | P50 tok/s | P90 tok/s | P99 tok/s | latency P50/P90/P99 ms | MAD ms | max ms | cold P50/P90 ms | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pre_n16` | 9997.3 | 10099.3 | 10140.7 | 24.407/24.630/24.663 | 0.127 | 24.667 | 48.014/48.139 | 202.3 |
| `n16_current` | 10635.8 | 10714.4 | 10783.6 | 22.942/23.076/23.183 | 0.097 | 23.202 | 44.973/45.030 | 202.9 |
| `n16_regq` | 11173.1 | 11244.8 | 11263.7 | 21.838/22.002/22.113 | 0.115 | 22.133 | 41.944/42.022 | 201.8 |

## Same-session decisions

- N16 only vs pre-N16: +6.39%, +638.5 tok/s
- regQ incremental: +5.05%, +537.3 tok/s
- cumulative final vs pre-N16: +11.76%, +1175.8 tok/s
- incremental paired deltas: [3.653, 4.959, 5.101, 5.381, 4.919, 5.185, 4.941, 5.273, 4.659, 5.262]
- cumulative paired absolute gains: [1018.8, 1394.4, 1194.7, 1231.5, 1129.3, 1222.3, 1128.2, 1160.7, 1141.2, 1205.0]
- incremental all positive: True
- cumulative all positive: True
- incremental median session-max latency delta: -4.61% (bar +5%)
- exact token oracle: True

**RETAIN - register-Q is all-positive and the same-session cumulative gain is +1175.8 tok/s**

## Dispatch audit

```json
{
  "pre_n16": {
    "contract": {
      "exact_fusion": true,
      "gqa_group": false,
      "grid2d": true,
      "r256": false,
      "ntile128": true,
      "bstage": true,
      "gemm_k128": false,
      "gemm_mma2": false,
      "gemm_n16": false,
      "attn_rows_forced": false,
      "gqa7_chains": 1,
      "attn_mma4": true,
      "attn_mma4_regq": false,
      "fused_swiglu": false
    },
    "attention": {
      "path": "mma4-fused",
      "heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "ntok": 244
    },
    "gemm": []
  },
  "n16_current": {
    "contract": {
      "exact_fusion": true,
      "gqa_group": false,
      "grid2d": true,
      "r256": false,
      "ntile128": true,
      "bstage": true,
      "gemm_k128": false,
      "gemm_mma2": false,
      "gemm_n16": true,
      "attn_rows_forced": false,
      "gqa7_chains": 1,
      "attn_mma4": true,
      "attn_mma4_regq": false,
      "fused_swiglu": false
    },
    "attention": {
      "path": "mma4-fused",
      "heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "ntok": 244
    },
    "gemm": [
      {
        "path": "bstage-n16-m32",
        "out_dim": 1152,
        "in_dim": 896,
        "ntok": 244
      },
      {
        "path": "bstage-n16",
        "out_dim": 4864,
        "in_dim": 896,
        "ntok": 244
      }
    ]
  },
  "n16_regq": {
    "contract": {
      "exact_fusion": true,
      "gqa_group": false,
      "grid2d": true,
      "r256": false,
      "ntile128": true,
      "bstage": true,
      "gemm_k128": false,
      "gemm_mma2": false,
      "gemm_n16": true,
      "attn_rows_forced": false,
      "gqa7_chains": 1,
      "attn_mma4": true,
      "attn_mma4_regq": true,
      "fused_swiglu": false
    },
    "attention": {
      "path": "mma4-regq",
      "heads": 14,
      "kv_heads": 2,
      "head_dim": 64,
      "ntok": 244
    },
    "gemm": [
      {
        "path": "bstage-n16-m32",
        "out_dim": 1152,
        "in_dim": 896,
        "ntok": 244
      },
      {
        "path": "bstage-n16",
        "out_dim": 4864,
        "in_dim": 896,
        "ntok": 244
      }
    ]
  }
}
```

## Raw sessions

```json
[
  {
    "repeat": 0,
    "position": 0,
    "arm": "pre_n16",
    "prefill_p50": 10496.593787848302,
    "prefill_p90": 10622.023983159022,
    "prefill_p99": 10622.06518381178,
    "latency_p50_ms": 23.245644,
    "latency_p90_ms": 23.4080331,
    "latency_p99_ms": 23.471991510000002,
    "latency_mad_ms": 0.1274720000000027,
    "latency_max_ms": 23.479098,
    "decode_p50": 207.56932715246904,
    "cold_prefill_p50_ms": 48.056034,
    "cold_prefill_p90_ms": 48.144340400000004,
    "cold_prefill_max_ms": 48.170596,
    "cold_decode_p50_ms": 6.862061,
    "samples_ms": [
      23.4,
      23.232,
      22.971,
      23.317,
      22.971,
      23.346,
      23.479,
      23.001,
      23.213,
      23.259
    ],
    "cold_samples_ms": [
      48.171,
      48.056,
      48.031,
      48.026,
      48.105
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 0,
    "position": 1,
    "arm": "n16_current",
    "prefill_p50": 11109.587426423073,
    "prefill_p90": 11145.046774106559,
    "prefill_p99": 11235.0301815497,
    "latency_p50_ms": 21.963066499999996,
    "latency_p90_ms": 22.1195886,
    "latency_p99_ms": 22.17281046,
    "latency_mad_ms": 0.06677249999999901,
    "latency_max_ms": 22.178724,
    "decode_p50": 208.26294389287858,
    "cold_prefill_p50_ms": 44.992715000000004,
    "cold_prefill_p90_ms": 45.0727136,
    "cold_prefill_max_ms": 45.117659999999994,
    "cold_decode_p50_ms": 6.874459,
    "samples_ms": [
      21.927,
      21.918,
      22.049,
      22.113,
      22.088,
      22.179,
      21.998,
      21.915,
      21.928,
      21.698
    ],
    "cold_samples_ms": [
      45.118,
      44.959,
      45.005,
      44.916,
      44.993
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 0,
    "position": 2,
    "arm": "n16_regq",
    "prefill_p50": 11515.431230507293,
    "prefill_p90": 11644.09452563673,
    "prefill_p99": 11671.622683154705,
    "latency_p50_ms": 21.1889685,
    "latency_p90_ms": 21.498723099999996,
    "latency_p99_ms": 21.93705541,
    "latency_mad_ms": 0.21491900000000186,
    "latency_max_ms": 21.985759,
    "decode_p50": 206.144877266976,
    "cold_prefill_p50_ms": 41.929868,
    "cold_prefill_p90_ms": 41.99189919999999,
    "cold_prefill_max_ms": 41.997265999999996,
    "cold_decode_p50_ms": 6.862767,
    "samples_ms": [
      21.986,
      21.109,
      21.22,
      20.961,
      21.203,
      21.175,
      20.98,
      20.9,
      21.445,
      21.41
    ],
    "cold_samples_ms": [
      41.997,
      41.984,
      41.93,
      31.835,
      23.834
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 1,
    "position": 0,
    "arm": "n16_current",
    "prefill_p50": 11054.89151815786,
    "prefill_p90": 11151.507619074939,
    "prefill_p99": 11164.449528154719,
    "latency_p50_ms": 22.07168,
    "latency_p90_ms": 22.2741565,
    "latency_p99_ms": 22.32921625,
    "latency_mad_ms": 0.13793199999999928,
    "latency_max_ms": 22.335334,
    "decode_p50": 207.90665711779758,
    "cold_prefill_p50_ms": 44.959171999999995,
    "cold_prefill_p90_ms": 45.053919199999996,
    "cold_prefill_max_ms": 45.082566,
    "cold_decode_p50_ms": 6.871913,
    "samples_ms": [
      22.335,
      22.094,
      22.023,
      21.926,
      21.884,
      22.267,
      22.202,
      21.852,
      22.064,
      22.079
    ],
    "cold_samples_ms": [
      45.011,
      45.083,
      44.959,
      36.338,
      22.662
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 1,
    "position": 1,
    "arm": "n16_regq",
    "prefill_p50": 11603.111880500925,
    "prefill_p90": 11727.516858655896,
    "prefill_p99": 11809.248159547333,
    "latency_p50_ms": 21.029002500000004,
    "latency_p90_ms": 21.470566599999998,
    "latency_p99_ms": 21.61130086,
    "latency_mad_ms": 0.22357349999999876,
    "latency_max_ms": 21.626938,
    "decode_p50": 207.048578696456,
    "cold_prefill_p50_ms": 41.879445000000004,
    "cold_prefill_p90_ms": 41.9895678,
    "cold_prefill_max_ms": 42.027975000000005,
    "cold_decode_p50_ms": 6.850786,
    "samples_ms": [
      20.911,
      20.84,
      20.824,
      20.646,
      20.971,
      21.627,
      21.271,
      21.087,
      21.453,
      21.329
    ],
    "cold_samples_ms": [
      41.932,
      41.879,
      42.028,
      26.243,
      25.9
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 1,
    "position": 2,
    "arm": "pre_n16",
    "prefill_p50": 10208.708240081942,
    "prefill_p90": 10303.789069381486,
    "prefill_p99": 10308.279012260904,
    "latency_p50_ms": 23.901305999999998,
    "latency_p90_ms": 24.0346338,
    "latency_p99_ms": 24.07009398,
    "latency_mad_ms": 0.1257310000000018,
    "latency_max_ms": 24.074034,
    "decode_p50": 204.17451304595795,
    "cold_prefill_p50_ms": 47.993291,
    "cold_prefill_p90_ms": 48.0723224,
    "cold_prefill_max_ms": 48.074895999999995,
    "cold_decode_p50_ms": 6.872548,
    "samples_ms": [
      23.682,
      23.761,
      23.96,
      23.843,
      24.001,
      23.669,
      24.03,
      23.779,
      23.978,
      24.074
    ],
    "cold_samples_ms": [
      48.075,
      47.993,
      48.068,
      33.319,
      24.147
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 2,
    "position": 0,
    "arm": "n16_regq",
    "prefill_p50": 11422.423843688895,
    "prefill_p90": 11500.847547544188,
    "prefill_p99": 11518.044887398211,
    "latency_p50_ms": 21.361501500000003,
    "latency_p90_ms": 21.840741299999998,
    "latency_p99_ms": 22.10847303,
    "latency_mad_ms": 0.12928100000000065,
    "latency_max_ms": 22.138220999999998,
    "decode_p50": 205.8640471602224,
    "cold_prefill_p50_ms": 41.936124,
    "cold_prefill_p90_ms": 42.008453,
    "cold_prefill_max_ms": 42.052483,
    "cold_decode_p50_ms": 6.849,
    "samples_ms": [
      22.138,
      21.22,
      21.181,
      21.637,
      21.346,
      21.26,
      21.377,
      21.403,
      21.245,
      21.808
    ],
    "cold_samples_ms": [
      42.052,
      41.942,
      41.936,
      29.246,
      24.32
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 2,
    "position": 1,
    "arm": "pre_n16",
    "prefill_p50": 10227.739825852372,
    "prefill_p90": 10334.240092909464,
    "prefill_p99": 10347.772585128747,
    "latency_p50_ms": 23.8567245,
    "latency_p90_ms": 24.0063217,
    "latency_p99_ms": 24.07263397,
    "latency_mad_ms": 0.12189599999999956,
    "latency_max_ms": 24.080002,
    "decode_p50": 204.04602049925995,
    "cold_prefill_p50_ms": 38.882853000000004,
    "cold_prefill_p90_ms": 48.144913,
    "cold_prefill_max_ms": 48.148299,
    "cold_decode_p50_ms": 5.96626,
    "samples_ms": [
      23.827,
      23.66,
      23.754,
      23.886,
      23.577,
      24.08,
      23.893,
      23.998,
      23.937,
      23.615
    ],
    "cold_samples_ms": [
      48.14,
      48.148,
      38.883,
      34.34,
      34.306
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 2,
    "position": 2,
    "arm": "n16_current",
    "prefill_p50": 10868.056687192906,
    "prefill_p90": 10936.129093548634,
    "prefill_p99": 11043.348169825656,
    "latency_p50_ms": 22.451116499999998,
    "latency_p90_ms": 22.7250857,
    "latency_p99_ms": 22.80417167,
    "latency_mad_ms": 0.07942249999999973,
    "latency_max_ms": 22.812959,
    "decode_p50": 206.0028949563355,
    "cold_prefill_p50_ms": 35.97061,
    "cold_prefill_p90_ms": 44.99807800000001,
    "cold_prefill_max_ms": 45.032026,
    "cold_decode_p50_ms": 6.15142,
    "samples_ms": [
      22.498,
      22.813,
      22.715,
      22.338,
      22.447,
      22.339,
      22.071,
      22.455,
      22.431,
      22.481
    ],
    "cold_samples_ms": [
      45.032,
      44.947,
      35.971,
      34.932,
      34.961
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 3,
    "position": 0,
    "arm": "n16_regq",
    "prefill_p50": 11291.595878461421,
    "prefill_p90": 11355.899482002515,
    "prefill_p99": 11379.331283469137,
    "latency_p50_ms": 21.609049,
    "latency_p90_ms": 21.746560699999996,
    "latency_p99_ms": 21.98482787,
    "latency_mad_ms": 0.07770799999999944,
    "latency_max_ms": 22.011302,
    "decode_p50": 203.32505616707328,
    "cold_prefill_p50_ms": 41.966381,
    "cold_prefill_p90_ms": 42.0902308,
    "cold_prefill_max_ms": 42.172156,
    "cold_decode_p50_ms": 6.858805,
    "samples_ms": [
      22.011,
      21.567,
      21.564,
      21.574,
      21.644,
      21.709,
      21.437,
      21.665,
      21.717,
      21.492
    ],
    "cold_samples_ms": [
      42.172,
      41.967,
      41.966,
      36.464,
      22.349
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 3,
    "position": 1,
    "arm": "n16_current",
    "prefill_p50": 10715.048521263912,
    "prefill_p90": 10831.55681559777,
    "prefill_p99": 10839.09634304987,
    "latency_p50_ms": 22.771714,
    "latency_p90_ms": 22.894421400000002,
    "latency_p99_ms": 23.09366844,
    "latency_mad_ms": 0.12432199999999938,
    "latency_max_ms": 23.115807,
    "decode_p50": 204.44705897429958,
    "cold_prefill_p50_ms": 44.999249999999996,
    "cold_prefill_p90_ms": 45.0768722,
    "cold_prefill_max_ms": 45.093537000000005,
    "cold_decode_p50_ms": 6.904123,
    "samples_ms": [
      22.529,
      22.87,
      22.818,
      22.85,
      22.778,
      22.621,
      22.616,
      22.765,
      22.509,
      23.116
    ],
    "cold_samples_ms": [
      45.094,
      45.052,
      44.889,
      44.999,
      44.951
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 3,
    "position": 2,
    "arm": "pre_n16",
    "prefill_p50": 10060.050573958371,
    "prefill_p90": 10149.114038548167,
    "prefill_p99": 10250.489424238951,
    "latency_p50_ms": 24.254351999999997,
    "latency_p90_ms": 24.4976736,
    "latency_p99_ms": 24.51054936,
    "latency_mad_ms": 0.11245800000000017,
    "latency_max_ms": 24.511979999999998,
    "decode_p50": 203.02736875469276,
    "cold_prefill_p50_ms": 48.079385,
    "cold_prefill_p90_ms": 48.100083999999995,
    "cold_prefill_max_ms": 48.109606,
    "cold_decode_p50_ms": 6.854025,
    "samples_ms": [
      24.496,
      24.386,
      24.289,
      24.161,
      23.778,
      24.25,
      24.071,
      24.24,
      24.259,
      24.512
    ],
    "cold_samples_ms": [
      48.11,
      48.086,
      48.079,
      47.992,
      44.722
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 4,
    "position": 0,
    "arm": "pre_n16",
    "prefill_p50": 10049.431271902748,
    "prefill_p90": 10162.331199499897,
    "prefill_p99": 10220.766914504793,
    "latency_p50_ms": 24.2800105,
    "latency_p90_ms": 24.524672899999995,
    "latency_p99_ms": 24.58236029,
    "latency_mad_ms": 0.16088900000000095,
    "latency_max_ms": 24.58877,
    "decode_p50": 203.06886176572095,
    "cold_prefill_p50_ms": 48.076817,
    "cold_prefill_p90_ms": 48.13225800000001,
    "cold_prefill_max_ms": 48.142714000000005,
    "cold_decode_p50_ms": 6.8735100000000005,
    "samples_ms": [
      24.518,
      24.188,
      24.027,
      24.152,
      23.858,
      24.307,
      24.253,
      24.589,
      24.323,
      24.474
    ],
    "cold_samples_ms": [
      48.143,
      48.117,
      48.077,
      27.363,
      26.655
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 4,
    "position": 1,
    "arm": "n16_regq",
    "prefill_p50": 11178.731735468944,
    "prefill_p90": 11245.150256031888,
    "prefill_p99": 11253.529425915951,
    "latency_p50_ms": 21.8271695,
    "latency_p90_ms": 22.023590600000002,
    "latency_p99_ms": 22.11736106,
    "latency_mad_ms": 0.08934849999999983,
    "latency_max_ms": 22.12778,
    "decode_p50": 201.70619395027995,
    "cold_prefill_p50_ms": 36.550314,
    "cold_prefill_p90_ms": 41.7360364,
    "cold_prefill_max_ms": 41.891404,
    "cold_decode_p50_ms": 6.435662,
    "samples_ms": [
      21.815,
      21.68,
      21.839,
      22.012,
      21.845,
      21.768,
      21.752,
      21.7,
      21.931,
      22.128
    ],
    "cold_samples_ms": [
      41.891,
      41.503,
      36.55,
      36.495,
      36.51
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 4,
    "position": 2,
    "arm": "n16_current",
    "prefill_p50": 10654.633197897025,
    "prefill_p90": 10733.243853628912,
    "prefill_p99": 10759.296010935337,
    "latency_p50_ms": 22.900961000000002,
    "latency_p90_ms": 23.0305787,
    "latency_p99_ms": 23.09416127,
    "latency_mad_ms": 0.08255750000000006,
    "latency_max_ms": 23.101226,
    "decode_p50": 203.51122835151045,
    "cold_prefill_p50_ms": 44.995870000000004,
    "cold_prefill_p90_ms": 45.026639599999996,
    "cold_prefill_max_ms": 45.037564,
    "cold_decode_p50_ms": 6.876133,
    "samples_ms": [
      22.993,
      22.843,
      22.74,
      23.101,
      22.955,
      22.672,
      23.023,
      22.961,
      22.847,
      22.828
    ],
    "cold_samples_ms": [
      45.01,
      45.038,
      44.996,
      30.891,
      24.223
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 5,
    "position": 0,
    "arm": "n16_current",
    "prefill_p50": 10616.937197131148,
    "prefill_p90": 10695.500858835898,
    "prefill_p99": 10807.860056289948,
    "latency_p50_ms": 22.982185,
    "latency_p90_ms": 23.1220782,
    "latency_p99_ms": 23.271427619999997,
    "latency_mad_ms": 0.11110799999999976,
    "latency_max_ms": 23.288021999999998,
    "decode_p50": 201.9373168686531,
    "cold_prefill_p50_ms": 45.010147,
    "cold_prefill_p90_ms": 45.0399252,
    "cold_prefill_max_ms": 45.048131999999995,
    "cold_decode_p50_ms": 6.876003000000001,
    "samples_ms": [
      22.952,
      22.843,
      23.012,
      22.55,
      22.868,
      23.104,
      22.879,
      23.09,
      23.071,
      23.288
    ],
    "cold_samples_ms": [
      45.028,
      45.048,
      45.002,
      45.01,
      34.259
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 5,
    "position": 1,
    "arm": "pre_n16",
    "prefill_p50": 9945.141062486508,
    "prefill_p90": 10049.445563370424,
    "prefill_p99": 10060.687105370218,
    "latency_p50_ms": 24.534596,
    "latency_p90_ms": 24.7362374,
    "latency_p99_ms": 24.744283940000003,
    "latency_mad_ms": 0.14860099999999932,
    "latency_max_ms": 24.745178,
    "decode_p50": 201.56083171468907,
    "cold_prefill_p50_ms": 47.986116,
    "cold_prefill_p90_ms": 48.13393620000001,
    "cold_prefill_max_ms": 48.150539,
    "cold_decode_p50_ms": 6.856471,
    "samples_ms": [
      24.541,
      24.436,
      24.25,
      24.702,
      24.283,
      24.528,
      24.405,
      24.735,
      24.591,
      24.745
    ],
    "cold_samples_ms": [
      48.151,
      47.986,
      48.109,
      47.934,
      28.058
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 5,
    "position": 2,
    "arm": "n16_regq",
    "prefill_p50": 11167.420857866253,
    "prefill_p90": 11244.376346727833,
    "prefill_p99": 11273.952560884052,
    "latency_p50_ms": 21.8493465,
    "latency_p90_ms": 21.979805,
    "latency_p99_ms": 21.994336399999998,
    "latency_mad_ms": 0.09172700000000056,
    "latency_max_ms": 21.995950999999998,
    "decode_p50": 201.87830580494074,
    "cold_prefill_p50_ms": 41.988704,
    "cold_prefill_p90_ms": 42.035706399999995,
    "cold_prefill_max_ms": 42.065011999999996,
    "cold_decode_p50_ms": 6.901626,
    "samples_ms": [
      21.809,
      21.783,
      21.707,
      21.637,
      21.89,
      21.96,
      21.996,
      21.978,
      21.922,
      21.788
    ],
    "cold_samples_ms": [
      41.989,
      42.065,
      41.964,
      41.992,
      29.214
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 6,
    "position": 0,
    "arm": "pre_n16",
    "prefill_p50": 9918.616123084223,
    "prefill_p90": 10037.901373497156,
    "prefill_p99": 10039.306753504305,
    "latency_p50_ms": 24.6002085,
    "latency_p90_ms": 24.814420000000002,
    "latency_p99_ms": 25.075199499999997,
    "latency_mad_ms": 0.11439149999999998,
    "latency_max_ms": 25.104174999999998,
    "decode_p50": 200.79866055034975,
    "cold_prefill_p50_ms": 48.034256,
    "cold_prefill_p90_ms": 48.151775799999996,
    "cold_prefill_max_ms": 48.174579,
    "cold_decode_p50_ms": 6.87535,
    "samples_ms": [
      24.308,
      24.52,
      24.304,
      24.782,
      24.564,
      25.104,
      24.66,
      24.608,
      24.592,
      24.749
    ],
    "cold_samples_ms": [
      48.175,
      48.118,
      48.034,
      44.131,
      25.363
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 6,
    "position": 1,
    "arm": "n16_current",
    "prefill_p50": 10526.66182513579,
    "prefill_p90": 10637.767921802166,
    "prefill_p99": 10656.728541249218,
    "latency_p50_ms": 23.1792435,
    "latency_p90_ms": 23.3107759,
    "latency_p99_ms": 23.33194039,
    "latency_mad_ms": 0.12325900000000267,
    "latency_max_ms": 23.334291999999998,
    "decode_p50": 202.383833964081,
    "cold_prefill_p50_ms": 44.855785000000004,
    "cold_prefill_p90_ms": 44.960257999999996,
    "cold_prefill_max_ms": 44.966004,
    "cold_decode_p50_ms": 6.872508,
    "samples_ms": [
      23.308,
      23.191,
      22.892,
      23.297,
      22.964,
      23.241,
      23.168,
      22.942,
      23.126,
      23.334
    ],
    "cold_samples_ms": [
      44.856,
      44.952,
      44.966,
      41.554,
      23.791
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 6,
    "position": 2,
    "arm": "n16_regq",
    "prefill_p50": 11046.80113003112,
    "prefill_p90": 11092.266908017427,
    "prefill_p99": 11210.25119503164,
    "latency_p50_ms": 22.0878715,
    "latency_p90_ms": 22.239094,
    "latency_p99_ms": 22.2468862,
    "latency_mad_ms": 0.10110250000000143,
    "latency_max_ms": 22.247752,
    "decode_p50": 201.7241784533162,
    "cold_prefill_p50_ms": 42.003947,
    "cold_prefill_p90_ms": 42.118551999999994,
    "cold_prefill_max_ms": 42.156706,
    "cold_decode_p50_ms": 6.884094,
    "samples_ms": [
      22.062,
      22.233,
      22.238,
      22.062,
      22.228,
      22.113,
      22.026,
      22.248,
      22.051,
      21.74
    ],
    "cold_samples_ms": [
      42.157,
      42.061,
      42.004,
      28.587,
      24.379
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 7,
    "position": 0,
    "arm": "n16_current",
    "prefill_p50": 10461.866138885956,
    "prefill_p90": 10583.178114035343,
    "prefill_p99": 10589.832161996852,
    "latency_p50_ms": 23.3228115,
    "latency_p90_ms": 23.5386343,
    "latency_p99_ms": 23.726070730000004,
    "latency_mad_ms": 0.1701119999999996,
    "latency_max_ms": 23.746897,
    "decode_p50": 201.97848532869506,
    "cold_prefill_p50_ms": 44.951663,
    "cold_prefill_p90_ms": 45.0333228,
    "cold_prefill_max_ms": 45.06717,
    "cold_decode_p50_ms": 6.855578,
    "samples_ms": [
      23.515,
      23.306,
      23.747,
      23.43,
      23.057,
      23.254,
      23.108,
      23.47,
      23.34,
      23.039
    ],
    "cold_samples_ms": [
      45.067,
      44.952,
      44.983,
      41.719,
      23.762
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 7,
    "position": 1,
    "arm": "n16_regq",
    "prefill_p50": 11013.480808579052,
    "prefill_p90": 11111.531791747633,
    "prefill_p99": 11172.161096995571,
    "latency_p50_ms": 22.154707000000002,
    "latency_p90_ms": 22.6631437,
    "latency_p99_ms": 22.76003347,
    "latency_mad_ms": 0.17171600000000353,
    "latency_max_ms": 22.770799,
    "decode_p50": 201.2509234286045,
    "cold_prefill_p50_ms": 41.952609,
    "cold_prefill_p90_ms": 42.0084044,
    "cold_prefill_max_ms": 42.029816,
    "cold_decode_p50_ms": 6.882414,
    "samples_ms": [
      22.454,
      22.184,
      21.974,
      22.259,
      22.125,
      22.039,
      21.992,
      21.827,
      22.771,
      22.651
    ],
    "cold_samples_ms": [
      41.976,
      42.03,
      41.903,
      41.953,
      41.912
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 7,
    "position": 2,
    "arm": "pre_n16",
    "prefill_p50": 9852.74508661875,
    "prefill_p90": 9939.949267187701,
    "prefill_p99": 9949.351738964086,
    "latency_p50_ms": 24.764671999999997,
    "latency_p90_ms": 25.069456,
    "latency_p99_ms": 25.1077933,
    "latency_mad_ms": 0.08222749999999834,
    "latency_max_ms": 25.112053,
    "decode_p50": 200.73255399197836,
    "cold_prefill_p50_ms": 45.324003000000005,
    "cold_prefill_p90_ms": 48.054596399999994,
    "cold_prefill_max_ms": 48.056571999999996,
    "cold_decode_p50_ms": 5.376063,
    "samples_ms": [
      24.738,
      24.522,
      24.779,
      24.766,
      25.112,
      24.798,
      24.764,
      24.55,
      25.065,
      24.634
    ],
    "cold_samples_ms": [
      48.057,
      48.052,
      45.324,
      28.444,
      28.41
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 8,
    "position": 0,
    "arm": "n16_regq",
    "prefill_p50": 10875.996098622349,
    "prefill_p90": 11015.620323432253,
    "prefill_p99": 11057.944824068529,
    "latency_p50_ms": 22.434763500000003,
    "latency_p90_ms": 22.4968413,
    "latency_p99_ms": 22.56262383,
    "latency_mad_ms": 0.07817050000000059,
    "latency_max_ms": 22.569933,
    "decode_p50": 199.98600184042272,
    "cold_prefill_p50_ms": 41.933748,
    "cold_prefill_p90_ms": 42.06179879999999,
    "cold_prefill_max_ms": 42.092777999999996,
    "cold_decode_p50_ms": 6.903829,
    "samples_ms": [
      22.406,
      22.463,
      22.478,
      22.161,
      22.489,
      22.332,
      22.056,
      22.317,
      22.57,
      22.467
    ],
    "cold_samples_ms": [
      41.934,
      42.093,
      42.015,
      41.8,
      34.422
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 8,
    "position": 1,
    "arm": "pre_n16",
    "prefill_p50": 9734.844988729685,
    "prefill_p90": 9876.765168271912,
    "prefill_p99": 9902.013754729542,
    "latency_p50_ms": 25.064600499999997,
    "latency_p90_ms": 25.486529100000002,
    "latency_p99_ms": 25.52459181,
    "latency_mad_ms": 0.19352500000000106,
    "latency_max_ms": 25.528821,
    "decode_p50": 200.43844322769786,
    "cold_prefill_p50_ms": 40.655323,
    "cold_prefill_p90_ms": 48.148239000000004,
    "cold_prefill_max_ms": 48.171827,
    "cold_decode_p50_ms": 5.618977,
    "samples_ms": [
      24.634,
      25.063,
      24.895,
      25.529,
      25.282,
      25.185,
      24.712,
      25.047,
      25.066,
      25.482
    ],
    "cold_samples_ms": [
      48.172,
      48.113,
      40.655,
      30.767,
      30.835
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 8,
    "position": 2,
    "arm": "n16_current",
    "prefill_p50": 10391.867135697452,
    "prefill_p90": 10457.70852248271,
    "prefill_p99": 10472.372481659542,
    "latency_p50_ms": 23.4799065,
    "latency_p90_ms": 23.58708,
    "latency_p99_ms": 23.774165699999998,
    "latency_mad_ms": 0.07602100000000256,
    "latency_max_ms": 23.794953,
    "decode_p50": 200.91260607420918,
    "cold_prefill_p50_ms": 44.986911,
    "cold_prefill_p90_ms": 45.0062184,
    "cold_prefill_max_ms": 45.013336,
    "cold_decode_p50_ms": 6.877001,
    "samples_ms": [
      23.296,
      23.564,
      23.388,
      23.492,
      23.512,
      23.548,
      23.467,
      23.336,
      23.795,
      23.462
    ],
    "cold_samples_ms": [
      44.987,
      44.937,
      45.013,
      44.996,
      24.604
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 9,
    "position": 0,
    "arm": "n16_regq",
    "prefill_p50": 10930.971374742225,
    "prefill_p90": 10985.552346207181,
    "prefill_p99": 11048.091693147551,
    "latency_p50_ms": 22.321937,
    "latency_p90_ms": 22.6723237,
    "latency_p99_ms": 22.848091269999998,
    "latency_mad_ms": 0.13533699999999982,
    "latency_max_ms": 22.867621,
    "decode_p50": 199.35294161369674,
    "cold_prefill_p50_ms": 41.992585,
    "cold_prefill_p90_ms": 42.11025079999999,
    "cold_prefill_max_ms": 42.152193999999994,
    "cold_decode_p50_ms": 6.895829,
    "samples_ms": [
      22.497,
      22.291,
      22.518,
      22.232,
      22.227,
      22.24,
      22.071,
      22.868,
      22.651,
      22.352
    ],
    "cold_samples_ms": [
      42.047,
      41.993,
      42.152,
      39.435,
      22.235
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 9,
    "position": 1,
    "arm": "n16_current",
    "prefill_p50": 10384.549453558715,
    "prefill_p90": 10496.434300944264,
    "prefill_p99": 10523.047280178933,
    "latency_p50_ms": 23.496446,
    "latency_p90_ms": 23.660298,
    "latency_p99_ms": 23.757684299999998,
    "latency_mad_ms": 0.08157850000000089,
    "latency_max_ms": 23.768504999999998,
    "decode_p50": 201.1051119209527,
    "cold_prefill_p50_ms": 41.254463,
    "cold_prefill_p90_ms": 44.3100362,
    "cold_prefill_max_ms": 44.898735,
    "cold_decode_p50_ms": 6.59557,
    "samples_ms": [
      23.648,
      23.52,
      23.387,
      23.253,
      23.55,
      23.181,
      23.5,
      23.471,
      23.493,
      23.769
    ],
    "cold_samples_ms": [
      44.899,
      43.427,
      41.254,
      41.186,
      41.251
    ],
    "oracle": "50/50"
  },
  {
    "repeat": 9,
    "position": 2,
    "arm": "pre_n16",
    "prefill_p50": 9725.955052719353,
    "prefill_p90": 9811.264160047233,
    "prefill_p99": 9819.794639607895,
    "latency_p50_ms": 25.087561,
    "latency_p90_ms": 25.506219599999998,
    "latency_p99_ms": 25.51495626,
    "latency_mad_ms": 0.19098450000000078,
    "latency_max_ms": 25.515927,
    "decode_p50": 199.88428784946254,
    "cold_prefill_p50_ms": 48.037870000000005,
    "cold_prefill_p90_ms": 48.159546000000006,
    "cold_prefill_max_ms": 48.214418,
    "cold_decode_p50_ms": 6.8672059999999995,
    "samples_ms": [
      25.516,
      25.254,
      25.505,
      25.009,
      25.123,
      24.845,
      25.497,
      24.999,
      25.052,
      24.872
    ],
    "cold_samples_ms": [
      48.214,
      48.006,
      48.077,
      48.038,
      44.468
    ],
    "oracle": "50/50"
  }
]
```