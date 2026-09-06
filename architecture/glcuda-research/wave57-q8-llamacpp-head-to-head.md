# GwenLand vs llama.cpp - T4 prefill head-to-head

- notebook: wave57-h2h-q8-v2
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- GwenLand Wave 50 patch: 915ea7ede79038970018ae10d2c7851d0e01e64f3a73d38e1c0eef764b6dc6c9
- llama.cpp commit: 4d9176092d00586775af140581bb0b558ddc4389
- model SHA-256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
- full CUDA parity: test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.83s
- effective prompt: 244 token ids, byte-for-byte equal; six position-balanced pairs
- GwenLand cold/warmup/measure: 5/5/10
- llama-bench: default warmup plus 10 raw repetitions

## Result

| engine | P50 tok/s | P90 | P99 | latency P50/P90/P99 ms | session-max median ms | session P50 range |
|---|---:|---:|---:|---:|---:|---:|
| `gwenland` | 11499.7 | 11583.3 | 11678.6 | 21.218/21.554/21.677 | 21.691 | 11212.0-11566.1 |
| `llamacpp` | 11453.5 | 11713.8 | 12164.0 | 21.304/22.324/27.810 | 28.392 | 11310.8-11536.8 |

**Observed median difference: +0.40% (+46.2 tok/s); effectively tied in this run.**

## Correctness qualification and protocol deviation

All six GwenLand sessions have `validation.passed=false`: their greedy
continuations match glproc for a 29-token prefix out of 50. Version 1 stopped
on this failure. Version 2 weakened the acceptance rule after seeing that
failure, requiring only a matching first token. This is a post-hoc protocol
change, not a correctness repair. The cause of the divergence has not been
established; matching the first token does not prove prefill logits or the
subsequent KV state are numerically correct.

These timings are archived as qualified performance observations. They do
not satisfy the original 50/50 production acceptance gate and must not be
used to declare a correctness-validated speedup. Four paired differences are
positive and two negative. Wave 58 must characterize this divergence before
promoting further optimizations.

| repeat | GwenLand / llama.cpp | absolute tok/s |
|---:|---:|---:|
| 0 | 1.0008x | +9.2 |
| 1 | 1.0025x | +29.2 |
| 2 | 0.9991x | -10.2 |
| 3 | 1.0012x | +13.8 |
| 4 | 1.0106x | +120.8 |
| 5 | 0.9913x | -98.8 |

## Prompt and fairness contract

- Effective ChatML tokenizer parity: 244 token ids, exact=True (glcore `encode_chat` vs pinned `llama-tokenize`).
- The raw user text and effective ChatML bytes are archived with the result.
- `llama-bench` cannot accept prompt text; its throughput run receives `-p 244`, so the
  compared dense-model compute shape is identical but its token IDs are synthetic.
- Both processes see exactly one Tesla T4 through `CUDA_VISIBLE_DEVICES=0` and offload
  every layer. Both engines consume the exact same native Q8_0 GGUF weights; GwenLand keeps
  Q8_0 as Q8_0. No quant conversion or byte-ratio correction is applied.
- GwenLand requires the prefill-produced first token to equal glproc in every session.
  The 50-token autoregressive matching prefix is retained as a diagnostic, not a prefill gate.
- The primary statistic is the median of each tool's ten raw per-session samples, then
  the median across six sessions. Tool order is balanced 3/3.

## Build and dispatch

```json
{
  "llama_build_flags": [
    "GGML_CUDA=ON",
    "CMAKE_BUILD_TYPE=Release",
    "CMAKE_CUDA_ARCHITECTURES=75-real",
    "LLAMA_CURL=OFF",
    "LLAMA_BUILD_TESTS=OFF",
    "LLAMA_BUILD_EXAMPLES=OFF",
    "LLAMA_BUILD_TOOLS=ON",
    "LLAMA_BUILD_SERVER=ON",
    "LLAMA_BUILD_APP=OFF",
    "GGML_CUDA_NCCL=OFF"
  ],
  "llama_build_jobs": 2,
  "llama_build_seconds": 1815.0792953968048,
  "libcuda": "/usr/local/nvidia/lib64/libcuda.so",
  "gwenland_dispatch": {
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
  },
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
  "driver_resource": {
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
  "direct_exact": {
    "ntok": 244,
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "warmup": 20,
    "iters": 200,
    "rounds": 7,
    "retained_us": 192.05,
    "candidate_us": 148.064,
    "speedup": 1.2971,
    "bit_exact": true,
    "retained_dynamic_shared": 15616,
    "candidate_dynamic_shared": 15616,
    "retained_blocks_per_sm": 3,
    "candidate_blocks_per_sm": 4
  }
}
```

## Raw paired sessions

```json
[
  {
    "repeat": 0,
    "position": 0,
    "tool": "gwenland",
    "p50_tps": 11511.49874398685,
    "p90_tps": 11774.0864736683,
    "p99_tps": 11849.297347148815,
    "p50_ms": 21.196316,
    "p90_ms": 21.301308,
    "p99_ms": 21.3599115,
    "max_ms": 21.366422999999998,
    "samples_tps": [
      11559.309657182617,
      11652.744581915524,
      11419.787018163968,
      11484.271490060693,
      11475.24908697698,
      11458.587560961212,
      11463.609804280475,
      11764.801180646014,
      11857.654110868873,
      11538.725997913005
    ],
    "samples_ms": [
      21.108527,
      20.939273,
      21.366422999999998,
      21.24645,
      21.263154999999998,
      21.294073,
      21.284744,
      20.739832,
      20.577426,
      21.146182
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  },
  {
    "repeat": 0,
    "position": 1,
    "tool": "llamacpp",
    "p50_tps": 11502.25,
    "p90_tps": 11978.25,
    "p99_tps": 12184.395,
    "p50_ms": 21.2132025,
    "p90_ms": 22.162831999999995,
    "p99_ms": 27.966214700000002,
    "max_ms": 28.611035,
    "samples_tps": [
      8528.18,
      12207.3,
      11952.8,
      11629.9,
      11377.2,
      11406.8,
      11492.4,
      11497.7,
      11506.8,
      11546.7
    ],
    "samples_ms": [
      28.611035,
      19.988097,
      20.413585,
      20.980372,
      21.446365,
      21.390797,
      21.231409,
      21.221598,
      21.204807,
      21.131552
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:04:40Z",
      "avg_ns": 21761961,
      "stddev_ns": 2449589,
      "avg_ts": 11314.58466,
      "stddev_ts": 1013.034373,
      "samples_ns": [
        28611035,
        19988097,
        20413585,
        20980372,
        21446365,
        21390797,
        21231409,
        21221598,
        21204807,
        21131552
      ],
      "samples_ts": [
        8528.18,
        12207.3,
        11952.8,
        11629.9,
        11377.2,
        11406.8,
        11492.4,
        11497.7,
        11506.8,
        11546.7
      ]
    }
  },
  {
    "repeat": 1,
    "position": 0,
    "tool": "llamacpp",
    "p50_tps": 11536.849999999999,
    "p90_tps": 11979.73,
    "p99_tps": 12187.333,
    "p50_ms": 21.150340999999997,
    "p90_ms": 22.2749001,
    "p99_ms": 27.32999751,
    "max_ms": 27.891675,
    "samples_tps": [
      8748.13,
      12210.4,
      11954.1,
      11602.8,
      11312.1,
      11269.8,
      11314.3,
      11470.9,
      11637.2,
      11725.8
    ],
    "samples_ms": [
      27.891675,
      19.982958,
      20.411448,
      21.029484,
      21.569731,
      21.650814,
      21.565657,
      21.271198,
      20.967287,
      20.808849
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:04:42Z",
      "avg_ns": 21714910,
      "stddev_ns": 2234705,
      "avg_ts": 11324.545232,
      "stddev_ts": 953.343869,
      "samples_ns": [
        27891675,
        19982958,
        20411448,
        21029484,
        21569731,
        21650814,
        21565657,
        21271198,
        20967287,
        20808849
      ],
      "samples_ts": [
        8748.13,
        12210.4,
        11954.1,
        11602.8,
        11312.1,
        11269.8,
        11314.3,
        11470.9,
        11637.2,
        11725.8
      ]
    }
  },
  {
    "repeat": 1,
    "position": 1,
    "tool": "gwenland",
    "p50_tps": 11566.082673397897,
    "p90_tps": 11709.300007444244,
    "p99_tps": 11731.446231094098,
    "p50_ms": 21.096562,
    "p90_ms": 21.4609654,
    "p99_ms": 21.611450440000002,
    "max_ms": 21.628171000000002,
    "samples_tps": [
      11706.565905759076,
      11474.031163844838,
      11281.582709883327,
      11515.971992400968,
      11665.279900741854,
      11379.330109096529,
      11468.953425567222,
      11733.906922610748,
      11621.464883743438,
      11616.193354394825
    ],
    "samples_ms": [
      20.843003999999997,
      21.265412,
      21.628171000000002,
      21.187964,
      20.916772,
      21.442387,
      21.274827,
      20.794438,
      20.995632,
      21.00516
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  },
  {
    "repeat": 2,
    "position": 0,
    "tool": "gwenland",
    "p50_tps": 11487.865340146802,
    "p90_tps": 11581.78593882216,
    "p99_tps": 11659.609485572293,
    "p50_ms": 21.2398545,
    "p90_ms": 21.7396835,
    "p99_ms": 22.44249215,
    "max_ms": 22.520582,
    "samples_tps": [
      10834.53349473828,
      11456.669514243786,
      11505.703834789787,
      11572.178093544366,
      11370.827331610282,
      11470.026845503817,
      11668.256546322309,
      11528.2255136356,
      11268.689571940815,
      11506.18509320906
    ],
    "samples_ms": [
      22.520582,
      21.297638,
      21.206873,
      21.085054,
      21.458420999999998,
      21.272835999999998,
      20.911436,
      21.165443,
      21.652917000000002,
      21.205986
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  },
  {
    "repeat": 2,
    "position": 1,
    "tool": "llamacpp",
    "p50_tps": 11498.1,
    "p90_tps": 11725.78,
    "p99_tps": 12143.578,
    "p50_ms": 21.221543500000003,
    "p90_ms": 22.216320699999997,
    "p99_ms": 27.62932147,
    "max_ms": 28.230766,
    "samples_tps": [
      8643.05,
      12190.0,
      11674.2,
      11375.4,
      11433.0,
      11375.4,
      11323.5,
      11563.2,
      11618.8,
      11600.5
    ],
    "samples_ms": [
      28.230766,
      20.016381,
      20.900839,
      21.449766,
      21.341733,
      21.449831,
      21.548049,
      21.101354,
      21.000444,
      21.033581
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:05:12Z",
      "avg_ns": 21807274,
      "stddev_ns": 2299220,
      "avg_ts": 11279.710521,
      "stddev_ts": 959.063509,
      "samples_ns": [
        28230766,
        20016381,
        20900839,
        21449766,
        21341733,
        21449831,
        21548049,
        21101354,
        21000444,
        21033581
      ],
      "samples_ts": [
        8643.05,
        12190,
        11674.2,
        11375.4,
        11433,
        11375.4,
        11323.5,
        11563.2,
        11618.8,
        11600.5
      ]
    }
  },
  {
    "repeat": 3,
    "position": 0,
    "tool": "llamacpp",
    "p50_tps": 11370.2,
    "p90_tps": 11701.89,
    "p99_tps": 12310.929,
    "p50_ms": 21.4596135,
    "p90_ms": 22.372330699999996,
    "p99_ms": 27.76381787,
    "max_ms": 28.362872,
    "samples_tps": [
      8602.8,
      12378.6,
      11626.7,
      11333.5,
      11240.8,
      11337.5,
      11462.7,
      11359.3,
      11445.8,
      11381.1
    ],
    "samples_ms": [
      28.362872,
      19.711402,
      20.986246,
      21.529048,
      21.706715,
      21.521504,
      21.286463,
      21.480106,
      21.31786,
      21.439121
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:05:14Z",
      "avg_ns": 21934133,
      "stddev_ns": 2328674,
      "avg_ts": 11216.876188,
      "stddev_ts": 974.098186,
      "samples_ns": [
        28362872,
        19711402,
        20986246,
        21529048,
        21706715,
        21521504,
        21286463,
        21480106,
        21317860,
        21439121
      ],
      "samples_ts": [
        8602.8,
        12378.6,
        11626.7,
        11333.5,
        11240.8,
        11337.5,
        11462.7,
        11359.3,
        11445.8,
        11381.1
      ]
    }
  },
  {
    "repeat": 3,
    "position": 1,
    "tool": "gwenland",
    "p50_tps": 11384.006852176852,
    "p90_tps": 11580.665468635681,
    "p99_tps": 11697.57819695946,
    "p50_ms": 21.433714000000002,
    "p90_ms": 21.647069300000002,
    "p99_ms": 21.743178229999998,
    "max_ms": 21.753857,
    "samples_tps": [
      11710.568500106547,
      11355.341971708072,
      11216.401762685118,
      11413.36091998799,
      11349.59578461128,
      11310.290686522592,
      11412.671732645631,
      11277.915382725301,
      11493.33895062424,
      11566.231798472252
    ],
    "samples_ms": [
      20.83588,
      21.487684,
      21.753857,
      21.378453,
      21.498562999999997,
      21.573274,
      21.379744,
      21.635204,
      21.229688,
      21.095894
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  },
  {
    "repeat": 4,
    "position": 0,
    "tool": "gwenland",
    "p50_tps": 11529.703637679846,
    "p90_tps": 11584.723075891707,
    "p99_tps": 11642.977726827588,
    "p50_ms": 21.162741500000003,
    "p90_ms": 21.2944768,
    "p99_ms": 21.39856828,
    "max_ms": 21.410134,
    "samples_tps": [
      11538.363687899368,
      11568.656753781814,
      11649.450465820464,
      11495.055924624845,
      11396.472343423913,
      11465.289353360498,
      11577.531143677399,
      11521.043587460324,
      11498.732547492358,
      11542.011289127713
    ],
    "samples_ms": [
      21.146846,
      21.091472,
      20.945194,
      21.226517,
      21.410134,
      21.281626000000003,
      21.075305,
      21.178637000000002,
      21.21973,
      21.140163
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  },
  {
    "repeat": 4,
    "position": 1,
    "tool": "llamacpp",
    "p50_tps": 11408.95,
    "p90_tps": 11640.630000000001,
    "p99_tps": 11966.493,
    "p50_ms": 21.3866485,
    "p90_ms": 22.459180499999995,
    "p99_ms": 28.03557285,
    "max_ms": 28.655172,
    "samples_tps": [
      8515.04,
      12002.7,
      11600.4,
      11429.8,
      11408.0,
      11409.9,
      11306.0,
      11482.1,
      11207.7,
      11234.6
    ],
    "samples_ms": [
      28.655172,
      20.328703,
      21.033746,
      21.347662,
      21.388406,
      21.384891,
      21.581479,
      21.250449,
      21.770737,
      21.718613
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:05:44Z",
      "avg_ns": 22045985,
      "stddev_ns": 2358061,
      "avg_ts": 11159.63889,
      "stddev_ts": 956.246292,
      "samples_ns": [
        28655172,
        20328703,
        21033746,
        21347662,
        21388406,
        21384891,
        21581479,
        21250449,
        21770737,
        21718613
      ],
      "samples_ts": [
        8515.04,
        12002.7,
        11600.4,
        11429.8,
        11408,
        11409.9,
        11306,
        11482.1,
        11207.7,
        11234.6
      ]
    }
  },
  {
    "repeat": 5,
    "position": 0,
    "tool": "llamacpp",
    "p50_tps": 11310.75,
    "p90_tps": 11477.01,
    "p99_tps": 12057.861,
    "p50_ms": 21.572403,
    "p90_ms": 22.7748712,
    "p99_ms": 27.855986620000003,
    "max_ms": 28.420555,
    "samples_tps": [
      8585.34,
      12122.4,
      11405.3,
      11017.0,
      11075.4,
      11265.0,
      11320.9,
      11301.2,
      11327.7,
      11320.3
    ],
    "samples_ms": [
      28.420555,
      20.127953,
      21.393639,
      22.147573,
      22.030748,
      21.659923,
      21.553112,
      21.590668,
      21.540114,
      21.554138
    ],
    "raw": {
      "build_commit": "4d9176092",
      "build_number": 1,
      "cpu_info": "Intel(R) Xeon(R) CPU @ 2.00GHz",
      "gpu_info": "Tesla T4",
      "backends": "CUDA",
      "model_filename": "/kaggle/working/qwen2.5-0.5b-instruct-q8_0.gguf",
      "model_type": "qwen2 1B Q8_0",
      "model_size": 669763072,
      "model_n_params": 630167424,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 2,
      "cpu_mask": "0x0",
      "cpu_strict": false,
      "poll": 50,
      "type_k": "f16",
      "type_v": "f16",
      "n_gpu_layers": 99,
      "n_cpu_moe": 0,
      "split_mode": "layer",
      "main_gpu": 0,
      "no_kv_offload": false,
      "flash_attn": -1,
      "devices": "auto",
      "tensor_split": "0.00",
      "tensor_buft_overrides": "none",
      "load_mode": "auto",
      "lazy_mode": "auto",
      "embeddings": false,
      "no_op_offload": 0,
      "no_host": false,
      "fit_target": 0,
      "fit_min_ctx": 0,
      "n_prompt": 244,
      "n_gen": 0,
      "n_depth": 0,
      "test_time": "2026-09-06T06:05:46Z",
      "avg_ns": 22201842,
      "stddev_ns": 2251034,
      "avg_ts": 11074.060671,
      "stddev_ts": 923.746492,
      "samples_ns": [
        28420555,
        20127953,
        21393639,
        22147573,
        22030748,
        21659923,
        21553112,
        21590668,
        21540114,
        21554138
      ],
      "samples_ts": [
        8585.34,
        12122.4,
        11405.3,
        11017,
        11075.4,
        11265,
        11320.9,
        11301.2,
        11327.7,
        11320.3
      ]
    }
  },
  {
    "repeat": 5,
    "position": 1,
    "tool": "gwenland",
    "p50_tps": 11211.970817647742,
    "p90_tps": 11302.146887360106,
    "p99_tps": 11417.87760718295,
    "p50_ms": 21.762476,
    "p90_ms": 22.3449063,
    "p99_ms": 22.53747813,
    "max_ms": 22.558875,
    "samples_tps": [
      10816.142205672933,
      11263.24908915397,
      10935.677643831164,
      11287.8591441721,
      11430.736576052157,
      11223.224789417347,
      11200.716845878136,
      11135.09822206252,
      10931.345238225373,
      11241.020141005698
    ],
    "samples_ms": [
      22.558875,
      21.663376000000003,
      22.312289,
      21.616145,
      21.345956,
      21.740631999999998,
      21.78432,
      21.912694,
      22.321132000000002,
      21.706215
    ],
    "oracle": "first-token exact; 29/50 decode prefix"
  }
]
```
