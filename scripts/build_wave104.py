"""Build the Wave 104 event-timed CUDA production-profile notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DIRECT = REPO / "notebooks/glcuda_t4_wave94_n16_prefetch.ipynb"
WAVE80 = REPO / "notebooks/glcuda_t4_wave80_mma_av_production.ipynb"
OUTPUT = REPO / "notebooks/glcuda_t4_wave104_cuda_profile.ipynb"

notebook = json.loads(DIRECT.read_text(encoding="utf-8"))
notebook["cells"][0]["id"] = "wave104-intro"
notebook["cells"][0]["source"] = [
    "# glcuda Wave 104 — event-timed CUDA production profile\n",
    "\n",
    "Five position-balanced Q8_0 profile pairs. Diagnostic only: profiled "
    "throughput is not a production claim.\n",
]

direct_code = "".join(notebook["cells"][1]["source"])
for old, new in (
    ("Wave 101", "Wave 104"),
    ("wave101", "wave104"),
    ("WAVE101", "WAVE104"),
):
    direct_code = direct_code.replace(old, new)
direct_code = direct_code.replace(
    "glcuda-t4-wave104-n16-prefetch-results.zip",
    "glcuda-t4-wave104-cuda-profile-results.zip",
)
ast.parse(direct_code)
notebook["cells"][1]["id"] = "wave104-direct-gate"
notebook["cells"][1]["source"] = direct_code.splitlines(keepends=True)

historical = json.loads(WAVE80.read_text(encoding="utf-8"))
wave80_code = "".join(historical["cells"][-1]["source"])
helpers = wave80_code[wave80_code.index("import math"):wave80_code.index("def percentile")]
helpers = helpers.replace("Wave80", "Wave104").replace("Wave 80", "Wave 104")

profile_code = helpers + r'''
PROFILE_PAIRS = 5
PROFILE_WARMUP = 2
PROFILE_ITERS = 3
EXPECTED_STAGES = {
    "qkv", "attn_norm", "attn_kv_write", "attention",
    "ffn_elementwise", "ffn_down", "ffn_gate_up", "attn_out",
}


def json_lines(haystack, prefix):
    return [
        json.loads(item)
        for item in re.findall(re.escape(prefix) + r"\s*(\{[^\n]+\})", haystack)
    ]


def check_dispatch(arm, haystack):
    contracts = json_lines(haystack, "[glcuda-contract]")
    attention = json_lines(haystack, "[glcuda-attn]")
    gemm = json_lines(haystack, "[glcuda-gemm]")
    if not contracts or not attention or not gemm:
        raise RuntimeError(f"{arm} dispatch evidence missing")
    contract = contracts[-1]
    candidate = arm == "candidate_n16_prefetch"
    required = {
        "exact_fusion": True,
        "grid2d": True,
        "ntile128": True,
        "bstage": True,
        "gemm_n16": True,
        "gemm_n32": False,
        "gemm_n16_prefetch": candidate,
        "attn_mma4": True,
        "attn_mma4_regq": True,
        "attn_mma4_av": True,
    }
    bad = {key: (contract.get(key), value) for key, value in required.items()
           if contract.get(key) is not value}
    if bad:
        raise RuntimeError(f"{arm} contract mismatch: {bad}")
    if attention[-1].get("path") != "mma4-regq-avmma" or attention[-1].get("ntok") != 244:
        raise RuntimeError(f"{arm} attention drift: {attention[-1]}")
    paths = {row.get("path") for row in gemm}
    expected = {
        "bstage-n16-m32",
        "bstage-n16-prefetch" if candidate else "bstage-n16",
    }
    if paths != expected or any(row.get("ntok") != 244 for row in gemm):
        raise RuntimeError(f"{arm} GEMM drift: {gemm}")
    return {"contract": contract, "attention": attention[-1], "gemm": gemm}


def parse_profile(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = data.get("engine") or {}
    if engine.get("name") != "glcuda" or engine.get("backend") != "cuda":
        raise RuntimeError(f"wrong engine: {engine}")
    workload = data.get("workload") or {}
    expected = {
        "kind": "prefill", "prompt": FIXED_PROMPT, "seed": 42,
        "temperature": 0.0, "max_new_tokens": 1,
        "cold_iters": 0, "warmup_iters": PROFILE_WARMUP,
        "measure_iters": PROFILE_ITERS, "verify_against": "glproc",
    }
    for key, value in expected.items():
        if workload.get(key) != value:
            raise RuntimeError(f"workload {key} drift: {workload.get(key)!r}")
    findings = (data.get("validation") or {}).get("findings") or []
    parity = [row for row in findings if row.get("check") == "parity"]
    if not parity or "50/50 tokens match oracle" not in parity[-1].get("message", ""):
        raise RuntimeError(f"oracle gate failed: {parity[-1] if parity else None}")
    prefill = (data.get("telemetry") or {}).get("prefill")
    if not prefill:
        raise RuntimeError("CUDA prefill telemetry is absent")
    stages = prefill.get("stages") or []
    by_name = {row.get("name"): row for row in stages}
    if set(by_name) != EXPECTED_STAGES:
        raise RuntimeError(f"stage set drift: {sorted(by_name)}")
    for name, row in by_name.items():
        if not math.isfinite(float(row.get("total_ms", 0))) or row["total_ms"] <= 0:
            raise RuntimeError(f"invalid stage timing: {name}={row}")
        if int(row.get("calls", 0)) <= 0:
            raise RuntimeError(f"invalid stage calls: {name}={row}")
    iterations = (data.get("measurements") or {}).get("iterations") or []
    if len(iterations) != PROFILE_ITERS:
        raise RuntimeError(f"measured iteration drift: {len(iterations)}")
    prompt_counts = {int(row.get("prompt_tokens", 0)) for row in iterations}
    if prompt_counts != {244}:
        raise RuntimeError(f"prompt-token drift: {prompt_counts}")
    wall = [float(row["prefill_ms"]) for row in iterations]
    return {
        "stages_ms": {name: float(by_name[name]["total_ms"]) for name in sorted(by_name)},
        "calls": {name: int(by_name[name]["calls"]) for name in sorted(by_name)},
        "profile_total_ms": float(prefill["total_ms"]),
        "wall_p50_ms": statistics.median(wall),
        "wall_samples_ms": wall,
        "oracle": "50/50",
    }


try:
    for key in list(os.environ):
        if key.startswith("GLCUDA_"):
            os.environ.pop(key)

    target = ROOT / "target-wave104"
    build_env = {**cargo_env, "CARGO_TARGET_DIR": str(target)}
    build = run(
        [cargo, "build", "--release", "-p", "glbench", "--locked"],
        cwd=TREE, env=build_env, timeout=7200, check=False,
    )
    save("cargo-build-glbench.log", build)
    if build.returncode:
        raise RuntimeError("Wave 104 glbench release build failed")
    glbench = target / "release/glbench"

    model_path = fetch_pinned_model()
    model_meta = {
        "repo": HF_REPO, "revision": HF_REVISION, "filename": HF_FILENAME,
        "bytes": model_path.stat().st_size, "sha256": sha256_file(model_path),
    }
    (RESULTS / "model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    common_env = {
        **cargo_env,
        "CUDA_VISIBLE_DEVICES": "0",
        "GLCUDA_FORCE_Q8": "1",
        "GLCUDA_GRID2D": "1",
        "GLCUDA_FUSE_Q8_GLUE": "1",
        "GLCUDA_NTILE128": "1",
        "GLCUDA_BSTAGE": "1",
        "GLCUDA_GEMM_N16": "1",
        "GLCUDA_ATTN_MMA4": "1",
        "GLCUDA_ATTN_MMA4_REGQ": "1",
        "GLCUDA_ATTN_MMA4_AV": "1",
        "GLCUDA_TELEMETRY": "1",
    }
    arm_env = {
        "retained_n16": {},
        "candidate_n16_prefetch": {"GLCUDA_GEMM_N16_PREFETCH": "1"},
    }
    orders = [
        ["retained_n16", "candidate_n16_prefetch"] if i % 2 == 0
        else ["candidate_n16_prefetch", "retained_n16"]
        for i in range(PROFILE_PAIRS)
    ]

    records = []
    dispatch = {}
    for repeat, order in enumerate(orders):
        for position, arm in enumerate(order):
            output = RESULTS / f"profile-r{repeat}-p{position}-{arm}.json"
            command = [
                glbench, "run", "--engine", "glcuda", "--model", model_path,
                "--prompt", FIXED_PROMPT, "--tokens", "1",
                "--cold-iters", "0", "--warmup", str(PROFILE_WARMUP),
                "--iters", str(PROFILE_ITERS), "--temperature", "0",
                "--seed", "42", "--kind", "prefill",
                "--verify-against", "glproc", "--out", output,
            ]
            process = run(
                command, cwd=TREE, env={**common_env, **arm_env[arm]},
                timeout=7200, check=False,
            )
            save(f"profile-r{repeat}-p{position}-{arm}.log", process)
            if process.returncode:
                raise RuntimeError(f"profile process failed: {arm} r{repeat}")
            dispatch.setdefault(
                arm, check_dispatch(arm, process.stdout + "\n" + process.stderr)
            )
            record = parse_profile(output)
            record.update({"repeat": repeat, "position": position, "arm": arm})
            records.append(record)
            print(
                f"{arm:26s} r{repeat} p{position}: profile "
                f"{record['profile_total_ms']:.3f} ms, wall P50 "
                f"{record['wall_p50_ms']:.3f} ms",
                flush=True,
            )

    summaries = {}
    for arm in arm_env:
        rows = [row for row in records if row["arm"] == arm]
        stage_medians = {
            stage: statistics.median(row["stages_ms"][stage] for row in rows)
            for stage in sorted(EXPECTED_STAGES)
        }
        total = sum(stage_medians.values())
        summaries[arm] = {
            "stage_median_ms": stage_medians,
            "stage_share": {stage: value / total for stage, value in stage_medians.items()},
            "sum_of_stage_medians_ms": total,
            "profile_total_median_ms": statistics.median(row["profile_total_ms"] for row in rows),
            "wall_p50_median_ms": statistics.median(row["wall_p50_ms"] for row in rows),
            "sessions": len(rows),
            "positions": [row["position"] for row in rows],
        }

    paired_stage_delta_ms = {}
    for stage in sorted(EXPECTED_STAGES):
        deltas = []
        for repeat in range(PROFILE_PAIRS):
            retained = next(row for row in records if row["repeat"] == repeat and row["arm"] == "retained_n16")
            candidate = next(row for row in records if row["repeat"] == repeat and row["arm"] == "candidate_n16_prefetch")
            deltas.append(candidate["stages_ms"][stage] - retained["stages_ms"][stage])
        paired_stage_delta_ms[stage] = {
            "median": statistics.median(deltas), "minimum": min(deltas), "maximum": max(deltas),
        }

    result = {
        "wave": 104,
        "status": "cuda_profile_complete",
        "diagnostic_only": True,
        "gpu": gpu.stdout.splitlines()[0],
        "model": model_meta,
        "method": {
            "pairs": PROFILE_PAIRS, "sessions": len(records),
            "warmup_iters": PROFILE_WARMUP, "measure_iters": PROFILE_ITERS,
            "position_balanced": True, "quant": "Q8_0", "prompt_tokens": 244,
            "telemetry": "GLCUDA_TELEMETRY event timing",
        },
        "summary": summaries,
        "paired_stage_delta_ms": paired_stage_delta_ms,
        "dispatch": dispatch,
        "direct_gate": verdict,
    }
    (RESULTS / "profile-records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (RESULTS / "wave104-profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS / "PROFILE_SUCCESS.json").write_text(
        json.dumps({"status": "valid", "sessions": len(records)}, indent=2), encoding="utf-8"
    )
    print("WAVE104_PROFILE", json.dumps(result), flush=True)
    archive()
except Exception:
    (RESULTS / "PROFILE_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
    archive()
    raise
'''

required = (
    '"GLCUDA_TELEMETRY": "1"',
    'PROFILE_PAIRS = 5',
    '"quant": "Q8_0"',
    '"prompt_tokens": 244',
    '"50/50 tokens match oracle"',
    'EXPECTED_STAGES',
)
missing = [needle for needle in required if needle not in profile_code]
if missing:
    raise RuntimeError(f"Wave 104 profile contract missing: {missing}")
ast.parse(profile_code)
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "id": "wave104-cuda-profile",
    "metadata": {},
    "outputs": [],
    "source": profile_code.splitlines(keepends=True),
})

OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
round_trip = json.loads(OUTPUT.read_text(encoding="utf-8"))
for cell in round_trip["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
print(json.dumps({
    "notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
    "cells": len(round_trip["cells"]), "profile_pairs": 5, "profile_sessions": 10,
}, indent=2))
