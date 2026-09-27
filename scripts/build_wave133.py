"""Build the self-contained Wave 133 captured-decode profile notebook."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE_REV = subprocess.check_output(
    ["git", "merge-base", "HEAD", "github/main"], cwd=ROOT, text=True
).strip()
SOURCE_REV = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
SOURCE = ROOT / "notebooks/glcuda_t4_wave132_decode_q8_glue.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave133_decode_graph_profile.ipynb"
STAGING = ROOT / ".kaggle-wave133"

patch = subprocess.check_output(
    ["git", "diff", "--binary", "--full-index", BASE_REV, "--", "glcuda"],
    cwd=ROOT,
)
if not patch:
    raise RuntimeError("Wave 133 patch is empty")
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace('BUILD = "wave131-fused-swiglu-production-ceiling-v1"',
                    'BUILD = "wave133-captured-decode-profile-v1"')
code = code.replace('ROOT = Path("/tmp/wave132")', 'ROOT = Path("/tmp/wave133")')
code = code.replace("glcuda-t4-wave132-decode-q8-glue-results.zip",
                    "glcuda-t4-wave133-decode-graph-profile-results.zip")
code = re.sub(r'BASE_REV = "[0-9a-f]+"', f'BASE_REV = "{BASE_REV}"', code, count=1)
code = re.sub(r'SOURCE_REV = "[0-9a-f]+"', f'SOURCE_REV = "{SOURCE_REV}"', code, count=1)
code = re.sub(r'PATCH_SHA256 = "[0-9a-f]+"', f'PATCH_SHA256 = "{patch_sha}"', code, count=1)
code = re.sub(
    r'PATCH_GZIP_B64 = """.*?"""',
    f'PATCH_GZIP_B64 = """{patch_b64}"""',
    code,
    count=1,
    flags=re.S,
)
code = code.replace('patch_path = RESULTS / "wave131.patch"',
                    'patch_path = RESULTS / "wave133.patch"', 1)
code = code.replace('"74 passed"', '"75 passed"', 1)

graph_anchor = '''    if graph.returncode or "4 passed" not in graph.stdout or "0 failed" not in graph.stdout:
        raise RuntimeError("Wave 132 graph replay gate failed")

'''
instrumented_graph = '''    phase = "instrumented-graph-replay"
    profile_graph_env = {**candidate_switch_env, "GLCUDA_PROFILE_DECODE_STAGES": "1"}
    profile_graph = run([cargo, "test", "--release", "-p", "glcuda", "--test", "graph_replay",
                         "--locked", "--", "--nocapture", "--test-threads=1"],
                        cwd=TREE, env=profile_graph_env, check=False)
    save("cargo-instrumented-graph-replay.log", profile_graph)
    profile_graph_log = profile_graph.stdout + "\\n" + profile_graph.stderr
    if (profile_graph.returncode or "4 passed" not in profile_graph.stdout
            or '"source":"cuda_events_inside_captured_graph"' not in profile_graph_log):
        raise RuntimeError("Wave 133 instrumented graph replay gate failed")

'''
if graph_anchor not in code:
    raise RuntimeError("graph replay insertion point moved")
code = code.replace(graph_anchor, graph_anchor + instrumented_graph, 1)

# Wave 133 is a diagnostic profile. Remove the inherited prefill experiment
# and preserve only the compiler, parity, model, and production glbench gates.
code, removed = re.subn(
    r'    phase = "build"\n.*?    phase = "glbench-ceiling"\n',
    '    phase = "glbench-ceiling"\n',
    code,
    count=1,
    flags=re.S,
)
if removed != 1:
    raise RuntimeError("inherited production block moved")

profile_loop = r'''    prod_env = {
        **common,
        "GLCUDA_FORCE_Q8": "1",
        "GLCUDA_GRID2D": "1",
        "GLCUDA_FUSE_Q8_GLUE": "1",
        "GLCUDA_Q8_NOSTORE": "1",
        "GLCUDA_FFN_GATE_UP_STACKED": "1",
        "GLCUDA_DEFER_FFN_RESIDUAL": "1",
        "GLCUDA_NTILE128": "1",
        "GLCUDA_BSTAGE": "1",
        "GLCUDA_GEMM_N16": "1",
        "GLCUDA_GEMM_N16_PREFETCH": "1",
        "GLCUDA_N16_FUSED_SWIGLU": "1",
        "GLCUDA_ATTN_MMA4": "1",
        "GLCUDA_ATTN_MMA4_REGQ": "1",
        "GLCUDA_ATTN_MMA4_AV": "1",
        "GLCUDA_DECODE_Q8_GLUE": "1",
    }
    profile_env = {**prod_env, "GLCUDA_PROFILE_DECODE_STAGES": "1"}
    profile_runs = []
    expected_stages = [
        "attn_input", "qkv", "attn_prepare", "attention", "attn_out",
        "ffn_input", "ffn_gate_up", "ffn_activation", "ffn_down", "lm_head",
    ]
    for index, arm in enumerate(["control", "profile", "profile", "control"]):
        out = RESULTS / f"glbench-{index + 1}-{arm}.json"
        env = profile_env if arm == "profile" else prod_env
        measured = run([
            glbench, "run", "--engine", "glcuda", "--model", MODEL,
            "--prompt", ceiling_prompt, "--tokens", "64", "--cold-iters", "1",
            "--warmup", "2", "--iters", "10", "--temperature", "0", "--seed", "42",
            "--kind", "end_to_end", "--verify-against", "glproc", "--out", out,
        ], cwd=TREE, env=env, check=False)
        save(f"glbench-{index + 1}-{arm}.log", measured)
        if measured.returncode or not out.is_file():
            raise RuntimeError(f"glbench profile run failed: {index + 1} {arm}")
        dispatch_log = measured.stdout + "\n" + measured.stderr
        if ('"path":"wave129-fused-swiglu-production"' not in dispatch_log
                or '"path":"wave132-exact-q8-glue"' not in dispatch_log):
            raise RuntimeError(f"production dispatch missing: {index + 1} {arm}")
        profiler_dispatched = '"source":"cuda_events_inside_captured_graph"' in dispatch_log
        if profiler_dispatched != (arm == "profile"):
            raise RuntimeError(f"profile dispatch mismatch: {index + 1} {arm}")
        session = json.loads(out.read_text(encoding="utf-8"))
        validation_text = json.dumps(session.get("validation", {}))
        if (not session.get("validation", {}).get("passed")
                or "50/50 tokens match oracle" not in validation_text):
            raise RuntimeError(f"oracle gate failed: {index + 1} {arm}")
        decode_profile = session.get("telemetry", {}).get("decode")
        if arm == "profile":
            names = [stage["name"] for stage in decode_profile["stages"]] if decode_profile else []
            if names != expected_stages or decode_profile.get("total_ms", 0) <= 0:
                raise RuntimeError(f"decode profile contract failed: {index + 1} {names}")
        elif decode_profile is not None:
            raise RuntimeError(f"control unexpectedly carried decode telemetry: {index + 1}")
        profile_runs.append({
            "order": index + 1,
            "arm": arm,
            "prefill_tps": session["analysis"]["prefill_tps"],
            "decode_tps": session["analysis"]["decode_tps"],
            "decode_profile": decode_profile,
        })

'''
start = '    base_env = dict(prod_env)\n'
end = '    summary = {\n'
start_at = code.find(start)
end_at = code.find(end, start_at)
if start_at < 0 or end_at < 0:
    raise RuntimeError("Wave 132 glbench block moved")
code = code[:start_at] + profile_loop + code[end_at:]

summary_block = r'''    profile_stages = {}
    for run_record in profile_runs:
        profile = run_record["decode_profile"]
        if not profile:
            continue
        for stage in profile["stages"]:
            profile_stages.setdefault(stage["name"], []).append(stage["total_ms"])
    ranked_stages = sorted(
        [
            {
                "name": name,
                "median_total_ms": statistics.median(values),
                "runs": len(values),
            }
            for name, values in profile_stages.items()
        ],
        key=lambda stage: -stage["median_total_ms"],
    )
    control_decode = [run["decode_tps"]["median"] for run in profile_runs
                      if run["arm"] == "control"]
    instrumented_decode = [run["decode_tps"]["median"] for run in profile_runs
                           if run["arm"] == "profile"]
    summary = {
        "wave": 133,
        "purpose": "captured_decode_stage_profile",
        "retention_authority": False,
        "gpu": fields,
        "model": model_meta,
        "runs": profile_runs,
        "control_decode_p50_tps": statistics.median(control_decode),
        "instrumented_decode_p50_tps": statistics.median(instrumented_decode),
        "instrumentation_slowdown": (
            statistics.median(control_decode) / statistics.median(instrumented_decode)
        ),
        "ranked_decode_stages": ranked_stages,
        "profile_source": "cuda_events_inside_captured_graph",
        "profile_is_perturbing": True,
    }
    report = [
        "# Wave 133 captured decode graph profile",
        "",
        "Diagnostic only: event nodes perturb the graph and are not retention authority.",
        f"- Control decode P50: {summary['control_decode_p50_tps']:.3f} tok/s",
        f"- Instrumented decode P50: {summary['instrumented_decode_p50_tps']:.3f} tok/s",
        f"- Instrumentation slowdown: {summary['instrumentation_slowdown']:.4f}x",
        "",
        "| stage | median total ms | runs |",
        "|---|---:|---:|",
    ]
    for stage in ranked_stages:
        report.append(
            f"| `{stage['name']}` | {stage['median_total_ms']:.6f} | {stage['runs']} |"
        )
    (RESULTS / "wave133-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (RESULTS / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("WAVE133_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
summary_at = code.find('    summary = {\n')
except_at = code.find('except Exception:\n', summary_at)
if summary_at < 0 or except_at < 0:
    raise RuntimeError("Wave 132 summary block moved")
code = code[:summary_at] + summary_block + code[except_at:]

ast.parse(code)
notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 133 - captured decode graph profile\n",
    "\nA-B-B-A production control plus opt-in CUDA-event attribution inside the captured graph.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))

STAGING.mkdir(exist_ok=True)
(STAGING / OUTPUT.name).write_bytes(OUTPUT.read_bytes())
(STAGING / "kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "jinxsuperdev/glcuda-t4-wave133-decode-graph-profile",
            "title": "glcuda T4 Wave133 decode graph profile",
            "code_file": OUTPUT.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps({"notebook": str(OUTPUT), "patch_sha256": patch_sha,
                  "patch_bytes": len(patch), "base_rev": BASE_REV}, indent=2))
