"""Build the self-contained Wave 112 three-arm T4 Q8 head-to-head notebook."""

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
BASE_REV = "bd5c956bafb3bb6738c3f1de348a4ebb55d9c29f"
HEAD_REV = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()
SOURCE = ROOT / "notebooks/glcuda_t4_wave111_residual_fusion.ipynb"
H2H_SOURCE = ROOT / "notebooks/glcuda_t4_wave57_h2h_q8.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave112_h2h_q8.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new)


patch = subprocess.check_output(
    [
        "git", "diff", "--binary", "--full-index", BASE_REV, HEAD_REV, "--",
        "Cargo.toml", "Cargo.lock", "glcore", "glproc", "glcuda",
    ],
    cwd=ROOT,
)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

wave111 = json.loads(SOURCE.read_text(encoding="utf-8"))
bootstrap = "".join(wave111["cells"][1]["source"])
bootstrap = re.sub(r'BASE_REV = "[0-9a-f]+"', f'BASE_REV = "{BASE_REV}"', bootstrap, count=1)
bootstrap = re.sub(r'SOURCE_REV = "[0-9a-f]+"', f'SOURCE_REV = "{HEAD_REV}"', bootstrap, count=1)
bootstrap = re.sub(r'PATCH_SHA256 = "[0-9a-f]+"', f'PATCH_SHA256 = "{patch_sha}"', bootstrap, count=1)
bootstrap = re.sub(
    r'PATCH_GZIP_B64 = """.*?"""', f'PATCH_GZIP_B64 = """{patch_b64}"""',
    bootstrap, count=1, flags=re.S,
)
bootstrap = bootstrap.replace("Wave 111", "Wave 112").replace("WAVE111", "WAVE112")
bootstrap = bootstrap.replace("wave111", "wave112")
bootstrap = bootstrap.replace(
    "glcuda-t4-wave112-residual-fusion-results.zip",
    "glcuda-t4-wave112-h2h-q8-results.zip",
)
bootstrap = replace_once(
    bootstrap,
    "import json\nimport os\n",
    "import json\nimport math\nimport os\nimport statistics\nimport time\n",
)
bootstrap = replace_once(
    bootstrap,
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave112-h2h-q8-results.zip")\n',
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave112-h2h-q8-results.zip")\n'
    'HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"\n'
    'HF_REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"\n'
    'HF_FILENAME = "qwen2.5-0.5b-instruct-q8_0.gguf"\n'
    'HF_EXPECTED_BYTES = 675710816\n'
    'HF_EXPECTED_SHA256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"\n'
    'COLD_ITERS = 1\nWARMUP_ITERS = 5\nMEASURE_ITERS = 10\nPRODUCTION_REPEATS = 10\n',
)

# The odd-looking explicit search keeps this generator tied to the verified Wave 111 template.
start = bootstrap.index('    build = run([cargo, "build", "--release", "-p", "glcuda"')
end = bootstrap.rindex("except Exception:\n    fail(phase)")
replacement = '''    parity = run([
        cargo, "test", "--release", "-p", "glcuda", "--test", "parity",
        "--locked", "--", "--nocapture", "--test-threads=1",
    ], cwd=TREE, env=common, timeout=14400, check=False)
    save("cargo-cuda-parity.log", parity)
    parity_summary = next((x for x in (parity.stdout + parity.stderr).splitlines()
                           if x.startswith("test result:")), "")
    if parity.returncode or "0 failed" not in parity_summary:
        raise RuntimeError(f"full CUDA parity failed: {parity_summary}")

    build = run([
        cargo, "build", "--release", "-p", "glbench", "--locked",
    ], cwd=TREE, env=common, timeout=7200, check=False)
    save("cargo-build-glbench.log", build)
    if build.returncode:
        raise RuntimeError("Wave 112 glbench release build failed")
    CARGO = cargo
    CARGO_ENV = cargo_env
    GLBENCH = TARGET / "release/glbench"
    PARITY = {"summary": parity_summary, "returncode": parity.returncode}
    BOOTSTRAP_OK = True
    print("WAVE112_BOOTSTRAP_OK", json.dumps(PARITY), flush=True)
'''
bootstrap = bootstrap[:start] + replacement + bootstrap[end:]
ast.parse(bootstrap)

old = json.loads(H2H_SOURCE.read_text(encoding="utf-8"))
model = "".join(old["cells"][4]["source"])
model = model.replace("STACK_OK", "BOOTSTRAP_OK").replace("WORK", "ROOT")
model = model.replace("Wave57-Q8-H2H", "Wave112-Q8-H2H")
ast.parse(model)

llama_build = "".join(old["cells"][6]["source"])
llama_build = llama_build.replace("WORK", "ROOT").replace("wave57", "wave112")
ast.parse(llama_build)
llama_smoke = "".join(old["cells"][8]["source"])
ast.parse(llama_smoke)

old_prod = "".join(old["cells"][10]["source"])
tokenizer_start = old_prod.index("# Exact effective-prompt gate.")
tokenizer_end_marker = 'print("exact-text tokenizer cross-check:", TOKENIZER_EXACT)'
tokenizer_end = old_prod.index(tokenizer_end_marker, tokenizer_start) + len(tokenizer_end_marker)
tokenizer = old_prod[tokenizer_start:tokenizer_end].replace("wave57", "wave112")

production_prefix = r'''if not globals().get("MODEL_OK") or not globals().get("LLAMA_OK"):
    raise RuntimeError("model/llama gate did not pass")

prompt_unit = "Measure this deterministic systems prompt carefully. Explain how token-parallel integer matrix multiplication uses shared memory, Tensor Cores, and fixed launch geometry. "
FIXED_PROMPT = prompt_unit * 8
CHATML_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n" + FIXED_PROMPT + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
COMMON_ENV = {
    **CARGO_ENV,
    "CUDA_VISIBLE_DEVICES": "0",
    "CARGO_TARGET_DIR": str(TARGET),
    "GLCUDA_FORCE_Q8": "1", "GLCUDA_GRID2D": "1",
    "GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",
    "GLCUDA_BSTAGE": "1", "GLCUDA_GEMM_N16": "1",
    "GLCUDA_GEMM_N16_PREFETCH": "1",
    "GLCUDA_ATTN_MMA4": "1", "GLCUDA_ATTN_MMA4_REGQ": "1",
    "GLCUDA_ATTN_MMA4_AV": "1",
}
ARM_ENV = {
    "gwen_retained": {},
    "gwen_wave111": {"GLCUDA_DEFER_FFN_RESIDUAL": "1"},
}
ORDERS = [
    ["gwen_retained", "gwen_wave111", "llamacpp"],
    ["gwen_wave111", "llamacpp", "gwen_retained"],
    ["llamacpp", "gwen_retained", "gwen_wave111"],
    ["gwen_retained", "llamacpp", "gwen_wave111"],
    ["llamacpp", "gwen_wave111", "gwen_retained"],
    ["gwen_wave111", "gwen_retained", "llamacpp"],
    ["gwen_retained", "gwen_wave111", "llamacpp"],
    ["gwen_wave111", "llamacpp", "gwen_retained"],
    ["llamacpp", "gwen_retained", "gwen_wave111"],
    ["gwen_retained", "llamacpp", "gwen_wave111"],
]
assert len(ORDERS) == PRODUCTION_REPEATS
assert all(sum(order.count(arm) for order in ORDERS) == 10 for arm in (*ARM_ENV, "llamacpp"))

def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] * (hi - index) + values[hi] * (index - lo)

def json_lines(hay, prefix):
    return [json.loads(x) for x in re.findall(re.escape(prefix) + r"\s*(\{[^\n]+\})", hay)]

def last_json_line(hay, prefix):
    rows = json_lines(hay, prefix)
    if not rows:
        raise RuntimeError(f"dispatch line missing: {prefix}")
    return rows[-1]

def run_gwenland(arm, out, cold, warmup, iters):
    cmd = [
        GLBENCH, "run", "--engine", "glcuda", "--model", MODEL_PATH,
        "--prompt", FIXED_PROMPT, "--tokens", "1",
        "--cold-iters", str(cold), "--warmup", str(warmup), "--iters", str(iters),
        "--temperature", "0", "--seed", "42", "--kind", "prefill",
        "--verify-against", "glproc", "--out", out,
    ]
    return run(cmd, cwd=TREE, env={**COMMON_ENV, **ARM_ENV[arm]}, timeout=14400, check=False)

def gate_dispatch(arm, hay):
    contract = last_json_line(hay, "[glcuda-contract]")
    expected = {
        "exact_fusion": True, "defer_ffn_residual": arm == "gwen_wave111",
        "gqa_group": False, "grid2d": True, "r256": False,
        "ntile128": True, "bstage": True, "gemm_n16": True,
        "gemm_n32": False, "gemm_n16_prefetch": True,
        "attn_rows_forced": False, "gqa7_chains": 1,
        "attn_mma4": True, "attn_mma4_regq": True, "attn_mma4_av": True,
    }
    bad = {key: (contract.get(key), value) for key, value in expected.items()
           if contract.get(key) != value}
    attention = last_json_line(hay, "[glcuda-attn]")
    gemm = json_lines(hay, "[glcuda-gemm]")
    if bad or attention.get("path") != "mma4-regq-avmma" or attention.get("ntok") != 244:
        raise RuntimeError(f"{arm} dispatch mismatch: bad={bad}, attention={attention}")
    expected_paths = {"bstage-n16-prefetch", "bstage-n16-m32"}
    if {row.get("path") for row in gemm} != expected_paths or any(row.get("ntok") != 244 for row in gemm):
        raise RuntimeError(f"{arm} GEMM dispatch mismatch: {gemm}")
    return {"contract": contract, "attention": attention, "gemm": gemm}

def gwen_stats(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    engine, workload = data.get("engine") or {}, data.get("workload") or {}
    if engine.get("name") != "glcuda" or engine.get("backend") != "cuda" or not engine.get("available"):
        raise RuntimeError(f"wrong GwenLand engine: {engine}")
    expected = {
        "engine": "glcuda", "kind": "prefill", "prompt": FIXED_PROMPT,
        "seed": 42, "temperature": 0.0, "max_new_tokens": 1,
        "cold_iters": COLD_ITERS, "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS, "verify_against": "glproc",
    }
    bad = {key: (workload.get(key), value) for key, value in expected.items()
           if workload.get(key) != value}
    if bad:
        raise RuntimeError(f"GwenLand workload drift: {bad}")
    validation = data.get("validation") or {}
    parity = [x for x in validation.get("findings", []) if x.get("check") == "parity"]
    match = re.search(r"(\d+)/(\d+) tokens match oracle", parity[-1].get("message", "")) if parity else None
    if validation.get("passed") is not True or not match or tuple(map(int, match.groups())) != (50, 50):
        raise RuntimeError(f"GwenLand full oracle failed: {validation}")
    rows = (data.get("measurements") or {}).get("iterations") or []
    cold = (data.get("measurements") or {}).get("cold") or []
    if len(rows) != MEASURE_ITERS or len(cold) != COLD_ITERS:
        raise RuntimeError(f"GwenLand sample-count drift: measured={len(rows)}, cold={len(cold)}")
    counts = [int(x.get("prompt_tokens", 0)) for x in rows]
    latency_ms = [float(x.get("prefill_ms", 0)) for x in rows]
    if len(set(counts)) != 1 or counts[0] != 244 or not all(math.isfinite(x) and x > 0 for x in latency_ms):
        raise RuntimeError(f"GwenLand timing contract failed: counts={counts}, latency={latency_ms}")
    tps = [244000.0 / x for x in latency_ms]
    return {
        "p50_tps": percentile(tps, .5), "p90_tps": percentile(tps, .9),
        "p99_tps": percentile(tps, .99), "p50_ms": percentile(latency_ms, .5),
        "p90_ms": percentile(latency_ms, .9), "p99_ms": percentile(latency_ms, .99),
        "max_ms": max(latency_ms), "samples_tps": tps, "samples_ms": latency_ms,
        "oracle": "50/50 tokens match oracle", "raw_validation": validation,
    }

def run_llamacpp(tag):
    cmd = [LLAMA_BENCH, "-m", MODEL_PATH, "-p", "244", "-n", "0",
           "-r", str(MEASURE_ITERS), "-ngl", "99", "-o", "json"]
    rc, out, err = sh(cmd, timeout=14400, env=LB_ENV)
    save_command_log(f"llamabench-{tag}.log", cmd, rc, out, err)
    if rc:
        raise RuntimeError(f"llama-bench failed at {tag}")
    payload = json.loads(out)
    rows = [row for row in payload if int(row.get("n_prompt", 0)) == 244 and int(row.get("n_gen", -1)) == 0]
    if len(rows) != 1:
        raise RuntimeError(f"llama-bench row mismatch: {payload}")
    row = rows[0]
    samples_tps = [float(x) for x in row.get("samples_ts", [])]
    samples_ns = [int(x) for x in row.get("samples_ns", [])]
    if (len(samples_tps) != MEASURE_ITERS or len(samples_ns) != MEASURE_ITERS or
            row.get("backends") != "CUDA" or row.get("gpu_info") != "Tesla T4" or
            int(row.get("n_gpu_layers", 0)) != 99 or
            not str(row.get("build_commit", "")).startswith(LLAMA_COMMIT[:7])):
        raise RuntimeError(f"llama-bench contract failed: {row}")
    latency_ms = [x / 1_000_000.0 for x in samples_ns]
    if not all(math.isfinite(x) and x > 0 for x in samples_tps + latency_ms):
        raise RuntimeError(f"llama-bench invalid samples: {row}")
    return {
        "p50_tps": percentile(samples_tps, .5), "p90_tps": percentile(samples_tps, .9),
        "p99_tps": percentile(samples_tps, .99), "p50_ms": percentile(latency_ms, .5),
        "p90_ms": percentile(latency_ms, .9), "p99_ms": percentile(latency_ms, .99),
        "max_ms": max(latency_ms), "samples_tps": samples_tps,
        "samples_ms": latency_ms, "raw": row,
    }

DISPATCH = {}
PROMPT_TOKENS = None
for arm in ARM_ENV:
    probe_path = RESULTS / f"{arm}-probe.json"
    p = run_gwenland(arm, probe_path, 0, 1, 1)
    save_log(f"{arm}-probe.log", p)
    if p.returncode:
        fail_with_archive("h2h-probe", f"{arm} production probe failed")
    DISPATCH[arm] = gate_dispatch(arm, p.stdout + "\n" + p.stderr)
    probe_data = json.loads(probe_path.read_text(encoding="utf-8"))
    count = int(probe_data["measurements"]["iterations"][0]["prompt_tokens"])
    if count != 244:
        fail_with_archive("h2h-probe", f"{arm} prompt contract drift: {count} != 244")
    PROMPT_TOKENS = count
'''

production_tail = r'''

RECORDS = []
try:
    for repeat, order in enumerate(ORDERS):
        for position, arm in enumerate(order):
            if arm in ARM_ENV:
                out_path = RESULTS / f"{arm}-r{repeat}-p{position}.json"
                p = run_gwenland(arm, out_path, COLD_ITERS, WARMUP_ITERS, MEASURE_ITERS)
                save_log(f"{arm}-r{repeat}-p{position}.log", p)
                if p.returncode:
                    raise RuntimeError(f"{arm} failed at repeat {repeat}")
                gate_dispatch(arm, p.stdout + "\n" + p.stderr)
                stats = gwen_stats(out_path)
            else:
                stats = run_llamacpp(f"r{repeat}-p{position}")
                (RESULTS / f"llamacpp-r{repeat}-p{position}.json").write_text(
                    json.dumps(stats["raw"], indent=2), encoding="utf-8")
            RECORDS.append({"repeat": repeat, "position": position, "arm": arm, **stats})
            print(f"{arm:14s} r{repeat} p{position}: {stats['p50_tps']:8.1f} tok/s | "
                  f"{stats['p50_ms']:.3f}/{stats['p90_ms']:.3f}/{stats['p99_ms']:.3f} ms", flush=True)
except Exception:
    (RESULTS / "h2h-partial.json").write_text(json.dumps(RECORDS, indent=2), encoding="utf-8")
    fail_with_archive("h2h-production", traceback.format_exc())

SUMMARY = {}
for arm in (*ARM_ENV, "llamacpp"):
    rows = [row for row in RECORDS if row["arm"] == arm]
    SUMMARY[arm] = {
        "p50_tps": statistics.median(row["p50_tps"] for row in rows),
        "p90_tps": statistics.median(row["p90_tps"] for row in rows),
        "p99_tps": statistics.median(row["p99_tps"] for row in rows),
        "p50_ms": statistics.median(row["p50_ms"] for row in rows),
        "p90_ms": statistics.median(row["p90_ms"] for row in rows),
        "p99_ms": statistics.median(row["p99_ms"] for row in rows),
        "max_ms": statistics.median(row["max_ms"] for row in rows),
        "min_session_p50_tps": min(row["p50_tps"] for row in rows),
        "max_session_p50_tps": max(row["p50_tps"] for row in rows),
        "positions": [row["position"] for row in rows], "sessions": len(rows),
    }

PAIRED = []
for repeat in range(PRODUCTION_REPEATS):
    rows = {row["arm"]: row for row in RECORDS if row["repeat"] == repeat}
    PAIRED.append({
        "repeat": repeat,
        "wave111_over_retained": rows["gwen_wave111"]["p50_tps"] / rows["gwen_retained"]["p50_tps"] - 1.0,
        "wave111_over_llamacpp": rows["gwen_wave111"]["p50_tps"] / rows["llamacpp"]["p50_tps"] - 1.0,
        "wave111_minus_llamacpp_tps": rows["gwen_wave111"]["p50_tps"] - rows["llamacpp"]["p50_tps"],
    })

retained = SUMMARY["gwen_retained"]
candidate = SUMMARY["gwen_wave111"]
llama = SUMMARY["llamacpp"]
internal_delta = candidate["p50_tps"] / retained["p50_tps"] - 1.0
h2h_delta = candidate["p50_tps"] / llama["p50_tps"] - 1.0
all_positive = all(row["wave111_over_retained"] > 0 for row in PAIRED)
tail_delta = candidate["max_ms"] / retained["max_ms"] - 1.0
retention_ok = internal_delta > 0 and all_positive and tail_delta <= 0.05
target_reached = candidate["p50_tps"] >= 15000.0
decision = "GOAL_REACHED" if retention_ok and target_reached else (
    "RETAIN_BELOW_GOAL" if retention_ok else "REJECT_WAVE111_PRODUCTION"
)

RESULT = {
    "wave": 112, "status": "production_measured", "gpu": fields,
    "source_revision": SOURCE_REV, "patch_sha256": PATCH_SHA256,
    "model": MODEL_META, "llama_commit": LLAMA_COMMIT,
    "method": {
        "quant": "Q8_0 vs Q8_0", "prompt_tokens": 244,
        "gwen_oracle": "50/50 every session", "repeats_per_arm": 10,
        "total_sessions": 30, "samples_per_session": 10,
        "cold_iters": COLD_ITERS, "warmup_iters": WARMUP_ITERS,
        "order": ORDERS, "seed": 42, "temperature": 0.0,
        "llama_prompt_note": "llama-bench uses synthetic IDs at the tokenizer-verified 244-token shape",
    },
    "summary": SUMMARY, "paired": PAIRED,
    "comparison": {
        "wave111_over_retained": internal_delta,
        "wave111_over_llamacpp": h2h_delta,
        "wave111_minus_llamacpp_tps": candidate["p50_tps"] - llama["p50_tps"],
        "all_internal_pairs_positive": all_positive,
        "tail_max_delta": tail_delta, "retention_ok": retention_ok,
    },
    "tokenizer_exact": TOKENIZER_EXACT, "dispatch": DISPATCH,
    "decision": decision,
    "target_15000_tps_achieved": retention_ok and target_reached,
}
(RESULTS / "production-records.json").write_text(json.dumps(RECORDS, indent=2), encoding="utf-8")
(RESULTS / "wave112-h2h-q8.json").write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
(RESULTS / "PRODUCTION_SUCCESS.json").write_text(
    json.dumps({"status": "valid", "decision": decision}, indent=2), encoding="utf-8")
archive()
print("WAVE112_RESULT", json.dumps(RESULT, indent=2), flush=True)
'''

production = production_prefix + "\n" + tokenizer + production_tail
ast.parse(production)

def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


notebook = {
    "cells": [
        md("# GwenLand glcuda Wave 112 — Q8 head-to-head on Tesla T4\n\n"
           "Thirty sessions: ten retained GwenLand, ten Wave111, ten pinned llama.cpp.\n"),
        py(bootstrap),
        md("## Fetch and verify the exact Q8_0 model\n"), py(model),
        md("## Build pinned stock-CUDA llama.cpp\n"), py(llama_build),
        md("## Reject CPU, multi-GPU, and revision drift\n"), py(llama_smoke),
        md("## Exact tokenizer gate and 30-session production comparison\n"), py(production),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")

round_trip = json.loads(OUTPUT.read_text(encoding="utf-8"))
for cell in round_trip["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
decoded_match = re.search(
    r'PATCH_GZIP_B64 = """(.*?)"""', "".join(round_trip["cells"][1]["source"]), re.S
)
if decoded_match is None:
    raise RuntimeError("generated notebook has no embedded patch")
decoded = gzip.decompress(base64.b64decode(decoded_match.group(1)))
if decoded != patch or hashlib.sha256(decoded).hexdigest() != patch_sha:
    raise RuntimeError("generated notebook embedded patch mismatch")
print(json.dumps({
    "notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
    "base_rev": BASE_REV, "head_rev": HEAD_REV, "patch_bytes": len(patch),
    "patch_sha256": patch_sha, "sessions": 30, "runs_per_arm": 10,
}, indent=2))
