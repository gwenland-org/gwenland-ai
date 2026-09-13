"""Build the self-contained Wave 118 in-process T4 stability notebook."""

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
BASE_REV = "5de5be39c0190b9367da18f0f2001e7f40208c92"
HEAD_REV = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
OUTPUT = ROOT / "notebooks/glcuda_t4_wave118_in_process_stability.ipynb"

patch = subprocess.check_output(
    [
        "git", "diff", "--binary", "--full-index", BASE_REV, "--",
        "Cargo.toml", "Cargo.lock", "glcore", "glproc", "glcuda",
    ],
    cwd=ROOT,
)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

code = r'''import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import traceback
import urllib.request
import zipfile

BUILD = "wave118-in-process-stability-v1"
REPO_URL = "https://github.com/gwenland-org/gwenland-ai.git"
BASE_REV = "__BASE_REV__"
SOURCE_REV = "__HEAD_REV__"
PATCH_SHA256 = "__PATCH_SHA__"
PATCH_GZIP_B64 = """__PATCH_B64__"""
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q8_0.gguf?download=true"
MODEL_BYTES = 675710816
MODEL_SHA256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
ROOT = Path("/kaggle/working/wave118")
TREE = ROOT / "repo"
RESULTS = ROOT / "results"
TARGET = ROOT / "target"
MODEL = ROOT / "qwen2.5-0.5b-instruct-q8_0.gguf"
FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave118-in-process-stability-results.zip")

if ROOT.exists():
    shutil.rmtree(ROOT)
RESULTS.mkdir(parents=True)

def run(cmd, *, cwd=None, env=None, timeout=14400, check=True):
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, env=merged, text=True,
                       capture_output=True, timeout=timeout)
    print("$", " ".join(str(x) for x in cmd), flush=True)
    if p.stdout:
        print(p.stdout[-12000:], flush=True)
    if p.stderr:
        print(p.stderr[-12000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p

def save(name, p):
    (RESULTS / name).write_text(
        f"RETURN_CODE {p.returncode}\n\nSTDOUT\n{p.stdout}\n\nSTDERR\n{p.stderr}",
        encoding="utf-8",
    )

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def archive():
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(RESULTS.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(RESULTS))
    digest = sha256_file(FINAL_ZIP)
    print("ARCHIVE", FINAL_ZIP, digest, flush=True)
    return digest

def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)

def bootstrap_ci(values, seed=118, draws=20000):
    rng = random.Random(seed)
    n = len(values)
    medians = [statistics.median(values[rng.randrange(n)] for _ in range(n))
               for _ in range(draws)]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]

phase = "bootstrap"
try:
    embedded = gzip.decompress(base64.b64decode(PATCH_GZIP_B64))
    if hashlib.sha256(embedded).hexdigest() != PATCH_SHA256:
        raise RuntimeError("embedded patch hash mismatch")
    patch_path = RESULTS / "wave118.patch"
    patch_path.write_bytes(embedded)
    (RESULTS / "source.json").write_text(json.dumps({
        "build": BUILD, "base_rev": BASE_REV, "source_rev": SOURCE_REV,
        "patch_sha256": PATCH_SHA256, "patch_bytes": len(embedded),
    }, indent=2), encoding="utf-8")

    gpu = run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version",
               "--format=csv,noheader,nounits"], timeout=60)
    save("nvidia-smi.log", gpu)
    fields = [x.strip() for x in gpu.stdout.splitlines()[0].split(",")]
    if len(fields) < 5 or fields[1] != "Tesla T4" or fields[2] != "7.5":
        raise RuntimeError(f"requires Tesla T4 sm_75, got {fields}")

    phase = "reconstruct"
    clone = run(["git", "clone", "--filter=blob:none", REPO_URL, TREE], timeout=1800)
    save("git-clone.log", clone)
    checkout = run(["git", "checkout", "--detach", BASE_REV], cwd=TREE, timeout=600)
    save("git-checkout.log", checkout)
    applied = run(["git", "apply", "--whitespace=error", patch_path], cwd=TREE)
    save("git-apply.log", applied)
    diff = run(["git", "diff", "--check"], cwd=TREE)
    save("git-diff-check.log", diff)

    cargo_candidates = [shutil.which("cargo"), Path.home() / ".cargo/bin/cargo",
                        "/usr/local/cargo/bin/cargo", "/opt/conda/bin/cargo"]
    cargo = next((str(x) for x in cargo_candidates if x and Path(x).is_file()), None)
    cargo_env = {}
    bootstrapped = False
    if cargo is None:
        bootstrapped = True
        rustup_script = ROOT / "rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", rustup_script)
        cargo_home = ROOT / "cargo-home"
        rustup_home = ROOT / "rustup-home"
        cargo_env = {"CARGO_HOME": cargo_home, "RUSTUP_HOME": rustup_home}
        install = run(["bash", rustup_script, "-y", "--profile", "minimal",
                       "--default-toolchain", "stable", "--no-modify-path"],
                      env=cargo_env, timeout=1800)
        save("rustup-install.log", install)
        cargo = str(cargo_home / "bin/cargo")
    if not Path(cargo).is_file():
        raise RuntimeError(f"cargo unavailable after bootstrap: {cargo}")
    (RESULTS / "cargo-discovery.json").write_text(json.dumps({
        "selected": cargo, "bootstrapped": bootstrapped,
        "candidates": [str(x) for x in cargo_candidates if x],
    }, indent=2), encoding="utf-8")
    common = {**cargo_env, "CARGO_TARGET_DIR": TARGET, "CUDA_VISIBLE_DEVICES": "0"}

    phase = "host-tests"
    tests = run([cargo, "test", "-p", "glcuda", "--lib", "--locked"], cwd=TREE, env=common)
    save("cargo-lib-tests.log", tests)
    if "67 passed" not in tests.stdout or "0 failed" not in tests.stdout:
        raise RuntimeError("unexpected host test summary")

    phase = "cuda-parity"
    parity = run([cargo, "test", "--release", "-p", "glcuda", "--test", "parity",
                  "--locked", "--", "--nocapture", "--test-threads=1"],
                 cwd=TREE, env=common, check=False)
    save("cargo-cuda-parity.log", parity)
    if parity.returncode or "0 failed" not in parity.stdout:
        raise RuntimeError("CUDA parity failed")

    phase = "compiler-resource"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    resource = run([ptxas, "-v", "-arch=sm_75", TREE / "glcuda/src/kernels/glcuda_sm75.ptx",
                    "-o", ROOT / "wave118.cubin"], check=False)
    save("ptxas-sm75.log", resource)
    if resource.returncode or "spill stores" not in resource.stderr:
        raise RuntimeError("ptxas resource gate failed")

    phase = "model"
    urllib.request.urlretrieve(MODEL_URL, MODEL)
    model_meta = {"bytes": MODEL.stat().st_size, "sha256": sha256_file(MODEL)}
    if model_meta != {"bytes": MODEL_BYTES, "sha256": MODEL_SHA256}:
        raise RuntimeError(f"model identity mismatch: {model_meta}")
    (RESULTS / "model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    phase = "build"
    build = run([cargo, "build", "--release", "-p", "glcuda", "--example",
                 "wave118_in_process_stability", "--locked"], cwd=TREE, env=common)
    save("cargo-build.log", build)
    exe = TARGET / "release/examples/wave118_in_process_stability"
    prod_env = {**common, "GLCUDA_FORCE_Q8": "1", "GLCUDA_GRID2D": "1",
                "GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",
                "GLCUDA_BSTAGE": "1", "GLCUDA_GEMM_N16": "1",
                "GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_ATTN_MMA4": "1",
                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1"}

    phase = "production"
    records = []
    invocation_rows = {}
    for invocation in ("a", "b"):
        measured = run([exe, MODEL, invocation], cwd=TREE, env=prod_env, check=False)
        save(f"production-{invocation}.log", measured)
        if measured.returncode:
            raise RuntimeError(f"invocation {invocation} failed")
        rows = [json.loads(x) for x in re.findall(
            r"\[wave118-sample\]\s*(\{[^\n]+\})", measured.stdout)]
        if len(rows) != 200 or any(x["prompt_tokens"] != 244 for x in rows):
            raise RuntimeError(f"invocation {invocation} sample contract failed: {len(rows)}")
        invocation_rows[invocation] = rows
        records.extend(rows)

    paired = []
    for invocation, rows in invocation_rows.items():
        for quartet in range(50):
            q = [x for x in rows if x["quartet"] == quartet]
            retained = [x["prefill_ms"] for x in q if x["arm"] == "retained"]
            candidate = [x["prefill_ms"] for x in q if x["arm"] == "candidate"]
            if len(retained) != 2 or len(candidate) != 2:
                raise RuntimeError("ABBA quartet contract failed")
            paired.append({"invocation": invocation, "quartet": quartet,
                           "log_speedup": math.log(statistics.mean(retained) /
                                                   statistics.mean(candidate))})
    logs = [x["log_speedup"] for x in paired]
    ci = bootstrap_ci(logs)
    by_invocation = {name: math.exp(statistics.median(
        x["log_speedup"] for x in paired if x["invocation"] == name))
        for name in ("a", "b")}
    summary = {
        "wave": 118, "gpu": fields, "model": model_meta,
        "samples_per_arm_per_invocation": 100, "total_samples": len(records),
        "oracle_matches": f"{len(records)}/{len(records)}",
        "median_speedup": math.exp(statistics.median(logs)),
        "p10_speedup": math.exp(percentile(logs, 0.10)),
        "p90_speedup": math.exp(percentile(logs, 0.90)),
        "bootstrap_ci95_speedup": [math.exp(ci[0]), math.exp(ci[1])],
        "invocation_median_speedup": by_invocation,
        "diagnostic_pass": all(x > 1.0 for x in by_invocation.values()) and math.exp(ci[0]) > 0.99,
        "retention_authority": False, "target_15000_tps_achieved": False,
    }
    (RESULTS / "production-records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (RESULTS / "paired-log-speedups.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (RESULTS / "wave118-stability.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WAVE118_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
except Exception:
    (RESULTS / "FAILED.json").write_text(json.dumps({
        "phase": phase, "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
    archive()
    raise
'''

code = (code.replace("__BASE_REV__", BASE_REV)
            .replace("__HEAD_REV__", HEAD_REV)
            .replace("__PATCH_SHA__", patch_sha)
            .replace("__PATCH_B64__", patch_b64))
ast.parse(code)

notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# GwenLand glcuda Wave 118 - in-process T4 stability gate\n",
            "\nOne model/context, ABBA production prefill, two reversed invocations.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": code.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
round_trip = json.loads(OUTPUT.read_text(encoding="utf-8"))
round_code = "".join(round_trip["cells"][1]["source"])
ast.parse(round_code)
match = re.search(r'PATCH_GZIP_B64 = """(.*?)"""', round_code, re.S)
if match is None:
    raise RuntimeError("generated notebook has no embedded patch")
decoded = gzip.decompress(base64.b64decode(match.group(1)))
if decoded != patch or hashlib.sha256(decoded).hexdigest() != patch_sha:
    raise RuntimeError("generated notebook patch mismatch")
print(json.dumps({"notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                  "base_rev": BASE_REV, "head_rev": HEAD_REV,
                  "patch_bytes": len(patch), "patch_sha256": patch_sha}, indent=2))
