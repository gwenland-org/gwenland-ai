"""Build the self-contained Wave 109 Kaggle T4 direct-gate notebook."""

from __future__ import annotations

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
NOTEBOOK = ROOT / "notebooks/glcuda_t4_wave109_n16_m32_prefetch_remat.ipynb"


def checked(*args: str) -> bytes:
    return subprocess.check_output(args, cwd=ROOT)


patch = checked(
    "git",
    "diff",
    "--binary",
    "--full-index",
    BASE_REV,
    HEAD_REV,
    "--",
    "Cargo.toml",
    "Cargo.lock",
    "glcore",
    "glproc",
    "glcuda",
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
import re
import shutil
import subprocess
import traceback
import urllib.request
import zipfile

BUILD = "wave109-n16-m32-prefetch-remat-v1"
REPO_URL = "https://github.com/gwenland-org/gwenland-ai.git"
BASE_REV = "__BASE_REV__"
SOURCE_REV = "__HEAD_REV__"
PATCH_SHA256 = "__PATCH_SHA__"
PATCH_GZIP_B64 = """__PATCH_B64__"""
ROOT = Path("/kaggle/working/wave109")
TREE = ROOT / "repo"
RESULTS = ROOT / "results"
TARGET = ROOT / "target"
FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave109-n16-m32-prefetch-remat-results.zip")

if ROOT.exists():
    shutil.rmtree(ROOT)
RESULTS.mkdir(parents=True)


def run(cmd, *, cwd=None, env=None, timeout=7200, check=True):
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    p = subprocess.run(
        [str(x) for x in cmd], cwd=cwd, env=merged, text=True,
        capture_output=True, timeout=timeout,
    )
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


def archive():
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(RESULTS.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(RESULTS))
    digest = hashlib.sha256(FINAL_ZIP.read_bytes()).hexdigest()
    print("ARCHIVE", FINAL_ZIP, digest, flush=True)
    return digest


def fail(phase):
    (RESULTS / "FAILED.json").write_text(
        json.dumps({"phase": phase, "traceback": traceback.format_exc()}, indent=2),
        encoding="utf-8",
    )
    archive()
    raise RuntimeError(f"Wave 109 failed in {phase}")


def resource(log, entry):
    marker = re.search(
        r"Compiling entry function ['\"]" + re.escape(entry) +
        r"['\"].*?(?=Compiling entry function|\Z)", log, re.S,
    )
    segment = marker.group(0) if marker else ""
    number = lambda pattern: [int(x) for x in re.findall(pattern, segment)]
    regs = number(r"Used (\d+) registers")
    smem = number(r"(\d+) bytes smem")
    return {
        "entry": entry,
        "found": bool(marker),
        "registers": regs[0] if regs else None,
        "static_shared_bytes": smem[0] if smem else 0,
        "stack_frame_bytes": max(number(r"(\d+) bytes stack frame"), default=0),
        "spill_store_bytes": max(number(r"(\d+) bytes spill stores"), default=0),
        "spill_load_bytes": max(number(r"(\d+) bytes spill loads"), default=0),
    }


phase = "bootstrap"
try:
    patch = gzip.decompress(base64.b64decode(PATCH_GZIP_B64))
    got = hashlib.sha256(patch).hexdigest()
    if got != PATCH_SHA256:
        raise RuntimeError(f"patch hash mismatch: {got} != {PATCH_SHA256}")
    patch_path = RESULTS / "wave109.patch"
    patch_path.write_bytes(patch)
    (RESULTS / "source.json").write_text(json.dumps({
        "build": BUILD, "base_rev": BASE_REV, "source_rev": SOURCE_REV,
        "patch_sha256": PATCH_SHA256, "patch_bytes": len(patch),
    }, indent=2), encoding="utf-8")

    gpu = run([
        "nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], timeout=60)
    save("nvidia-smi.log", gpu)
    fields = [x.strip() for x in gpu.stdout.splitlines()[0].split(",")]
    if len(fields) < 5 or fields[1] != "Tesla T4" or fields[2] != "7.5":
        raise RuntimeError(f"Wave 109 requires Tesla T4 sm_75, got {fields}")

    phase = "reconstruct"
    clone = run(["git", "clone", "--filter=blob:none", REPO_URL, TREE], timeout=1800)
    save("git-clone.log", clone)
    checkout = run(["git", "checkout", "--detach", BASE_REV], cwd=TREE, timeout=600)
    save("git-checkout.log", checkout)
    applied = run(["git", "apply", "--whitespace=error", patch_path], cwd=TREE)
    save("git-apply.log", applied)
    diff_check = run(["git", "diff", "--check"], cwd=TREE)
    save("git-diff-check.log", diff_check)

    phase = "ptxas-resource"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    if not Path(ptxas).is_file():
        raise RuntimeError(f"ptxas unavailable: {ptxas}")
    retained_ptx = TREE / "glcuda/src/kernels/glcuda_sm75.ptx"
    candidate_ptx = TREE / "glcuda/src/kernels/glcuda_sm75_wave109.ptx"
    p_retained = run([ptxas, "-v", "-arch=sm_75", retained_ptx,
                      "-o", ROOT / "retained.cubin"], cwd=TREE, timeout=1800)
    save("ptxas-retained.log", p_retained)
    p_candidate = run([ptxas, "-v", "-arch=sm_75", candidate_ptx,
                       "-o", ROOT / "candidate.cubin"], cwd=TREE, timeout=1800)
    save("ptxas-candidate.log", p_candidate)
    retained = resource(p_retained.stdout + "\n" + p_retained.stderr,
                        "gl_gemm_mma_q8_bstage_n16_m32")
    candidate = resource(p_candidate.stdout + "\n" + p_candidate.stderr,
                         "gl_gemm_mma_q8_bstage_n16_m32_prefetch_remat")
    resources = {"retained": retained, "candidate": candidate}
    (RESULTS / "resources.json").write_text(
        json.dumps(resources, indent=2), encoding="utf-8"
    )
    if not retained["found"] or not candidate["found"]:
        raise RuntimeError(f"resource entry missing: {resources}")
    if retained["registers"] > 64 or candidate["registers"] > 64:
        raise RuntimeError(f"register gate failed: {resources}")
    if retained["static_shared_bytes"] != 9728 or candidate["static_shared_bytes"] != 9728:
        raise RuntimeError(f"shared-memory gate failed: {resources}")
    for row in resources.values():
        if row["stack_frame_bytes"] or row["spill_store_bytes"] or row["spill_load_bytes"]:
            raise RuntimeError(f"stack/spill gate failed: {resources}")

    phase = "build-test"
    cargo_candidates = [
        shutil.which("cargo"),
        Path.home() / ".cargo/bin/cargo",
        "/usr/local/cargo/bin/cargo",
        "/opt/rust/bin/cargo",
        "/opt/conda/bin/cargo",
        "/usr/local/bin/cargo",
        "/usr/bin/cargo",
    ]
    cargo = next(
        (str(path) for path in cargo_candidates if path and Path(path).is_file()),
        None,
    )
    cargo_env = {}
    bootstrapped = False
    if cargo is None:
        bootstrapped = True
        rustup_url = "https://sh.rustup.rs"
        rustup_script = ROOT / "rustup-init.sh"
        with urllib.request.urlopen(rustup_url, timeout=120) as response:
            rustup_script.write_bytes(response.read())
        cargo_home = ROOT / "cargo-home"
        rustup_home = ROOT / "rustup-home"
        cargo_env = {
            "CARGO_HOME": cargo_home,
            "RUSTUP_HOME": rustup_home,
        }
        install = run(
            ["bash", rustup_script, "-y", "--profile", "minimal",
             "--default-toolchain", "stable", "--no-modify-path"],
            env=cargo_env, timeout=1800,
        )
        save("rustup-install.log", install)
        cargo = str(cargo_home / "bin/cargo")
    if not Path(cargo).is_file():
        raise RuntimeError(f"cargo unavailable after discovery/bootstrap: {cargo}")
    (RESULTS / "cargo-discovery.json").write_text(json.dumps({
        "selected": cargo,
        "bootstrapped": bootstrapped,
        "candidates": [str(path) for path in cargo_candidates if path],
    }, indent=2), encoding="utf-8")
    cargo_version = run([cargo, "--version"], env=cargo_env, timeout=60)
    save("cargo-version.log", cargo_version)
    common = {
        **cargo_env,
        "CARGO_TARGET_DIR": TARGET,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    tests = run([cargo, "test", "-p", "glcuda", "--lib", "--locked"],
                cwd=TREE, env=common)
    save("cargo-lib-tests.log", tests)
    summary = next((x for x in (tests.stdout + tests.stderr).splitlines()
                    if x.startswith("test result:")), "")
    if "67 passed" not in summary or "0 failed" not in summary:
        raise RuntimeError(f"unexpected host test summary: {summary}")
    build = run([cargo, "build", "--release", "-p", "glcuda", "--example",
                 "wave109_n16_m32_prefetch_remat", "--locked"], cwd=TREE, env=common)
    save("cargo-build.log", build)

    phase = "direct-run-1"
    env = {
        **common,
        "GLCUDA_GRID2D": "1",
        "GLCUDA_NTILE128": "1",
        "GLCUDA_BSTAGE": "1",
        "GLCUDA_GEMM_N16": "1",
        "GLCUDA_GEMM_N16_M32_PREFETCH_REMAT": "1",
    }
    exe = TARGET / "release/examples/wave109_n16_m32_prefetch_remat"
    records = []
    for index in (1, 2):
        phase = f"direct-run-{index}"
        measured = run([exe], cwd=TREE, env=env, check=False)
        save(f"direct-run-{index}.log", measured)
        direct_line = next((x for x in measured.stdout.splitlines()
                            if x.startswith("[wave109-direct] ")), "")
        resource_line = next((x for x in measured.stdout.splitlines()
                              if x.startswith("[wave109-resource] ")), "")
        direct = json.loads(direct_line.split("] ", 1)[1]) if direct_line else {}
        driver = json.loads(resource_line.split("] ", 1)[1]) if resource_line else {}
        records.append({"run": index, "direct": direct, "driver": driver,
                        "returncode": measured.returncode})
        if measured.returncode or direct.get("pass") is not True or direct.get("bit_exact") is not True:
            raise RuntimeError(f"direct run {index} gate failed: {records[-1]}")
        if driver.get("retained_active_blocks_per_sm", 0) < 4 or driver.get("candidate_active_blocks_per_sm", 0) < 4:
            raise RuntimeError(f"driver occupancy gate failed: {records[-1]}")
    (RESULTS / "direct-results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    verdict = {
        "pass": True,
        "runs": len(records),
        "minimum_speedup": min(x["direct"]["speedup"] for x in records),
        "all_bit_exact": all(x["direct"]["bit_exact"] for x in records),
        "resources": resources,
    }
    (RESULTS / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print("WAVE109_VERDICT", json.dumps(verdict), flush=True)
    archive()
except Exception:
    fail(phase)
'''

code = (
    code.replace("__BASE_REV__", BASE_REV)
    .replace("__HEAD_REV__", HEAD_REV)
    .replace("__PATCH_SHA__", patch_sha)
    .replace("__PATCH_B64__", patch_b64)
)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GwenLand glcuda Wave 109 — occupancy-neutral N16/M32 prefetch gate\n",
                "\n",
                "Self-contained SM75 resource, parity, and two-run direct A/B gate.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
round_trip = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
round_trip_code = "".join(round_trip["cells"][1]["source"])
match = re.search(r'PATCH_GZIP_B64 = """(.*?)"""', round_trip_code, re.S)
if match is None:
    raise RuntimeError("generated notebook has no embedded patch")
decoded = gzip.decompress(base64.b64decode(match.group(1)))
decoded_sha = hashlib.sha256(decoded).hexdigest()
if decoded != patch or decoded_sha != patch_sha:
    raise RuntimeError(
        f"generated notebook patch mismatch: {decoded_sha} != {patch_sha}"
    )
compile(round_trip_code, str(NOTEBOOK), "exec")
print(json.dumps({
    "notebook": str(NOTEBOOK),
    "bytes": NOTEBOOK.stat().st_size,
    "base_rev": BASE_REV,
    "head_rev": HEAD_REV,
    "patch_bytes": len(patch),
    "patch_sha256": patch_sha,
    "decoded_patch_sha256": decoded_sha,
    "patch_gzip_b64_bytes": len(patch_b64),
}, indent=2))
