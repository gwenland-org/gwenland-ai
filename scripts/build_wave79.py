"""Build the byte-locked Wave 79 Tesla T4 device-gate notebook."""

import ast
import base64
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_REV = "bd5c956bafb3bb6738c3f1de348a4ebb55d9c29f"
SOURCE_REV = "48ae7c7ea53e299efca02afb8c996f938dec16bf"
PATCH_PATHS = (
    "glcuda/src",
    "glcuda/tests/parity.rs",
    "glcuda/examples/wave78_mma_av_attention.rs",
)


patch = subprocess.check_output(
    [
        "git",
        "diff",
        "--binary",
        "--full-index",
        BASE_REV,
        SOURCE_REV,
        "--",
        *PATCH_PATHS,
    ],
    cwd=REPO,
)
if not patch:
    raise RuntimeError("Wave 79 source patch is empty")
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(patch).decode("ascii")

code = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

BASE_REV = __BASE_REV__
SOURCE_REV = __SOURCE_REV__
PATCH_SHA256 = __PATCH_SHA__
PATCH_B64 = __PATCH_B64__
REPO_URL = "https://github.com/gwenland-org/gwenland-ai.git"
ROOT = Path("/kaggle/working/wave79")
TREE = ROOT / "repo"
RESULTS = ROOT / "results"
ARCHIVE = Path("/kaggle/working/glcuda-t4-wave79-mma-av-device-results.zip")


def run(cmd, cwd=None, env=None, timeout=3600, check=True):
    merged = os.environ.copy()
    if env:
        merged.update({str(key): str(value) for key, value in env.items()})
    process = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    print("$", " ".join(str(part) for part in cmd), flush=True)
    print(process.stdout[-16000:], flush=True)
    print(process.stderr[-16000:], file=sys.stderr, flush=True)
    if check and process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {cmd}")
    return process


def save(name, process):
    (RESULTS / name).write_text(
        process.stdout + "\n--- STDERR ---\n" + process.stderr,
        encoding="utf-8",
    )


def write_archive():
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(RESULTS.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(RESULTS))
    print("ARCHIVE", ARCHIVE, hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())


def entry_resources(log, entry):
    match = re.search(
        r"Compiling entry function ['\"]" + re.escape(entry)
        + r"['\"].*?(?=Compiling entry function|\Z)",
        log,
        re.S,
    )
    segment = match.group(0) if match else ""
    register_match = re.search(r"Used (\d+) registers", segment)
    barrier_match = re.search(r"used (\d+) barriers", segment)
    return {
        "found": bool(match),
        "registers": int(register_match.group(1)) if register_match else None,
        "barriers": int(barrier_match.group(1)) if barrier_match else None,
        "stack_bytes": max([int(x) for x in re.findall(r"(\d+) bytes stack frame", segment)] or [0]),
        "spill_store_bytes": max([int(x) for x in re.findall(r"(\d+) bytes spill stores", segment)] or [0]),
        "spill_load_bytes": max([int(x) for x in re.findall(r"(\d+) bytes spill loads", segment)] or [0]),
    }


try:
    shutil.rmtree(ROOT, ignore_errors=True)
    RESULTS.mkdir(parents=True)
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=60,
    )
    save("nvidia-smi.log", gpu)
    gpu_line = gpu.stdout.strip().splitlines()[0]
    if "Tesla T4" not in gpu_line or "7.5" not in gpu_line:
        raise RuntimeError(f"Wave 79 requires Tesla T4 sm_75, got {gpu_line}")

    if not shutil.which("cargo"):
        installer = ROOT / "rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run(["sh", installer, "-y", "--profile", "minimal"], timeout=1800)
        os.environ["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.environ["PATH"]
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    save("rust.log", run([cargo, "--version"], timeout=60))

    save(
        "git-clone.log",
        run(["git", "clone", "--filter=blob:none", REPO_URL, TREE], timeout=1800),
    )
    save(
        "git-checkout.log",
        run(["git", "checkout", "--detach", BASE_REV], cwd=TREE, timeout=300),
    )
    patch_bytes = base64.b64decode(PATCH_B64)
    actual_patch_sha = hashlib.sha256(patch_bytes).hexdigest()
    if actual_patch_sha != PATCH_SHA256:
        raise RuntimeError(
            f"embedded patch SHA mismatch: {actual_patch_sha} != {PATCH_SHA256}"
        )
    patch_path = ROOT / "wave79.patch"
    patch_path.write_bytes(patch_bytes)
    save(
        "git-apply.log",
        run(["git", "apply", "--binary", "--index", patch_path], cwd=TREE, timeout=300),
    )
    save("git-diff-check.log", run(["git", "diff", "--cached", "--check"], cwd=TREE))
    changed = run(["git", "diff", "--cached", "--name-only"], cwd=TREE)
    save("changed-files.log", changed)
    (RESULTS / "source.json").write_text(
        json.dumps(
            {
                "base_revision": BASE_REV,
                "source_revision": SOURCE_REV,
                "patch_sha256": PATCH_SHA256,
                "changed_files": changed.stdout.splitlines(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ptx_path = TREE / "glcuda/src/kernels/glcuda_sm75.ptx"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    if not Path(ptxas).is_file():
        raise RuntimeError(f"ptxas not found: {ptxas}")
    assembled = run(
        [ptxas, "-v", "-arch=sm_75", ptx_path, "-o", ROOT / "wave79.cubin"],
        cwd=TREE,
        timeout=1800,
        check=False,
    )
    save("ptxas-v.log", assembled)
    if assembled.returncode:
        raise RuntimeError("Wave 79 full-module ptxas gate failed")
    ptx_log = assembled.stdout + "\n" + assembled.stderr
    resources = {
        entry: entry_resources(ptx_log, entry)
        for entry in [
            "gl_attn_mma4_regq_fused_f32",
            "gl_attn_mma4_regq_avmma_fused_f32",
        ]
    }
    (RESULTS / "resources.json").write_text(
        json.dumps(resources, indent=2), encoding="utf-8"
    )
    if not all(item["found"] for item in resources.values()):
        raise RuntimeError(f"missing ptxas entry: {resources}")
    if any(
        item["stack_bytes"]
        or item["spill_store_bytes"]
        or item["spill_load_bytes"]
        for item in resources.values()
    ):
        raise RuntimeError(f"resource gate failed: {resources}")

    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "GLCUDA_JIT_VERBOSE": "1",
        "RUST_BACKTRACE": "1",
    }
    parity = run(
        [
            cargo,
            "test",
            "--release",
            "-p",
            "glcuda",
            "--test",
            "parity",
            "--locked",
            "--",
            "--test-threads=1",
            "--nocapture",
        ],
        cwd=TREE,
        env=env,
        timeout=7200,
        check=False,
    )
    save("parity.log", parity)
    parity_text = parity.stdout + "\n" + parity.stderr
    if parity.returncode:
        raise RuntimeError("Wave 79 serial device parity failed")
    if "fused_mma4_attention_matches_oracle_at_production_and_tail_shapes ... ok" not in parity_text:
        raise RuntimeError("Wave 78 parity case did not report success")
    if "SKIP:" in parity_text:
        raise RuntimeError("Wave 79 parity unexpectedly skipped a device case")

    build = run(
        [
            cargo,
            "build",
            "--release",
            "-p",
            "glcuda",
            "--example",
            "wave78_mma_av_attention",
            "--locked",
        ],
        cwd=TREE,
        env=env,
        timeout=7200,
        check=False,
    )
    save("cargo-build.log", build)
    if build.returncode:
        raise RuntimeError("Wave 79 A/B harness build failed")
    direct = run(
        [TREE / "target/release/examples/wave78_mma_av_attention"],
        cwd=TREE,
        env=env,
        timeout=7200,
        check=False,
    )
    save("wave79-direct.log", direct)
    direct_line = next(
        (line for line in direct.stdout.splitlines() if line.startswith("[wave78-direct] ")),
        "",
    )
    if direct.returncode or not direct_line:
        raise RuntimeError("Wave 79 fused-attention direct gate failed")
    direct_result = json.loads(direct_line.split("] ", 1)[1])

    result = {
        "status": "device_pass",
        "base_revision": BASE_REV,
        "source_revision": SOURCE_REV,
        "patch_sha256": PATCH_SHA256,
        "gpu": gpu_line,
        "ptxas": resources,
        "parity": "33/33 serial device pass",
        "direct": direct_result,
    }
    (RESULTS / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS / "PASS.json").write_text(
        json.dumps({"status": "device_pass"}, indent=2), encoding="utf-8"
    )
    write_archive()
except Exception:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_archive()
    raise
'''

code = (
    code.replace("__BASE_REV__", repr(BASE_REV))
    .replace("__SOURCE_REV__", repr(SOURCE_REV))
    .replace("__PATCH_SHA__", repr(patch_sha))
    .replace("__PATCH_B64__", repr(patch_b64))
)
ast.parse(code)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "wave79-intro",
            "metadata": {},
            "source": [
                "# glcuda Wave 79 - fused compensated-MMA AV T4 device gate\n",
                "\n",
                "Hard-selected Tesla T4 compiler, parity, and interleaved direct A/B gate.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "wave79-run",
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
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = REPO / "notebooks/glcuda_t4_wave79_mma_av_device.ipynb"
output.write_text(
    json.dumps(notebook, indent=1) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"generated {output.relative_to(REPO)}")
print(f"source={SOURCE_REV} base={BASE_REV} patch_sha256={patch_sha}")
