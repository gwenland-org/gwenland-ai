"""Build the self-contained Wave 120 repaired event-profile notebook."""

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
SOURCE = ROOT / "notebooks/glcuda_t4_wave118_in_process_stability.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave120_event_profile.ipynb"

patch = subprocess.check_output([
    "git", "diff", "--binary", "--full-index", BASE_REV, "--",
    "Cargo.toml", "Cargo.lock", "glcore", "glproc", "glcuda",
], cwd=ROOT)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave118", "wave120").replace("Wave 118", "Wave 120")
code = code.replace("WAVE118", "WAVE120")
# Wave 120 extends the existing harness instead of cloning its Rust target.
code = code.replace("wave120_in_process_stability", "wave118_in_process_stability")
code = re.sub(r'BASE_REV = "[0-9a-f]+"', f'BASE_REV = "{BASE_REV}"', code, count=1)
code = re.sub(r'SOURCE_REV = "[0-9a-f]+"', f'SOURCE_REV = "{HEAD_REV}"', code, count=1)
code = re.sub(r'PATCH_SHA256 = "[0-9a-f]+"', f'PATCH_SHA256 = "{patch_sha}"', code, count=1)
code = re.sub(r'PATCH_GZIP_B64 = """.*?"""', f'PATCH_GZIP_B64 = """{patch_b64}"""',
              code, count=1, flags=re.S)

start = code.index('    phase = "production"')
end = code.rindex("except Exception:")
replacement = r'''    phase = "build"
    build = run([cargo, "build", "--release", "-p", "glcuda", "--example",
                 "wave118_in_process_stability", "--locked"], cwd=TREE, env=common)
    save("cargo-build.log", build)
    exe = TARGET / "release/examples/wave118_in_process_stability"
    prod_env = {**common, "GLCUDA_FORCE_Q8": "1", "GLCUDA_GRID2D": "1",
                "GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",
                "GLCUDA_BSTAGE": "1", "GLCUDA_GEMM_N16": "1",
                "GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_ATTN_MMA4": "1",
                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1",
                "GLCUDA_TELEMETRY": "1"}

    phase = "event-profile"
    measured = run([exe, MODEL, "profile"], cwd=TREE, env=prod_env, check=False)
    save("event-profile.log", measured)
    if measured.returncode or "[wave120-profile]" not in measured.stdout:
        raise RuntimeError("single-pass production profile failed")

    profile = json.loads(re.search(r"\[wave120-profile\]\s*(\{[^\n]+\})", measured.stdout).group(1))
    stages = [json.loads(x) for x in re.findall(r"\[wave120-stage\]\s*(\{[^\n]+\})", measured.stdout)]
    if len(stages) != 8 or profile["gpu_prefill_ms"] <= 0:
        raise RuntimeError(f"event profile contract failed: {profile}, {len(stages)} stages")
    stage_sum = sum(x["total_ms"] for x in stages)
    summary = {"wave": 120, "gpu": fields, "model": model_meta,
               "profile": profile, "stages": stages, "stage_sum_ms": stage_sum,
               "stage_sum_over_gpu_total": stage_sum / profile["gpu_prefill_ms"],
               "retention_authority": False,
               "target_15000_tps_achieved": profile["gpu_prefill_tps"] >= 15000}
    (RESULTS / "wave120-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WAVE120_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
code = code[:start] + replacement + code[end:]
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 120 - repaired T4 event profile\n",
    "\nWhole-prefill GPU timing with post-workload detailed event drain.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))
print(json.dumps({"notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                  "base_rev": BASE_REV, "head_rev": HEAD_REV,
                  "patch_bytes": len(patch), "patch_sha256": patch_sha}, indent=2))
