"""Build the self-contained Wave 121 LM-head GEMM T4 gate notebook."""

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
OUTPUT = ROOT / "notebooks/glcuda_t4_wave121_lmhead_gemm.ipynb"

patch = subprocess.check_output([
    "git", "diff", "--binary", "--full-index", BASE_REV, "--",
    "Cargo.toml", "Cargo.lock", "glcore", "glproc", "glcuda",
], cwd=ROOT)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave118", "wave121").replace("Wave 118", "Wave 121")
code = code.replace("WAVE118", "WAVE121")
code = code.replace("wave121_in_process_stability", "wave118_in_process_stability")
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

    phase = "lmhead-ab"
    records = []
    orders = [("retained", "candidate"), ("candidate", "retained")] * 3
    for repeat, order in enumerate(orders):
        for arm in order:
            env = dict(prod_env)
            if arm == "candidate":
                env["GLCUDA_LMHEAD_GEMM"] = "1"
            measured = run([exe, MODEL, "profile"], cwd=TREE, env=env, check=False)
            save(f"profile-{repeat}-{arm}.log", measured)
            if measured.returncode or "[wave120-profile]" not in measured.stdout:
                raise RuntimeError(f"{arm} profile failed at repeat {repeat}")
            profile = json.loads(re.search(r"\[wave120-profile\]\s*(\{[^\n]+\})", measured.stdout).group(1))
            stages = [json.loads(x) for x in re.findall(r"\[wave120-stage\]\s*(\{[^\n]+\})", measured.stdout)]
            lm = next(x for x in stages if x["name"] == "lm_head")
            if len(stages) != 9 or profile["oracle_token"] != 3323:
                raise RuntimeError(f"{arm} contract failed: {profile}, {len(stages)} stages")
            records.append({"repeat": repeat, "arm": arm, **profile,
                            "lm_head_ms": lm["total_ms"]})
    by_arm = {arm: [x for x in records if x["arm"] == arm]
              for arm in ("retained", "candidate")}
    med = lambda arm, key: statistics.median(x[key] for x in by_arm[arm])
    summary = {"wave": 121, "gpu": fields, "model": model_meta,
               "records": records,
               "median_gpu_prefill_ms": {arm: med(arm, "gpu_prefill_ms") for arm in by_arm},
               "median_prefill_tps": {arm: med(arm, "gpu_prefill_tps") for arm in by_arm},
               "median_lm_head_ms": {arm: med(arm, "lm_head_ms") for arm in by_arm}}
    summary["speedup"] = (summary["median_gpu_prefill_ms"]["retained"] /
                          summary["median_gpu_prefill_ms"]["candidate"])
    candidate_lm = summary["median_lm_head_ms"]["candidate"]
    if candidate_lm <= 0:
        raise RuntimeError(f"LM-head event was not recorded: {summary['median_lm_head_ms']}")
    summary["lm_head_speedup"] = summary["median_lm_head_ms"]["retained"] / candidate_lm
    summary["target_15000_tps_achieved"] = summary["median_prefill_tps"]["candidate"] >= 15000
    (RESULTS / "wave121-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WAVE121_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
code = code[:start] + replacement + code[end:]
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 121 - LM-head tensor-core GEMM gate\n",
    "\nSix counterbalanced T4 event-timed pairs with the production oracle.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))
print(json.dumps({"notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                  "base_rev": BASE_REV, "head_rev": HEAD_REV,
                  "patch_bytes": len(patch), "patch_sha256": patch_sha}, indent=2))
