"""Build the self-contained Wave 131 production and ceiling T4 notebook."""

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
HEAD_REV = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
SOURCE = ROOT / "notebooks/glcuda_t4_wave123_q8_nostore.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave131_fused_swiglu_production.ipynb"
STAGING = ROOT / ".kaggle-wave131"

patch = subprocess.check_output(
    [
        "git",
        "diff",
        "--binary",
        "--full-index",
        BASE_REV,
        "--",
        "Cargo.toml",
        "Cargo.lock",
        "glcore",
        "glproc",
        "glcuda",
    ],
    cwd=ROOT,
)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave123", "wave131").replace("Wave 123", "Wave 131")
code = code.replace("WAVE123", "WAVE131")
code = code.replace("q8-nostore-stacked", "fused-swiglu-production-ceiling")
code = code.replace("q8-nostore-results", "fused-swiglu-production-results")
code = code.replace("wave131_q8_nostore", "wave131_fused_swiglu_production")
code = code.replace('"67 passed"', '"73 passed"')
code = code.replace('"wave": 123', '"wave": 131')
code = code.replace('"q8_nostore_plus_stacked_gate_up"', '"wave129_fused_swiglu_production"')
code = code.replace(
    "# Wave 131 Q8 no-store + stacked gate/up",
    "# Wave 131 fused SwiGLU production and ceiling",
)
code = re.sub(r'BASE_REV = "[0-9a-f]+"', f'BASE_REV = "{BASE_REV}"', code, count=1)
code = re.sub(r'SOURCE_REV = "[0-9a-f]+"', f'SOURCE_REV = "{HEAD_REV}"', code, count=1)
code = re.sub(r'PATCH_SHA256 = "[0-9a-f]+"', f'PATCH_SHA256 = "{patch_sha}"', code, count=1)
code = re.sub(
    r'PATCH_GZIP_B64 = """.*?"""',
    f'PATCH_GZIP_B64 = """{patch_b64}"""',
    code,
    count=1,
    flags=re.S,
)
code = code.replace(
    '"GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_ATTN_MMA4": "1",',
    '"GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_N16_FUSED_SWIGLU": "1", '
    '"GLCUDA_ATTN_MMA4": "1",',
)
code = code.replace(
    '"GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",',
    '"GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_Q8_NOSTORE": "1", '
    '"GLCUDA_FFN_GATE_UP_STACKED": "1", "GLCUDA_DEFER_FFN_RESIDUAL": "1", '
    '"GLCUDA_NTILE128": "1",',
)

resource_start = code.index('    phase = "compiler-resource"')
resource_end = code.index('    phase = "model"', resource_start)
resource_gate = r'''    phase = "compiler-resource"
    ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    resource = run([
        ptxas, "-v", "-arch=sm_75",
        TREE / "glcuda/src/kernels/glcuda_sm75_wave129.ptx",
        "-o", ROOT / "wave131.cubin",
    ], check=False)
    save("ptxas-wave129.log", resource)
    resource_log = resource.stdout + "\n" + resource.stderr
    expected_resource = [
        "Used 80 registers", "used 1 barriers", "16384 bytes smem",
        "0 bytes stack frame", "0 bytes spill stores", "0 bytes spill loads",
    ]
    if resource.returncode or any(item not in resource_log for item in expected_resource):
        raise RuntimeError(f"Wave 131 resource gate failed: {resource_log[-4000:]}")

'''
code = code[:resource_start] + resource_gate + code[resource_end:]

needle = "    summary = {\n"
if needle not in code:
    raise RuntimeError("Wave 123 summary insertion point moved")
ceiling = r'''    phase = "glbench-ceiling"
    glbench_build = run([cargo, "build", "--release", "-p", "glbench", "--locked"],
                        cwd=TREE, env=common)
    save("glbench-build.log", glbench_build)
    glbench = TARGET / "release/glbench"
    ceiling_prompt = ("Measure this deterministic systems prompt carefully. Explain how "
                      "token-parallel integer matrix multiplication uses shared memory, "
                      "Tensor Cores, and fixed launch geometry. " * 8)
    base_env = dict(prod_env)
    base_env.pop("GLCUDA_N16_FUSED_SWIGLU", None)
    ceiling_runs = []
    for index, arm in enumerate(["retained", "candidate", "candidate", "retained"]):
        out = RESULTS / f"glbench-{index + 1}-{arm}.json"
        env = prod_env if arm == "candidate" else base_env
        measured = run([
            glbench, "run", "--engine", "glcuda", "--model", MODEL,
            "--prompt", ceiling_prompt, "--tokens", "64", "--cold-iters", "1",
            "--warmup", "2", "--iters", "10", "--temperature", "0", "--seed", "42",
            "--kind", "end_to_end", "--out", out,
        ], cwd=TREE, env=env, check=False)
        save(f"glbench-{index + 1}-{arm}.log", measured)
        if measured.returncode or not out.is_file():
            raise RuntimeError(f"glbench ceiling run failed: {index + 1} {arm}")
        dispatch_log = measured.stdout + "\n" + measured.stderr
        fused_dispatched = '"path":"wave129-fused-swiglu-production"' in dispatch_log
        contract_enabled = (
            '"q8_nostore":true' in dispatch_log
            and '"ffn_gate_up_stacked":true' in dispatch_log
            and '"defer_ffn_residual":true' in dispatch_log
        )
        if not contract_enabled:
            raise RuntimeError(f"retained prerequisite contract missing: {index + 1} {arm}")
        if (arm == "candidate") != fused_dispatched:
            raise RuntimeError(
                f"fused dispatch mismatch: run={index + 1} arm={arm} dispatched={fused_dispatched}"
            )
        session = json.loads(out.read_text(encoding="utf-8"))
        analysis = session["analysis"]
        ceiling_runs.append({
            "order": index + 1,
            "arm": arm,
            "prefill_tps": analysis["prefill_tps"],
            "decode_tps": analysis["decode_tps"],
            "ceiling_efficiency": analysis.get("ceiling_efficiency"),
            "ceiling_basis": analysis.get("ceiling_basis"),
            "bottleneck": analysis.get("bottleneck"),
            "notes": analysis.get("notes", []),
        })

'''
code = code.replace(needle, ceiling + needle, 1)
code = code.replace(
    '        "production_ab": {',
    '        "glbench_ceiling_runs": ceiling_runs,\n        "production_ab": {',
    1,
)
code = code.replace(
    '        f"- Target 15k reached: {summary[\'target_15000_tps_achieved\']}",',
    '        f"- Target 15k reached: {summary[\'target_15000_tps_achieved\']}",\n'
    '        f"- Decode P50 range: {min(r[\'decode_tps\'][\'median\'] for r in ceiling_runs):.3f}.."\n'
    '        f"{max(r[\'decode_tps\'][\'median\'] for r in ceiling_runs):.3f} tok/s",',
    1,
)
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 131 - production fused SwiGLU and ceilings\n",
    "\nCounterbalanced production A/B plus reproduced glbench prefill/decode ceilings on Tesla T4.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))
STAGING.mkdir(exist_ok=True)
(STAGING / OUTPUT.name).write_bytes(OUTPUT.read_bytes())
(STAGING / "kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "jinxsuperdev/glcuda-t4-wave131-fused-swiglu-production",
            "title": "glcuda T4 Wave131 fused SwiGLU production",
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
print(
    json.dumps(
        {
            "notebook": str(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
            "base_rev": BASE_REV,
            "head_rev": HEAD_REV,
            "patch_bytes": len(patch),
            "patch_sha256": patch_sha,
        },
        indent=2,
    )
)
