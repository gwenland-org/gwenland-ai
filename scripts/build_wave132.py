"""Build the self-contained Wave 132 decode-fusion T4 notebook."""

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
SOURCE = ROOT / "notebooks/glcuda_t4_wave131_fused_swiglu_production.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave132_decode_q8_glue.ipynb"
STAGING = ROOT / ".kaggle-wave132"

patch = subprocess.check_output(
    [
        "git",
        "diff",
        "--binary",
        "--full-index",
        BASE_REV,
        "--",
        "glcuda",
    ],
    cwd=ROOT,
)
if not patch:
    raise RuntimeError("Wave 132 patch is empty")
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace('ROOT = Path("/kaggle/working/wave131")',
                    'ROOT = Path("/kaggle/working/wave132")')
code = code.replace('ROOT = Path("/tmp/wave131")', 'ROOT = Path("/tmp/wave132")')
code = code.replace("glcuda-t4-wave131-fused-swiglu-production-results.zip",
                    "glcuda-t4-wave132-decode-q8-glue-results.zip")
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
code = code.replace('"73 passed"', '"74 passed"')

common_line = '    common = {**cargo_env, "CARGO_TARGET_DIR": TARGET, "CUDA_VISIBLE_DEVICES": "0"}\n'
candidate_env = '''    candidate_switch_env = {
        **common,
        "GLCUDA_FUSE_Q8_GLUE": "1",
        "GLCUDA_DECODE_Q8_GLUE": "1",
    }
'''
if common_line not in code:
    raise RuntimeError("common environment insertion point moved")
code = code.replace(common_line, common_line + candidate_env, 1)

parity_tail = '''    if parity.returncode or "0 failed" not in parity.stdout:
        raise RuntimeError("CUDA parity failed")

'''
graph_gate = '''    phase = "graph-replay"
    graph = run([cargo, "test", "--release", "-p", "glcuda", "--test", "graph_replay",
                 "--locked", "--", "--nocapture", "--test-threads=1"],
                cwd=TREE, env=candidate_switch_env, check=False)
    save("cargo-graph-replay.log", graph)
    if graph.returncode or "4 passed" not in graph.stdout or "0 failed" not in graph.stdout:
        raise RuntimeError("Wave 132 graph replay gate failed")

'''
if parity_tail not in code:
    raise RuntimeError("CUDA parity insertion point moved")
code = code.replace(parity_tail, parity_tail + graph_gate, 1)

prod_env_tail = '                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1"}\n'
if prod_env_tail not in code:
    raise RuntimeError("production environment moved")
code = code.replace(
    prod_env_tail,
    '                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1",\n'
    '                "GLCUDA_DECODE_Q8_GLUE": "1"}\n',
    1,
)
code = code.replace('base_env.pop("GLCUDA_N16_FUSED_SWIGLU", None)',
                    'base_env.pop("GLCUDA_DECODE_Q8_GLUE", None)', 1)
code = code.replace(
    '            "--kind", "end_to_end", "--out", out,',
    '            "--kind", "end_to_end", "--verify-against", "glproc", "--out", out,',
    1,
)
old_dispatch = '''        fused_dispatched = '"path":"wave129-fused-swiglu-production"' in dispatch_log
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
'''
new_dispatch = '''        prefill_dispatched = '"path":"wave129-fused-swiglu-production"' in dispatch_log
        decode_dispatched = '"path":"wave132-exact-q8-glue"' in dispatch_log
        contract_enabled = (
            '"q8_nostore":true' in dispatch_log
            and '"ffn_gate_up_stacked":true' in dispatch_log
            and '"defer_ffn_residual":true' in dispatch_log
            and '"n16_fused_swiglu":true' in dispatch_log
        )
        if not contract_enabled or not prefill_dispatched:
            raise RuntimeError(f"retained Wave 131 contract missing: {index + 1} {arm}")
        if (arm == "candidate") != decode_dispatched:
            raise RuntimeError(
                f"decode dispatch mismatch: run={index + 1} arm={arm} dispatched={decode_dispatched}"
            )
'''
if old_dispatch not in code:
    raise RuntimeError("glbench dispatch gate moved")
code = code.replace(old_dispatch, new_dispatch, 1)

session_line = '        analysis = session["analysis"]\n'
oracle_gate = '''        validation_text = json.dumps(session.get("validation", {}))
        if not session.get("validation", {}).get("passed") or "50/50 tokens match oracle" not in validation_text:
            raise RuntimeError(f"oracle gate failed: run={index + 1} arm={arm}")
'''
code = code.replace(session_line, session_line + oracle_gate, 1)
code = code.replace('"wave": 131,', '"wave": 132,', 1)
code = code.replace('"candidate": "wave129_fused_swiglu_production",',
                    '"candidate": "wave132_exact_decode_q8_glue",', 1)
code = code.replace('f"# Wave 131', 'f"# Wave 132', 1)

ast.parse(code)
notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 132 - exact Q8 decode glue\n",
    "\nProduction A-B-B-A decode fusion gate with graph replay and a 50-token glproc oracle.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))

STAGING.mkdir(exist_ok=True)
(STAGING / OUTPUT.name).write_bytes(OUTPUT.read_bytes())
(STAGING / "kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "jinxsuperdev/glcuda-t4-wave132-decode-q8-glue",
            "title": "glcuda T4 Wave132 decode Q8 glue",
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
