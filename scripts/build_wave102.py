"""Build the Wave 102 production Q8 A/B notebook from retained Wave 101."""

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DIRECT_NOTEBOOK = REPO / "notebooks/glcuda_t4_wave94_n16_prefetch.ipynb"
PRODUCTION_NOTEBOOK = REPO / "notebooks/glcuda_t4_wave80_mma_av_production.ipynb"
OUTPUT = REPO / "notebooks/glcuda_t4_wave102_n16_prefetch_production.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:80]!r}")
    return text.replace(old, new)


notebook = json.loads(DIRECT_NOTEBOOK.read_text(encoding="utf-8"))
notebook["cells"][0]["id"] = "wave102-intro"
notebook["cells"][0]["source"] = [
    "# glcuda Wave 102 — N16 register-prefetch production Q8 gate\n",
    "\n",
    "Hard-selected Tesla T4, byte-locked Q8 model, two direct gates, and ten "
    "position-balanced production pairs.\n",
]

direct_code = "".join(notebook["cells"][1]["source"])
for old, new in (
    ("Wave 101", "Wave 102"),
    ("wave101", "wave102"),
    ("WAVE101", "WAVE102"),
):
    direct_code = direct_code.replace(old, new)
direct_code = direct_code.replace(
    "glcuda-t4-wave102-n16-prefetch-results.zip",
    "glcuda-t4-wave102-n16-prefetch-production-results.zip",
)
ast.parse(direct_code)
notebook["cells"][1]["id"] = "wave102-direct-gate"
notebook["cells"][1]["source"] = direct_code.splitlines(keepends=True)

historical = json.loads(PRODUCTION_NOTEBOOK.read_text(encoding="utf-8"))
production_code = "".join(historical["cells"][-1]["source"])
for old, new in (
    ("Wave 80", "Wave 102"),
    ("Wave80", "Wave102"),
    ("wave80", "wave102"),
    ("retained_regq", "retained_n16"),
    ("candidate_avmma", "candidate_n16_prefetch"),
):
    production_code = production_code.replace(old, new)

production_code = replace_once(
    production_code,
    '        "attn_mma4_av": candidate,\n',
    '        "gemm_n16_prefetch": candidate,\n'
    '        "attn_mma4_av": True,\n',
)
production_code = replace_once(
    production_code,
    '    expected_attention = "mma4-regq-avmma" if candidate else "mma4-regq"\n',
    '    expected_attention = "mma4-regq-avmma"\n',
)
production_code = replace_once(
    production_code,
    '    if paths != {"bstage-n16-m32", "bstage-n16"} or any(\n'
    '        row.get("ntok") != 244 for row in gemm\n'
    '    ):\n',
    '    expected_paths = {"bstage-n16-m32", "bstage-n16"}\n'
    '    if candidate:\n'
    '        expected_paths.add("bstage-n16-prefetch")\n'
    '    if paths != expected_paths or any(\n'
    '        row.get("ntok") != 244 for row in gemm\n'
    '    ):\n',
)
production_code = replace_once(
    production_code,
    '    build_env = {"CARGO_TARGET_DIR": str(target)}\n',
    '    build_env = {**cargo_env, "CARGO_TARGET_DIR": str(target)}\n',
)
production_code = replace_once(
    production_code,
    '    common_env = {\n'
    '        "CUDA_VISIBLE_DEVICES": "0",\n',
    '    common_env = {\n'
    '        **cargo_env,\n'
    '        "CUDA_VISIBLE_DEVICES": "0",\n',
)
production_code = replace_once(
    production_code,
    '        "GLCUDA_ATTN_MMA4_REGQ": "1",\n'
    '    }\n'
    '    arm_env = {\n'
    '        "retained_n16": {},\n'
    '        "candidate_n16_prefetch": {"GLCUDA_ATTN_MMA4_AV": "1"},\n'
    '    }\n',
    '        "GLCUDA_ATTN_MMA4_REGQ": "1",\n'
    '        "GLCUDA_ATTN_MMA4_AV": "1",\n'
    '    }\n'
    '    arm_env = {\n'
    '        "retained_n16": {},\n'
    '        "candidate_n16_prefetch": {"GLCUDA_GEMM_N16_PREFETCH": "1"},\n'
    '    }\n',
)
production_code = production_code.replace("write_archive()", "archive()")
production_code = (
    '\n# Wave 101 direct gate leaves these authoritative values in scope.\n'
    'gpu_line = gpu.stdout.splitlines()[0]\n'
    'direct_result = verdict\n'
    + production_code
)

required = (
    '"gemm_n16_prefetch": candidate',
    '"attn_mma4_av": True',
    '"GLCUDA_GEMM_N16_PREFETCH": "1"',
    'expected_paths.add("bstage-n16-prefetch")',
    'PRODUCTION_REPEATS = 10',
    'target_reached = candidate["prefill_p50_median"] >= 15_000.0',
    '"quant": "Q8_0"',
)
missing = [needle for needle in required if needle not in production_code]
if missing:
    raise RuntimeError(f"Wave 102 production contract missing: {missing}")
ast.parse(production_code)
notebook["cells"].append(
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "wave102-production-ab",
        "metadata": {},
        "outputs": [],
        "source": production_code.splitlines(keepends=True),
    }
)

OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
round_trip = json.loads(OUTPUT.read_text(encoding="utf-8"))
for cell in round_trip["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
print(json.dumps({
    "notebook": str(OUTPUT),
    "bytes": OUTPUT.stat().st_size,
    "cells": len(round_trip["cells"]),
    "production_repeats": 10,
    "production_sessions": 20,
}, indent=2))
