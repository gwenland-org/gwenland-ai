"""Build the self-contained Wave 123 Q8 no-store T4 A/B notebook."""

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
OUTPUT = ROOT / "notebooks/glcuda_t4_wave123_q8_nostore.ipynb"

INTENT_TO_ADD = ["glcuda/examples/wave123_q8_nostore.rs"]
subprocess.run(["git", "add", "--intent-to-add", "--", *INTENT_TO_ADD], cwd=ROOT, check=True)
try:
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
finally:
    subprocess.run(["git", "reset", "--", *INTENT_TO_ADD], cwd=ROOT, check=True)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave118", "wave123").replace("Wave 118", "Wave 123")
code = code.replace("WAVE118", "WAVE123")
code = code.replace('ROOT = Path("/kaggle/working/wave123")', 'ROOT = Path("/tmp/wave123")')
code = code.replace(
    'BUILD = "wave123-in-process-stability-v1"',
    'BUILD = "wave123-q8-nostore-stacked-v1"',
)
code = code.replace(
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave123-in-process-stability-results.zip")',
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave123-q8-nostore-results.zip")',
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

start = code.index('    phase = "build"')
end = code.rindex("except Exception:")
replacement = r'''    phase = "build"
    build = run([cargo, "build", "--release", "-p", "glcuda", "--example",
                 "wave123_q8_nostore", "--locked"], cwd=TREE, env=common)
    save("cargo-build.log", build)
    exe = TARGET / "release/examples/wave123_q8_nostore"
    prod_env = {**common, "GLCUDA_FORCE_Q8": "1", "GLCUDA_GRID2D": "1",
                "GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",
                "GLCUDA_BSTAGE": "1", "GLCUDA_GEMM_N16": "1",
                "GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_ATTN_MMA4": "1",
                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1"}

    phase = "production-ab"
    records = []
    summaries = []
    for invocation in ["a", "b"]:
        measured = run([exe, MODEL, invocation], cwd=TREE, env=prod_env, check=False)
        save(f"production-{invocation}.log", measured)
        if measured.returncode:
            raise RuntimeError(f"production {invocation} failed")
        for raw in re.findall(r"\[wave123-sample\]\s*(\{[^\n]+\})", measured.stdout):
            records.append(json.loads(raw))
        found = re.search(r"\[wave123-summary\]\s*(\{[^\n]+\})", measured.stdout)
        if not found:
            raise RuntimeError(f"production {invocation} summary missing")
        summaries.append(json.loads(found.group(1)))
    if len(records) != 80:
        raise RuntimeError(f"expected 80 production samples, saw {len(records)}")
    if {r["oracle_token"] for r in records} != {3323}:
        raise RuntimeError("oracle drift in production samples")

    retained = [r["prefill_ms"] for r in records if r["arm"] == "retained"]
    candidate = [r["prefill_ms"] for r in records if r["arm"] == "candidate"]
    if len(retained) != len(candidate) or len(retained) != 40:
        raise RuntimeError(f"bad arm sample counts: retained={len(retained)} candidate={len(candidate)}")
    retained_sorted = sorted(retained)
    candidate_sorted = sorted(candidate)
    retained_median = retained_sorted[len(retained_sorted) // 2]
    candidate_median = candidate_sorted[len(candidate_sorted) // 2]

    phase = "candidate-profile"
    prof_env = {**prod_env, "GLCUDA_TELEMETRY": "1"}
    prof = run([exe, MODEL, "profile"], cwd=TREE, env=prof_env, check=False)
    save("candidate-profile.log", prof)
    if prof.returncode or "[wave123-profile]" not in prof.stdout:
        raise RuntimeError("candidate production profile failed")
    profile = json.loads(re.search(r"\[wave123-profile\]\s*(\{[^\n]+\})", prof.stdout).group(1))
    stages = [json.loads(x) for x in re.findall(r"\[wave123-stage\]\s*(\{[^\n]+\})", prof.stdout)]
    expected_names = [
        "qkv",
        "attn_norm",
        "attn_kv_write",
        "attention",
        "attn_out_quant",
        "ffn_down",
        "ffn_gate_up",
        "attn_out",
        "lm_head",
        "ffn_residual_norm_quant",
        "ffn_silu_quant",
        "ffn_residual_add",
    ]
    if [s["name"] for s in stages] != expected_names or profile["oracle_token"] != 3323:
        raise RuntimeError(f"profile contract failed: {profile}, {stages}")
    stage_sum = sum(x["total_ms"] for x in stages)
    ranked = sorted(
        [
            {**s,
             "share_of_gpu_total": s["total_ms"] / profile["gpu_prefill_ms"],
             "share_of_stage_sum": s["total_ms"] / stage_sum if stage_sum else 0.0}
            for s in stages
        ],
        key=lambda s: -s["total_ms"],
    )
    by_name = {x["name"]: x for x in stages}
    summary = {
        "wave": 123,
        "candidate": "q8_nostore_plus_stacked_gate_up",
        "gpu": fields,
        "model": model_meta,
        "production_ab": {
            "samples_per_arm": len(retained),
            "retained_median_ms": retained_median,
            "candidate_median_ms": candidate_median,
            "retained_tps": 244000.0 / retained_median,
            "candidate_tps": 244000.0 / candidate_median,
            "speedup": retained_median / candidate_median,
            "all_candidate_deltas_positive": all(c < r for c, r in zip(candidate, retained)),
            "invocation_summaries": summaries,
        },
        "candidate_profile": profile,
        "candidate_stages": stages,
        "candidate_ranked_stages": ranked,
        "candidate_stage_sum_ms": stage_sum,
        "candidate_attention_ms": sum(by_name[n]["total_ms"] for n in [
            "qkv", "attn_norm", "attn_kv_write", "attention", "attn_out_quant", "attn_out"
        ]),
        "candidate_ffn_ms": sum(by_name[n]["total_ms"] for n in [
            "ffn_down", "ffn_gate_up", "ffn_residual_norm_quant", "ffn_silu_quant",
            "ffn_residual_add"
        ]),
        "retention_authority": True,
        "target_15000_tps_achieved": 244000.0 / candidate_median >= 15000,
    }
    report = [
        "# Wave 123 Q8 no-store + stacked gate/up",
        "",
        f"- Production A/B samples: {len(retained)} per arm, counterbalanced a+b",
        f"- Retained median: {retained_median:.6f} ms = {summary['production_ab']['retained_tps']:.1f} tok/s",
        f"- Candidate median: {candidate_median:.6f} ms = {summary['production_ab']['candidate_tps']:.1f} tok/s",
        f"- Speedup: {summary['production_ab']['speedup']:.4f}x",
        f"- Target 15k reached: {summary['target_15000_tps_achieved']}",
        f"- Candidate profile GPU: {profile['gpu_prefill_ms']:.6f} ms = {profile['gpu_prefill_tps']:.1f} tok/s",
        "",
        "| candidate stage | ms | share of GPU total | calls | bytes read | macs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in ranked:
        report.append(
            f"| `{s['name']}` | {s['total_ms']:.6f} | {100.0 * s['share_of_gpu_total']:.2f}% | "
            f"{s['calls']} | {s['bytes_read']} | {s['macs']} |"
        )
    (RESULTS / "production-records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (RESULTS / "wave123-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("WAVE123_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
code = code[:start] + replacement + code[end:]
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 123 - Q8 no-store stacked gate/up\n",
    "\nProduction T4 A/B for Q8 no-store glue and stacked FFN gate/up dispatch.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))
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
