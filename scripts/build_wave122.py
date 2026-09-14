"""Build the self-contained Wave 122 elementwise-stage profile notebook."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_REV = "5de5be39c0190b9367da18f0f2001e7f40208c92"
HEAD_REV = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
SOURCE = ROOT / "notebooks/glcuda_t4_wave118_in_process_stability.ipynb"
OUTPUT = ROOT / "notebooks/glcuda_t4_wave122_elementwise_profile.ipynb"
EVIDENCE = ROOT / "benchmarks/glcuda-t4-wave122-elementwise-profile-results.zip"

# Keep this historical notebook byte-for-byte tied to the patch that produced
# its archived T4 evidence.  Re-diffing a newer dirty tree silently turns the
# Wave 122 notebook into a different experiment.
with zipfile.ZipFile(EVIDENCE) as archive:
    patch = archive.read("wave122.patch")
    source = json.loads(archive.read("source.json"))
HEAD_REV = source["source_rev"]
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave118", "wave122").replace("Wave 118", "Wave 122")
code = code.replace("WAVE118", "WAVE122")
code = code.replace(
    'BUILD = "wave122-in-process-stability-v1"',
    'BUILD = "wave122-elementwise-profile-v1"',
)
code = code.replace(
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave122-in-process-stability-results.zip")',
    'FINAL_ZIP = Path("/kaggle/working/glcuda-t4-wave122-elementwise-profile-results.zip")',
)
# Wave 122 extends the existing harness instead of cloning its Rust target.
code = code.replace("wave122_in_process_stability", "wave118_in_process_stability")
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
                 "wave118_in_process_stability", "--locked"], cwd=TREE, env=common)
    save("cargo-build.log", build)
    exe = TARGET / "release/examples/wave118_in_process_stability"
    prod_env = {**common, "GLCUDA_FORCE_Q8": "1", "GLCUDA_GRID2D": "1",
                "GLCUDA_FUSE_Q8_GLUE": "1", "GLCUDA_NTILE128": "1",
                "GLCUDA_BSTAGE": "1", "GLCUDA_GEMM_N16": "1",
                "GLCUDA_GEMM_N16_PREFETCH": "1", "GLCUDA_ATTN_MMA4": "1",
                "GLCUDA_ATTN_MMA4_REGQ": "1", "GLCUDA_ATTN_MMA4_AV": "1",
                "GLCUDA_TELEMETRY": "1"}

    phase = "elementwise-profile"
    measured = run([exe, MODEL, "profile"], cwd=TREE, env=prod_env, check=False)
    save("elementwise-profile.log", measured)
    if measured.returncode or "[wave120-profile]" not in measured.stdout:
        raise RuntimeError("single-pass production profile failed")

    profile = json.loads(re.search(r"\[wave120-profile\]\s*(\{[^\n]+\})", measured.stdout).group(1))
    stages = [json.loads(x) for x in re.findall(r"\[wave120-stage\]\s*(\{[^\n]+\})", measured.stdout)]
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
    stage_names = [s["name"] for s in stages]
    if stage_names != expected_names or profile["gpu_prefill_ms"] <= 0:
        raise RuntimeError(f"event profile contract failed: {profile}, {stage_names}")
    if profile["oracle_token"] != 3323:
        raise RuntimeError(f"oracle token drifted: {profile}")

    stage_sum = sum(x["total_ms"] for x in stages)
    by_name = {x["name"]: x for x in stages}
    glue_names = ["attn_out_quant", "ffn_residual_norm_quant", "ffn_silu_quant", "ffn_residual_add"]
    ffn_names = ["ffn_down", "ffn_gate_up", "ffn_residual_norm_quant", "ffn_silu_quant",
                 "ffn_residual_add"]
    attention_names = ["qkv", "attn_norm", "attn_kv_write", "attention", "attn_out_quant",
                       "attn_out"]
    ranked = sorted(
        [
            {**s,
             "share_of_gpu_total": s["total_ms"] / profile["gpu_prefill_ms"],
             "share_of_stage_sum": s["total_ms"] / stage_sum if stage_sum else 0.0}
            for s in stages
        ],
        key=lambda s: -s["total_ms"],
    )
    glue_ranked = [s for s in ranked if s["name"] in glue_names]
    summary = {
        "wave": 122,
        "gpu": fields,
        "model": model_meta,
        "profile": profile,
        "stages": stages,
        "ranked_stages": ranked,
        "stage_sum_ms": stage_sum,
        "stage_sum_over_gpu_total": stage_sum / profile["gpu_prefill_ms"],
        "attention_ms": sum(by_name[n]["total_ms"] for n in attention_names),
        "ffn_ms": sum(by_name[n]["total_ms"] for n in ffn_names),
        "glue_ms": sum(by_name[n]["total_ms"] for n in glue_names),
        "glue_ranked": glue_ranked,
        "next_fusion_candidate": glue_ranked[0]["name"] if glue_ranked else None,
        "retention_authority": False,
        "target_15000_tps_achieved": profile["gpu_prefill_tps"] >= 15000,
    }
    report = [
        "# Wave 122 elementwise profile",
        "",
        f"- GPU prefill: {profile['gpu_prefill_ms']:.6f} ms over {profile['prompt_tokens']} tokens "
        f"= {profile['gpu_prefill_tps']:.1f} tok/s",
        f"- Stage sum: {stage_sum:.6f} ms ({summary['stage_sum_over_gpu_total']:.3f}x GPU total)",
        f"- Attention bucket: {summary['attention_ms']:.6f} ms",
        f"- FFN bucket: {summary['ffn_ms']:.6f} ms",
        f"- Glue subtotal: {summary['glue_ms']:.6f} ms",
        f"- Next fusion candidate: {summary['next_fusion_candidate']}",
        "",
        "| stage | ms | share of GPU total | calls | bytes read | macs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in ranked:
        report.append(
            f"| `{s['name']}` | {s['total_ms']:.6f} | {100.0 * s['share_of_gpu_total']:.2f}% | "
            f"{s['calls']} | {s['bytes_read']} | {s['macs']} |"
        )
    (RESULTS / "wave122-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("WAVE122_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
code = code[:start] + replacement + code[end:]
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 122 - elementwise-stage T4 profile\n",
    "\nEvent-timed production profile with the former elementwise bucket split into named substage counters.\n",
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
