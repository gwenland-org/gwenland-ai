"""Build the self-contained Wave 119 T4 kernel-profile notebook."""

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
OUTPUT = ROOT / "notebooks/glcuda_t4_wave119_kernel_profile.ipynb"

patch = subprocess.check_output([
    "git", "diff", "--binary", "--full-index", BASE_REV, "--",
    "Cargo.toml", "Cargo.lock", "glcore", "glproc", "glcuda",
], cwd=ROOT)
patch_sha = hashlib.sha256(patch).hexdigest()
patch_b64 = base64.b64encode(gzip.compress(patch, compresslevel=9)).decode("ascii")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
code = "".join(notebook["cells"][1]["source"])
code = code.replace("wave118", "wave119").replace("Wave 118", "Wave 119")
code = code.replace("WAVE118", "WAVE119")
# Wave 119 extends the existing harness instead of cloning its Rust target.
code = code.replace("wave119_in_process_stability", "wave118_in_process_stability")
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
    if measured.returncode or "[wave119-profile]" not in measured.stdout:
        raise RuntimeError("single-pass production profile failed")

    phase = "sass"
    nvdisasm = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    sass = run([nvdisasm, ROOT / "wave119.cubin"], check=False)
    save("nvdisasm.log", sass)

    phase = "ncu"
    ncu_candidates = [shutil.which("ncu"), "/usr/local/cuda/bin/ncu",
                      "/opt/nvidia/nsight-compute/ncu"]
    ncu = next((str(x) for x in ncu_candidates if x and Path(x).is_file()), None)
    counter_status = "unavailable"
    profiles = {}
    metrics = ",".join([
        "gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active", "smsp__inst_executed.avg.per_cycle_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "dram__bytes_read.sum", "lts__t_bytes_srcunit_tex_op_read.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    ])
    targets = {
        "ffn": "regex:gl_gemm_mma_q8_bstage_n16.*",
        "attention": "regex:gl_attn_mma4_regq_avmma_fused_f32",
    }
    if ncu:
        query = run([ncu, "--query-metrics", "--chip", "tu104"], check=False, timeout=600)
        save("ncu-query.log", query)
        for name, kernel in targets.items():
            prof = run([ncu, "--csv", "--page", "raw", "--target-processes", "all",
                        "--replay-mode", "application", "--cache-control", "none",
                        "--kernel-name-base", "function", "--kernel-name", kernel,
                        "--launch-count", "1", "--metrics", metrics,
                        exe, MODEL, "profile"], cwd=TREE, env=prod_env,
                       check=False, timeout=14400)
            save(f"ncu-{name}.log", prof)
            profiles[name] = prof.returncode
        joined = "\n".join((RESULTS / f"ncu-{x}.log").read_text(encoding="utf-8")
                           for x in targets)
        if "ERR_NVGPUCTRPERM" in joined or "permission" in joined.lower():
            counter_status = "permission-denied"
        elif profiles and all(value == 0 for value in profiles.values()):
            counter_status = "collected"
        else:
            counter_status = "failed"

    summary = {"wave": 119, "gpu": fields, "model": model_meta,
               "event_profile_ok": True, "ncu": ncu,
               "counter_status": counter_status, "profile_returncodes": profiles,
               "sass_available": sass.returncode == 0,
               "retention_authority": False, "target_15000_tps_achieved": False}
    (RESULTS / "wave119-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WAVE119_RESULT", json.dumps(summary, indent=2), flush=True)
    archive()
'''
code = code[:start] + replacement + code[end:]
ast.parse(code)

notebook["cells"][0]["source"] = [
    "# GwenLand glcuda Wave 119 - T4 kernel profile\n",
    "\nObservation-only event, Nsight Compute, resource, and SASS evidence.\n",
]
notebook["cells"][1]["source"] = code.splitlines(keepends=True)
OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")), encoding="utf-8")
ast.parse("".join(json.loads(OUTPUT.read_text(encoding="utf-8"))["cells"][1]["source"]))
print(json.dumps({"notebook": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                  "base_rev": BASE_REV, "head_rev": HEAD_REV,
                  "patch_bytes": len(patch), "patch_sha256": patch_sha}, indent=2))
