"""Build the Wave 80 production Q8 A/B notebook from the verified Wave 79 gate."""

import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "notebooks/glcuda_t4_wave79_mma_av_device.ipynb"
OUTPUT = REPO / "notebooks/glcuda_t4_wave80_mma_av_production.ipynb"

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
notebook["cells"][0]["id"] = "wave80-intro"
notebook["cells"][0]["source"] = [
    "# glcuda Wave 80 - fused compensated-MMA AV production Q8 gate\n",
    "\n",
    "Hard-selected Tesla T4, byte-locked Q8 model, and ten position-balanced production pairs.\n",
]

device_code = "".join(notebook["cells"][1]["source"])
device_code = device_code.replace("Wave 79", "Wave 80")
device_code = device_code.replace("wave79", "wave80")
device_code = device_code.replace(
    "glcuda-t4-wave80-mma-av-device-results.zip",
    "glcuda-t4-wave80-mma-av-production-results.zip",
)
ast.parse(device_code)
notebook["cells"][1]["id"] = "wave80-device-gate"
notebook["cells"][1]["source"] = device_code.splitlines(keepends=True)

production_code = r'''
import math
import statistics
import time

HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
HF_REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"
HF_FILENAME = "qwen2.5-0.5b-instruct-q8_0.gguf"
HF_EXPECTED_BYTES = 675710816
HF_EXPECTED_SHA256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
PRODUCTION_REPEATS = 10
COLD_ITERS = 1
WARMUP_ITERS = 5
MEASURE_ITERS = 10
FIXED_PROMPT = (
    "Measure this deterministic systems prompt carefully. Explain how token-parallel "
    "integer matrix multiplication uses shared memory, Tensor Cores, and fixed launch geometry. "
) * 8


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned_model():
    model = ROOT / HF_FILENAME
    part = ROOT / f"{HF_FILENAME}.part"
    if (
        model.is_file()
        and model.stat().st_size == HF_EXPECTED_BYTES
        and sha256_file(model) == HF_EXPECTED_SHA256
    ):
        return model
    if model.exists():
        model.unlink()
    if part.exists() and part.stat().st_size > HF_EXPECTED_BYTES:
        part.unlink()
    url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILENAME}?download=true"
    for attempt in range(1, 6):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "GwenLand-glcuda-Wave80/1.0", "Accept-Encoding": "identity"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
            status = getattr(response, "status", response.getcode())
            if start and status != 206:
                response.close()
                part.unlink(missing_ok=True)
                start = 0
                response = urllib.request.urlopen(
                    urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "GwenLand-glcuda-Wave80/1.0",
                            "Accept-Encoding": "identity",
                        },
                    ),
                    timeout=120,
                )
                status = getattr(response, "status", response.getcode())
            if status not in (200, 206):
                raise RuntimeError(f"HTTP {status}")
            mode = "ab" if start and status == 206 else "wb"
            downloaded = start
            last_print = time.monotonic()
            with response, part.open(mode) as output:
                while True:
                    chunk = response.read(8 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last_print >= 20:
                        print(
                            f"model fetch {downloaded / (1 << 20):.1f}/"
                            f"{HF_EXPECTED_BYTES / (1 << 20):.1f} MiB",
                            flush=True,
                        )
                        last_print = time.monotonic()
            if part.stat().st_size != HF_EXPECTED_BYTES:
                raise RuntimeError(f"truncated model: {part.stat().st_size}/{HF_EXPECTED_BYTES}")
            actual = sha256_file(part)
            if actual != HF_EXPECTED_SHA256:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"model SHA mismatch: {actual}")
            part.replace(model)
            return model
        except Exception as exc:
            print(f"fetch attempt {attempt}/5 failed: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] * (hi - index) + values[hi] * (index - lo)


def json_lines(haystack, prefix):
    return [
        json.loads(item)
        for item in re.findall(re.escape(prefix) + r"\s*(\{[^\n]+\})", haystack)
    ]


def last_json_line(haystack, prefix):
    matches = json_lines(haystack, prefix)
    if not matches:
        raise RuntimeError(f"dispatch line missing: {prefix}")
    return matches[-1]


def check_dispatch(arm, haystack):
    contract = last_json_line(haystack, "[glcuda-contract]")
    candidate = arm == "candidate_avmma"
    required = {
        "exact_fusion": True,
        "grid2d": True,
        "ntile128": True,
        "bstage": True,
        "gemm_n16": True,
        "gemm_n32": False,
        "attn_mma4": True,
        "attn_mma4_regq": True,
        "attn_mma4_av": candidate,
    }
    bad = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) is not expected
    }
    if bad:
        raise RuntimeError(f"{arm} contract mismatch: {bad}; full={contract}")
    attention = last_json_line(haystack, "[glcuda-attn]")
    expected_attention = "mma4-regq-avmma" if candidate else "mma4-regq"
    if attention.get("path") != expected_attention or attention.get("ntok") != 244:
        raise RuntimeError(f"{arm} attention dispatch drift: {attention}")
    gemm = json_lines(haystack, "[glcuda-gemm]")
    paths = {row.get("path") for row in gemm}
    if paths != {"bstage-n16-m32", "bstage-n16"} or any(
        row.get("ntok") != 244 for row in gemm
    ):
        raise RuntimeError(f"{arm} GEMM dispatch drift: {gemm}")
    return {"contract": contract, "attention": attention, "gemm": gemm}


def timing_rows(rows, expected, label):
    if len(rows) != expected:
        raise RuntimeError(f"{label} count {len(rows)} != {expected}")
    counts = [int(row.get("prompt_tokens", 0)) for row in rows]
    prefill = [float(row.get("prefill_ms", 0)) for row in rows]
    decode = [float(row.get("decode_ms", 0)) for row in rows]
    if len(set(counts)) != 1 or counts[0] != 244:
        raise RuntimeError(f"{label} prompt-token drift: {counts}")
    if not all(math.isfinite(value) and value > 0 for value in prefill + decode):
        raise RuntimeError(f"{label} malformed timings: {rows}")
    return counts[0], prefill, decode


def session_stats(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = data.get("engine") or {}
    workload = data.get("workload") or {}
    if engine.get("name") != "glcuda" or engine.get("backend") != "cuda" or not engine.get("available"):
        raise RuntimeError(f"wrong engine contract: {engine}")
    expected = {
        "engine": "glcuda",
        "kind": "prefill",
        "prompt": FIXED_PROMPT,
        "seed": 42,
        "temperature": 0.0,
        "max_new_tokens": 1,
        "cold_iters": COLD_ITERS,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "verify_against": "glproc",
    }
    for key, value in expected.items():
        if workload.get(key) != value:
            raise RuntimeError(f"workload {key} mismatch: {workload.get(key)!r} != {value!r}")
    validation = data.get("validation") or {}
    parity = [
        finding for finding in validation.get("findings", [])
        if finding.get("check") == "parity"
    ]
    match = re.search(
        r"(\d+)/(\d+) tokens match oracle",
        parity[-1].get("message", "") if parity else "",
    )
    if not match or int(match.group(1)) < 1 or int(match.group(2)) != 50:
        raise RuntimeError(
            f"Q8 first-token oracle failed: {parity[-1] if parity else None}"
        )
    oracle_matches = int(match.group(1))
    measurements = data.get("measurements") or {}
    count, prefill_ms, decode_ms = timing_rows(
        measurements.get("iterations") or [], MEASURE_ITERS, "measured"
    )
    _, cold_prefill_ms, cold_decode_ms = timing_rows(
        measurements.get("cold") or [], COLD_ITERS, "cold"
    )
    median_latency = statistics.median(prefill_ms)
    throughput = [count * 1000.0 / value for value in prefill_ms]
    return {
        "prefill_p50": percentile(throughput, 0.50),
        "prefill_p90": percentile(throughput, 0.90),
        "prefill_p99": percentile(throughput, 0.99),
        "latency_p50_ms": percentile(prefill_ms, 0.50),
        "latency_p90_ms": percentile(prefill_ms, 0.90),
        "latency_p99_ms": percentile(prefill_ms, 0.99),
        "latency_mad_ms": statistics.median(
            abs(value - median_latency) for value in prefill_ms
        ),
        "latency_max_ms": max(prefill_ms),
        "decode_p50": percentile([1000.0 / value for value in decode_ms], 0.50),
        "cold_prefill_ms": cold_prefill_ms[0],
        "cold_decode_ms": cold_decode_ms[0],
        "samples_ms": [round(value, 6) for value in prefill_ms],
        "oracle": f"first-token exact; {oracle_matches}/50 decode prefix",
        "oracle_matches": oracle_matches,
    }


try:
    for key in list(os.environ):
        if key.startswith("GLCUDA_"):
            os.environ.pop(key)

    target = ROOT / "target-wave80"
    build_env = {"CARGO_TARGET_DIR": str(target)}
    build = run(
        [cargo, "build", "--release", "-p", "glbench", "--locked"],
        cwd=TREE,
        env=build_env,
        timeout=7200,
        check=False,
    )
    save("cargo-build-glbench.log", build)
    if build.returncode:
        raise RuntimeError("Wave 80 glbench release build failed")
    glbench = target / "release/glbench"

    model_path = fetch_pinned_model()
    model_meta = {
        "repo": HF_REPO,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }
    (RESULTS / "model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    common_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CARGO_TARGET_DIR": str(target),
        "GLCUDA_FORCE_Q8": "1",
        "GLCUDA_GRID2D": "1",
        "GLCUDA_FUSE_Q8_GLUE": "1",
        "GLCUDA_NTILE128": "1",
        "GLCUDA_BSTAGE": "1",
        "GLCUDA_GEMM_N16": "1",
        "GLCUDA_ATTN_MMA4": "1",
        "GLCUDA_ATTN_MMA4_REGQ": "1",
    }
    arm_env = {
        "retained_regq": {},
        "candidate_avmma": {"GLCUDA_ATTN_MMA4_AV": "1"},
    }
    orders = [
        ["retained_regq", "candidate_avmma"]
        if repeat % 2 == 0
        else ["candidate_avmma", "retained_regq"]
        for repeat in range(PRODUCTION_REPEATS)
    ]

    def run_arm(arm, output, cold, warmup, iterations):
        env = {**common_env, **arm_env[arm]}
        command = [
            glbench,
            "run",
            "--engine", "glcuda",
            "--model", model_path,
            "--prompt", FIXED_PROMPT,
            "--tokens", "1",
            "--cold-iters", str(cold),
            "--warmup", str(warmup),
            "--iters", str(iterations),
            "--temperature", "0",
            "--seed", "42",
            "--kind", "prefill",
            "--verify-against", "glproc",
            "--out", output,
        ]
        return run(command, cwd=TREE, env=env, timeout=14400, check=False)

    dispatch = {}
    for arm in arm_env:
        process = run_arm(arm, RESULTS / f"stabilize-{arm}.json", 0, 0, 1)
        save(f"stabilize-{arm}.log", process)
        if process.returncode:
            raise RuntimeError(f"stabilization failed for {arm}")
        dispatch[arm] = check_dispatch(arm, process.stdout + "\n" + process.stderr)

    records = []
    for repeat, order in enumerate(orders):
        for position, arm in enumerate(order):
            output = RESULTS / f"glbench-r{repeat}-p{position}-{arm}.json"
            process = run_arm(arm, output, COLD_ITERS, WARMUP_ITERS, MEASURE_ITERS)
            save(f"glbench-r{repeat}-p{position}-{arm}.log", process)
            if process.returncode:
                raise RuntimeError(f"glbench failed: {arm} repeat {repeat}")
            check_dispatch(arm, process.stdout + "\n" + process.stderr)
            stats = session_stats(output)
            records.append({
                "repeat": repeat,
                "position": position,
                "arm": arm,
                **stats,
            })
            print(
                f"{arm:16s} r{repeat} p{position}: {stats['prefill_p50']:8.1f} tok/s | "
                f"P50/P90/P99 {stats['latency_p50_ms']:.3f}/"
                f"{stats['latency_p90_ms']:.3f}/{stats['latency_p99_ms']:.3f} ms",
                flush=True,
            )

    summary = {}
    for arm in arm_env:
        rows = [row for row in records if row["arm"] == arm]
        summary[arm] = {
            "prefill_p50_median": statistics.median(row["prefill_p50"] for row in rows),
            "prefill_p90_median": statistics.median(row["prefill_p90"] for row in rows),
            "prefill_p99_median": statistics.median(row["prefill_p99"] for row in rows),
            "latency_p50_median_ms": statistics.median(row["latency_p50_ms"] for row in rows),
            "latency_p90_median_ms": statistics.median(row["latency_p90_ms"] for row in rows),
            "latency_p99_median_ms": statistics.median(row["latency_p99_ms"] for row in rows),
            "latency_mad_median_ms": statistics.median(row["latency_mad_ms"] for row in rows),
            "latency_max_median_ms": statistics.median(row["latency_max_ms"] for row in rows),
            "decode_p50_median": statistics.median(row["decode_p50"] for row in rows),
            "sessions": len(rows),
            "positions": [row["position"] for row in rows],
        }

    paired = []
    for repeat in range(PRODUCTION_REPEATS):
        retained = next(
            row for row in records
            if row["repeat"] == repeat and row["arm"] == "retained_regq"
        )
        candidate = next(
            row for row in records
            if row["repeat"] == repeat and row["arm"] == "candidate_avmma"
        )
        paired.append({
            "repeat": repeat,
            "throughput_delta": candidate["prefill_p50"] / retained["prefill_p50"] - 1.0,
            "absolute_tps": candidate["prefill_p50"] - retained["prefill_p50"],
            "tail_max_delta": candidate["latency_max_ms"] / retained["latency_max_ms"] - 1.0,
            "decode_delta": candidate["decode_p50"] / retained["decode_p50"] - 1.0,
        })

    retained = summary["retained_regq"]
    candidate = summary["candidate_avmma"]
    median_ratio = candidate["prefill_p50_median"] / retained["prefill_p50_median"] - 1.0
    absolute_gain = candidate["prefill_p50_median"] - retained["prefill_p50_median"]
    tail_delta = candidate["latency_max_median_ms"] / retained["latency_max_median_ms"] - 1.0
    oracle_ok = all(row["oracle_matches"] >= 1 for row in records)
    decode_ok = all(row["decode_delta"] >= -0.05 for row in paired)
    tail_ok = tail_delta <= 0.05
    all_positive = all(row["throughput_delta"] > 0 for row in paired)
    retention_ok = median_ratio >= 0.01 and all_positive and decode_ok and tail_ok
    target_reached = candidate["prefill_p50_median"] >= 15_000.0
    if target_reached and retention_ok:
        decision = "GOAL_REACHED"
        verdict = "RETAIN - production gate passes and reaches 15,000 tok/s"
    elif retention_ok:
        decision = "RETAIN_BELOW_GOAL"
        verdict = "RETAIN - production gate passes, still below 15,000 tok/s"
    else:
        decision = "REJECT"
        verdict = "REJECT - candidate misses the production retention gate"

    result = {
        "wave": 80,
        "status": "production_measured",
        "gpu": gpu_line,
        "source_revision": SOURCE_REV,
        "patch_sha256": PATCH_SHA256,
        "model": model_meta,
        "direct_gate": direct_result,
        "method": {
            "repeats": PRODUCTION_REPEATS,
            "sessions": len(records),
            "cold_iters": COLD_ITERS,
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
            "order": orders,
            "quant": "Q8_0",
            "prompt_tokens": 244,
            "seed": 42,
            "temperature": 0.0,
            "oracle": "glproc first-token exact; contiguous decode prefix recorded",
        },
        "summary": summary,
        "paired": paired,
        "comparison": {
            "ratio_of_session_p50_medians": median_ratio,
            "absolute_gain_tps": absolute_gain,
            "median_paired_delta": statistics.median(row["throughput_delta"] for row in paired),
            "worst_paired_delta": min(row["throughput_delta"] for row in paired),
            "all_positive": all_positive,
            "tail_session_max_delta": tail_delta,
            "oracle_ok": oracle_ok,
            "decode_ok": decode_ok,
            "tail_ok": tail_ok,
        },
        "dispatch": dispatch,
        "decision": decision,
        "verdict": verdict,
        "target_15000_tps_achieved": target_reached and retention_ok,
    }
    (RESULTS / "production-records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (RESULTS / "wave80-production.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (RESULTS / "PRODUCTION_SUCCESS.json").write_text(
        json.dumps({"status": "valid", "decision": decision}, indent=2),
        encoding="utf-8",
    )
    write_archive()
    print(json.dumps(result, indent=2))
except Exception:
    (RESULTS / "PRODUCTION_FAILED.txt").write_text(
        traceback.format_exc(), encoding="utf-8"
    )
    write_archive()
    raise
'''

ast.parse(production_code)
notebook["cells"].append(
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "wave80-production-ab",
        "metadata": {},
        "outputs": [],
        "source": production_code.splitlines(keepends=True),
    }
)

OUTPUT.write_text(
    json.dumps(notebook, indent=1) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"generated {OUTPUT.relative_to(REPO)}")
