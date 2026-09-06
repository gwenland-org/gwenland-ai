"""Build the pinned Wave58 diagnostic notebook from the archived Wave57 stack."""
import ast
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
nb = json.loads((repo / 'notebooks/glcuda_t4_wave57_h2h_q8.ipynb').read_text())
cells = nb['cells'][:5]
cells[0]['source'] = '# Wave58: Q8 teacher-forced correctness and production profile\n\nDiagnostic only; 50/50 acceptance is not relaxed. No llama.cpp rebuild.\n'
for cell in cells:
    cell['source'] = cell['source'].replace('wave57-h2h-q8-v2', 'wave58-q8-diagnostic-v1').replace('glcuda-wave57-h2h-q8-', 'glcuda-wave58-').replace('glcuda_t4_wave57_h2h_q8_results.zip', 'glcuda_t4_wave58_results.zip')
source = (repo / 'glcuda/examples/wave58_logits.rs').read_text()
code = 'diagnostic_source = ' + repr(source) + '\n'
code += '''
if not globals().get("MODEL_OK"):
    raise RuntimeError("Model gate failed")
(TREE / "glcuda/examples/wave58_logits.rs").write_text(diagnostic_source)
prompt = "Measure this deterministic systems prompt carefully. Explain how token-parallel integer matrix multiplication uses shared memory, Tensor Cores, and fixed launch geometry. " * 8
prompt_path = RESULTS / "prompt.txt"
prompt_path.write_text(prompt)
env = {"CUDA_VISIBLE_DEVICES":"0", "GLCUDA_FORCE_Q8":"1", "GLCUDA_GRID2D":"1", "GLCUDA_FUSE_Q8_GLUE":"1", "GLCUDA_NTILE128":"1", "GLCUDA_BSTAGE":"1", "GLCUDA_GEMM_N16":"1", "GLCUDA_ATTN_MMA4":"1", "GLCUDA_ATTN_MMA4_REGQ":"1", "CARGO_TARGET_DIR":str(TARGET)}
try:
    p = run([CARGO,"build","--release","-p","glcuda","--example","wave58_logits","--locked"], cwd=TREE, env=env, timeout=1800, check=False)
    save_log("diagnostic-build.log",p)
    if p.returncode: raise RuntimeError("diagnostic compile failed")
    results = {}
    for name, extra, unset in [
        ("retained", [], []),
        ("sequential", ["sequential"], []),
        ("pre_n16", [], ["GLCUDA_GEMM_N16","GLCUDA_ATTN_MMA4_REGQ"]),
        ("rows", [], ["GLCUDA_ATTN_MMA4","GLCUDA_ATTN_MMA4_REGQ"]),
    ]:
        prefix = ["env"] + [part for key in unset for part in ("-u",key)]
        if name == "rows": prefix += ["GLCUDA_ATTN_ROWS=1"]
        p = run(prefix + [TARGET / "release/examples/wave58_logits", MODEL_PATH, prompt_path] + extra, cwd=TREE, env=env, timeout=1800, check=False)
        save_log(name+"-logits.log", p)
        if p.returncode: raise RuntimeError(name+" diagnostic failed")
        results[name] = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
        if len(results[name]) != 50: raise RuntimeError("missing logits rows")
        print(name, "top mismatches", [r["step"] for r in results[name] if r["cpu_top"] != r["gpu_top"]], flush=True)
    (RESULTS / "logits.json").write_text(json.dumps(results,indent=2))
    for name, overrides in [("unprofiled",{}),("events",{"GLCUDA_TELEMETRY":"1"})]:
        p = run([GLBENCH,"run","--engine","glcuda","--model",MODEL_PATH,"--prompt",prompt,"--tokens","1","--kind","prefill","--temperature","0","--seed","42","--cold-iters","1","--warmup","5","--iters","10","--verify-against","glproc","--out",RESULTS/(name+".json")], cwd=TREE,env={**env,**overrides},timeout=1800,check=False)
        save_log(name+".log",p)
        if p.returncode: raise RuntimeError(name+" profiling failed")
    (RESULTS / "DIAGNOSTIC_COMPLETE.json").write_text(json.dumps({"model":MODEL_META,"gpu":GPU_INFO,"note":"Diagnostic completed; no acceptance claim, investigate all parity failures."},indent=2))
    archive()
except Exception:
    fail_phase("wave58-diagnostic")
'''
cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code})
for i, cell in enumerate(cells):
    cell['id'] = f'wave58-{i:02}'
    if cell['cell_type'] == 'code':
        ast.parse(cell['source'])
nb['cells'] = cells
(repo/'notebooks/glcuda_t4_wave58_diagnostic.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
print('Wave58 notebook generated and syntax checked')
