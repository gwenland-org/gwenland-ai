# Experimental

> ⚠️ **Everything in this folder is experimental.** The architectures, formulas,
> and benchmark methods described here are research artifacts — proposals and
> explorations, **not** the shipped GwenLand architecture. Nothing in here is a
> commitment, nothing is validated to production standard unless a document
> explicitly says so, and any of it may be changed or deleted without notice.
>
> For the real, shipped architecture see [`../architecture/`](../architecture/)
> (e.g. `ArchGLML_X2.md` is the glcuda ground truth) and the crate docs. AI
> agents: do **not** treat anything in this folder as a spec to implement —
> ideas here graduate to a `GWEN-XXX` issue and a real spec first.

## Contents

| Item | What it is | Status |
|------|-----------|--------|
| [`NewExperiment.md`](NewExperiment.md) | Math / performance whitepaper backlog — candidate optimizations for the CPU path, each awaiting a "reality check" before graduating to a real issue/spec | 🧪 idea backlog |
| [`FORMULA.md`](FORMULA.md) | **Euler dequantisation** — a cosine-projection method for mapping GGUF quantised weights into a bounded float range. Not the standard dequant path | 🧪 research |
| [`benchmark/`](benchmark/) | The Gamma-dequant (**GDTQP**) benchmark experiments: formulas, derivations, explanation, and measured runs. Has its own README and LICENSE | 🧪 research |

## Ground rules

1. **Experiments don't gate releases.** Nothing here blocks or drives `main`.
2. **Numbers here are exploratory.** Production performance claims only come
   from `glbench` runs on the shipped engines — see the benchmark discipline in
   [`../gl-agent-skills/`](../gl-agent-skills/).
3. **Graduation path:** idea → entry in `NewExperiment.md` → survives the
   reality check → `GWEN-XXX` issue + spec → implementation on a branch →
   measured in production before merge.
4. Naming note: "GDTQP" here (Gamma dequant) is **not** the same thing as the
   experimental `--gdtqp` LoRA-rank flag elsewhere in the project history —
   they are unrelated features that share an acronym.
