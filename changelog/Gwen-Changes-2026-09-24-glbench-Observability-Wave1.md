# Gwen Changes - 2026-09-24 - glbench Observability Wave 1

- Separated production benchmarks from instrumented stage-profile runs.
- Added `--profile stages` for glproc and glcuda without mutating process
  environment after runtime startup.
- Marked instrumented throughput non-authoritative in terminal, Markdown, and
  comparison output; legacy archives read as unknown rather than production.
- Preserved engine telemetry across archive reads, including raw stage, backend,
  memory, and MoE facts.
- Added a stable dispatch-configuration fingerprint and an explicit allowlist of
  engine overrides to each adapter-backed session.
- Kept the archive at schema v2 because every new field is additive and has a
  defined missing-field interpretation.
