# Percival Wave P6 — Implementation Plan
# Generated: 2026-07-30
# Precondition: IP-26 audited — no code change existed to measure (see below)

## Summary

IP-26 ("Reject CPU TENSOR_ALIGNMENT = 32") turned out to require zero code
changes: `glproc::memory::ARENA_ALIGN` and `glictus-caliburni::constants::
TENSOR_ALIGNMENT` are both already `64`, and no allocation site in glproc
specifies 32-byte alignment for a CPU tensor buffer. The finding is an audit
observation about llama.cpp's own `ggml_backend_cpu_buffer_type()`
(ARTX21-F03) — GwenLand never copied that design, so there was nothing to
narrow the gap on. No A/B was run since there was no candidate to compare
against a baseline. Full evidence in the wave-gate report delivered alongside
this document.

Because IP-26 produced no result to gate on, Wave P6 selection below was
made directly from re-reading each of the 31 IMPLEMENTATION-PLAN items
against the current `glproc`/`glcore::gate` source, not from IP-26's
(nonexistent) measurement. Two more items that looked promising on paper
turned out to be **already substantially implemented** during this check
(see Deferred Items) — flagging that up front so the same rediscovery isn't
repeated next sprint.

## Selected Items (in execution order)

### Wave P6-A: IP-08 — Cache-aligned atomics in the worker pool

**Category:** XS
**Impact:** Medium — unverified until measured. `PoolShared` in
`glproc/src/threading.rs:55` packs `generation: AtomicU64` and
`remaining: AtomicUsize` back-to-back with no padding; every worker thread
touches both on every one of the ~170 job dispatches per decode step (per
the module's own doc comment). Adjacent hot atomics written by different
threads are the textbook false-sharing setup, but this repo's history
(rejected-optimizations.md entries 3/8) says isolated-kernel wins routinely
evaporate in production — this must be measured, not assumed.
**Confidence:** High — mechanical, three-field struct, no algorithm change.
**Dependency:** none.

#### What it does
Pad `generation` and `remaining` onto separate cache lines (`#[repr(align(64))]`
wrapper struct or explicit padding bytes between them, no new dependency —
`crossbeam-utils::CachePadded` is not in the dependency tree and this is a
two-field struct, not worth adding it for). `shutdown: AtomicBool` and the
`Mutex`/`Condvar` pair are only touched on the cold park/shutdown path and
don't need padding.

#### Implementation approach
- `glproc/src/threading.rs`: wrap `generation` and `remaining` each in a
  small `#[repr(align(64))] struct Padded<T>(T)` (or two separate structs),
  update the handful of access sites (`load`/`store`/`fetch_sub` calls
  already centralized in this one file).
- No change to `bridge_row_dot`, `moe.rs`, `loader.rs`, or `runner.rs` — the
  atomics are private to `PoolShared`.

#### Gate
- `cargo test -p glproc` green.
- `glbench ab --engine glproc --model <baseline-pkg> --model <candidate-pkg>`
  on the i3-1115G4 reference machine, decode-heavy prompt, 2 repeats,
  thermal-checked (no throttling in any run).
- Decode P50 improves ≥1% on both repeats → merge. Flat or mixed → document
  in `rejected-optimizations.md` per the project's existing pattern (this
  would be entry 9, same shape as 3/8: real in isolation, neutral in
  production).

#### Risk
Low blast radius — a private struct-layout change behind existing
lock/wake logic, no public API or output change. Worst case is "no
measured benefit," not incorrect output (`cargo test` catches any ordering
mistake in the padding wrapper).

---

### Wave P6-B: IP-16 (residual) — Compile-time size assertions for GQ4A/GQ2A blocks

**Category:** XS
**Impact:** Low — this is defense-in-depth, not throughput. `BlockQ4K` and
`BlockQ8K` (`glproc/src/kernels/qdot/q8_k/mod.rs`) already carry
`#[repr(C)]` plus a `const _: () = assert!(size_of::<...>() == N);` —
exactly what IP-16 asks for. `GQ4ABlock`/`GQ2ABlock`
(`glproc/src/kernels/gquant/mod.rs`) are also `#[repr(C)]` but their size
check is a runtime `assert_eq!` inside `#[cfg(test)]`, so a layout-breaking
change to those structs would build clean in release and only fail if the
test suite happens to run.
**Confidence:** High — same one-line pattern already proven twice in this
codebase; copy it.
**Dependency:** none.

#### What it does
Add `const _: () = assert!(std::mem::size_of::<GQ4ABlock>() == GQ4ABlock::BYTES);`
and the `GQ2ABlock` equivalent next to the existing struct definitions in
`glproc/src/kernels/gquant/mod.rs`, matching the pattern at
`q8_k/mod.rs:61,89-93`. Remove the now-redundant runtime assertions from the
test module or leave them (harmless, but redundant).

#### Implementation approach
Two lines in `glproc/src/kernels/gquant/mod.rs`, immediately after the
`GQ4ABlock`/`GQ2ABlock` struct definitions.

#### Gate
`cargo build -p glproc` — a broken layout now fails the build itself, not
just `cargo test`.

#### Risk
None beyond a possible compile error if the structs currently have
undocumented padding — which is exactly the bug class this closes, so a
build failure here would itself be the finding. Note: GQ4A/GQ2A here is the experimental `--gdtqp` LoRA-rank quant format
(GWEN-219), unrelated to the Gamma-dequant benchmark method that shares the
"gdtqp" name in `benchmark/` — don't conflate when writing the commit
message.

---

### Wave P6-C: IP-01 — Consolidate `QuantFormat` dispatch into one table

**Category:** S
**Impact:** Low-Medium — no expected throughput change (the current 4-way
`match QuantFormat { ... }` in `glproc/src/kernels/bridge/mod.rs` is a small
closed set, already branch-predictor-friendly). The value is
maintainability: today, adding a 5th quant format means editing
`match QuantFormat` arms in `bridge/mod.rs`, `loader.rs`, `runner.rs`, and
`threading.rs` (four call sites, confirmed by grep), plus a related
`qdot::supports(fmt)` guard in `moe.rs` that would also need the new
variant. IP-01's per-dtype table pattern ("adding a quant format = adding
one entry") would cut the four matches to one; `moe.rs`'s guard could
likewise become a `DISPATCH[fmt].supported` lookup.
**Confidence:** Medium — not XS/mechanical like P6-A/B; five touch points
raises real integration risk (a silently-wrong branch would misroute a
kernel, not crash).
**Dependency:** none for this narrower CPU-only-table version. (IMPLEMENTATION-PLAN.md
lists IP-01 as unblocking IP-02/04/06/07/10/12/13/14/15/17, but IP-12/13/14/15
are architecture ports not applicable to this CPU tier — see Deferred — so
IP-01 here is scoped to what it actually unblocks on i3: IP-06.)

#### What it does
Replace the scattered `match QuantFormat` arms with one
`static DISPATCH: [FormatOps; N]` (or equivalent const table) indexed by
`QuantFormat as usize`, where `FormatOps` holds function pointers for
`block_numel`, `block_bytes`, `dequant_block`, `row_dot` — mirroring
`type_traits_cpu[]`'s shape but scoped to what glproc actually has today
(4 formats, not llama.cpp's full set).

#### Implementation approach
- Build the table in `glproc/src/kernels/bridge/mod.rs` from the existing
  per-format functions already implemented there (no new kernel code).
- Migrate `loader.rs`, `runner.rs`, `threading.rs` call sites from their
  local `match QuantFormat` to `DISPATCH[fmt as usize].row_dot(...)` (or a
  thin wrapper preserving today's function names so call sites don't
  change, only the body). Update `moe.rs`'s `qdot::supports(*fmt)` guard to
  read from the same table.
- Keep `SimdStrategy` dispatch (Avx512/Avx2/Scalar) nested inside each
  table entry exactly as today — IP-01 is about the per-format axis, not
  the per-ISA axis (that's already handled by `SimdStrategy`).

#### Gate
- `cargo test -p glproc` green — this is a pure refactor, so the existing
  suite (including reference-vector kernel tests) is the correctness gate.
- `cargo clippy -p glproc -D warnings` clean.
- No benchmark required (no algorithmic change expected) — but run one
  `glbench ab` pass as a regression tripwire; flat is the expected and
  acceptable result.

#### Risk
Medium — five call sites to migrate correctly. Mitigate by migrating one
call site at a time behind the existing test suite, not as one large diff.

---

### Wave P6-D: IP-06 — Per-op `use_ref` toggle for differential testing

**Category:** S
**Impact:** Low-Medium (developer velocity, not user-facing perf) — this
project has repeatedly found that isolated kernel probes disagree with
production by 0.07×-2.40× (rejected-optimizations.md's own anti-pattern
note) and that new kernels need parity testing against a reference path
(the Q6_K dequant corruption bug, the falcon/qwen35 Unicode blockers, the
HF `added_tokens` bug — all found by comparing against a reference, not by
reasoning). A per-op scalar/AVX2/AVX-512 override independent of the global
`SimdStrategy` would make "run just this one op through the reference path
while everything else stays fast" a one-line test fixture instead of a
global strategy swap.
**Confidence:** Medium — plumbing a per-call override through existing
call sites is straightforward, but it touches the same files as P6-C,
so sequencing after P6-C (which centralizes those call sites into the
table) makes this a one-file change instead of four.
**Dependency:** P6-C (soft — doable standalone, but strictly easier after).

#### What it does
Add an optional per-op override (e.g., a `force_strategy: Option<SimdStrategy>`
parameter threaded through the dispatch table entry from P6-C) so a test can
pin one op's kernel to `Scalar` while the rest of the run uses whatever
`SimdStrategy` was auto-detected, without a global env var or rebuild.

#### Implementation approach
- After P6-C lands, add the override field to the table lookup call in
  `glproc/src/kernels/bridge/mod.rs`.
- Expose a test-only entry point (`#[cfg(test)]` or a `pub(crate)` hook) —
  this is testing infrastructure, not a user-facing feature, per the
  cross-cutting rule against speculative surface area.

#### Gate
`cargo test -p glproc` green, plus one new test demonstrating the override
actually changes which kernel runs for a single op (e.g. assert scalar and
AVX2 paths produce identical output for a hand-picked input, with the
override forcing scalar while global strategy is AVX2).

#### Risk
Low — additive, opt-in, test-only surface. Worst case is dead code if
never used, not a regression.

## Deferred Items (with reason)

- **IP-12 (AVX-512 vecdot), and by extension any "add wider SIMD"
  proposal** — the whole *category* is closed:
  `rejected-optimizations.md` entry 3 already tested AVX-512F/VNNI-512 on
  this exact machine (thermal risk, then re-measured under explicit
  override — neutral in production both times). Re-proposing native
  AVX-512 vecdot kernels here would be the same experiment a third time.
- **IP-13 (Zen4/Zen5 256-bit GEMM), IP-14 (baseline NEON), IP-15 (SVE VL
  landmine)** — all architecture-specific to hardware GwenLand's reference
  tier doesn't have (AMD Zen, ARM). Genuine gaps in llama.cpp, not
  actionable on the i3-1115G4. Revisit only when/if a Zen or ARM tier is
  brought up, as its own scoped experiment (per rejected-optimizations.md's
  own per-hardware-tier rule).
- **IP-17 (vec_dot_type indirection)** — the concept is already shipped in
  effect: rejected-optimizations.md entry 7 documents that GwenLand's
  answer to Q4_K decode is a load-time repack to Q8_0, which *is* an
  activation-format pre-conversion decision per weight format. Formalizing
  it as a generic trait depends on IP-01 and IP-16 (both touched this
  wave); revisit as a P7 item once P6-C's table exists, if there's still a
  concrete need beyond what repack-to-Q8_0 already does.
- **IP-02, IP-04, IP-07, IP-09, IP-10, IP-11** — all depend on IP-01 or
  IP-03 and are speculative ("GwenLand may need") rather than addressing a
  known bottleneck. IP-04 (dynamic chunk stealing) specifically: the
  worker pool's own doc comment (`threading.rs:6-13`) states row cost is
  uniform for the current dense-matvec decode path, so static contiguous
  chunking is *already* the balance-optimal choice — dynamic stealing
  would add atomic contention for zero balancing gain here, the same shape
  as entry 6's rejected topology change. Only reconsider IP-04 if MoE
  expert-routing imbalance (`moe.rs`) is measured as a real bottleneck —
  that's a data-dependent load pattern where stealing could plausibly
  help, unlike the uniform dense path.
- **IP-03, IP-05, IP-18, IP-19, IP-20, IP-21, IP-25, IP-27, IP-30, IP-31**
  — all are GATE cross-backend scheduling/fusion work (concurrent split
  execution across backends, five-pass backend assignment, plan-time
  fusion across backend boundaries). Checked `glcore::gate::mod.rs` and
  `architecture/GATE/GATE-mapping.md` directly: `BackendKind` is a closed
  4-variant enum but glvulkan/glmetal are stubs and glcuda isn't wired into
  GATE yet — `glproc` is "the first engine wired to it" (GATE's own doc
  comment). None of these items has more than one real backend to
  schedule across right now, so there's nothing to measure or even
  meaningfully implement yet. Revisit once a second engine is actually
  wired into GATE.
- **IP-22, IP-23, IP-24** — target `glvulkan`, which per the cross-cutting
  rule is a different hardware tier and (per GATE-mapping.md) currently a
  stub with no real shader pipeline to cache or fix subgroup sizing in.
- **IP-26** — this sprint's Part 1. Already satisfied, no code change; see
  the wave-gate report.
- **IP-28, IP-29** — GPU-tier (CUDA/Metal), explicitly out of scope per the
  cross-cutting rule.
- **IP-06's formal dependency IP-01** — resolved by sequencing IP-06 as
  P6-D, after P6-C.

## Dependencies Map

```text
P6-A (IP-08)   — independent, do anytime
P6-B (IP-16r)  — independent, do anytime
P6-C (IP-01)   — independent (scoped to i3, no arch-port items to unblock)
                     |
                     v  (soft: easier after, not blocked by)
P6-D (IP-06)   — sequenced after P6-C

Deferred IP-17 -----> depends on P6-C (IP-01) + P6-B (IP-16) if ever picked up
Deferred IP-04 -----> blocked on: MoE load-imbalance measurement (not scheduled)
Deferred GATE items -> blocked on: a second real engine wired into glcore::gate
Deferred arch ports -> blocked on: non-x86/non-Intel reference hardware
```
