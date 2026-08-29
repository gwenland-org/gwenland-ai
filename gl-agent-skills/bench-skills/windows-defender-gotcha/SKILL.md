# Windows Defender Gotcha

> **Domain:** bench-skills
> **Applies to:** ANY benchmark or timing-sensitive run on Windows
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] Defender exclusions exist for **both** the `target/` build directory **and** the GGUF model directory — verified, not remembered.
- [ ] If I can't verify exclusions, I don't benchmark on this machine — I either fix it or say the numbers are polluted.

## Context

Windows Defender real-time protection rescans large files when they're
rebuilt or first mapped — which describes exactly a benchmark loop: every
`cargo build` produces a "new" multi-MB binary, and every model load maps a
multi-GB GGUF. The scan competes for disk, memory bandwidth, and CPU during
precisely the cold phase being measured. Measured pollution on this project:
**2–4×** on affected runs — worse than almost any real regression we've ever
chased. Un-excluded Windows benchmarks are not noisy; they are *wrong*.

## Rules

1. **Exclude before measuring — both paths:**
   - the repo's `target/` directory (the freshly built binaries), and
   - the directory holding the GGUF models.

   GUI: *Windows Security → Virus & threat protection → Manage settings →
   Exclusions → Add or remove exclusions* → add both folders.

   Admin PowerShell equivalent:
   ```powershell
   Add-MpPreference -ExclusionPath "C:\path\to\GwenLand\target"
   Add-MpPreference -ExclusionPath "C:\path\to\models"
   # verify:
   Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
   ```
2. **Verify, don't assume** — exclusions get lost (Windows resets, folder
   moves, new machine). The `Get-MpPreference` check is part of the
   benchmark ritual, every session.
3. **NEVER quote a Windows benchmark without stating the exclusion was
   verified.** A session archived from an un-excluded machine is tainted;
   mark it as such if it must be kept at all.
4. **Cold-start numbers are the most contaminated** (first mmap of the
   model triggers the scan) — a "page-in cost" measured without exclusions
   is mostly Defender, and will send you chasing a phantom mmap regression.
5. **Suspect Defender on symptoms:** unexplained 2–4× slowdowns, first-run
   much slower than usual, high `Antimalware Service Executable` CPU during
   the run → stop, check exclusions, rerun. Do not start diagnosing the
   engine from a possibly-tainted run.
6. **Scope the exclusions narrowly** (those two directories — not `C:\`).
   This is a security trade the user makes knowingly; documenting it is part
   of why this file exists.

## ✅ Correct Pattern

```text
Windows bench ritual:
1. Get-MpPreference … → confirm both paths listed.
2. cargo build --release -p glbench
3. glbench run … --out benchmarks/… (cold + warm reported separately)
4. PR/gate note: "measured on <box>, Defender exclusions verified".
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Benchmarked right after rebuild, decode is suddenly 3× slower — must be
   my change" → it was the rescan of the new binary. Check exclusions first.

❌ Disabling Defender entirely "for clean numbers" — narrow exclusions
   exist precisely so you don't have to do this.

❌ Archiving un-excluded sessions into benchmarks/ as if comparable.
```

## GwenLand-Specific Notes

- Linux is the cleaner benchmarking OS for this project (RAPL energy data
  is also Linux-only) — when a Windows number and a Linux number disagree
  wildly on the same hardware class, audit the Windows environment first.
- This gotcha also pollutes *test* timing (parity suites mapping models) —
  a mysteriously slow first test run on Windows has the same suspect.

## Related Skills

- [glbench-usage.md](glbench-usage.md)
- [measurement-discipline.md](measurement-discipline.md)
