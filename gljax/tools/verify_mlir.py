"""Parse gljax-emitted StableHLO through a real MLIR parser.

⛔ There is no PJRT plugin for Windows (see ``gljax/README.md``), so
``PJRT_Client_Compile`` — the validation step ARTX01 §2.4 says to rely on —
cannot run on the reference machine. A locally installed ``jaxlib`` carries the
same MLIR parser and the StableHLO dialect, which is enough to answer the one
question the Rust tests cannot: *does this module parse and verify?*

What this checks:  syntax, dialect registration, op signatures, region
                   structure, type well-formedness.
What it does NOT:  whether the computation is the one you meant, or whether a
                   PJRT plugin will compile it. Those need ARTX12 Part B and a
                   plugin respectively.

Usage::

    python gljax/tools/verify_mlir.py                # traces every module/policy
    python gljax/tools/verify_mlir.py module.mlir    # a file you already dumped

Requires ``jaxlib`` on the Python path and ``cargo`` on PATH.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (module, precision policy) pairs the example knows how to emit.
CASES = [
    ("mlp", "bf16"),
    ("mlp", "f32"),
    ("mlp", "f64"),
    ("ops", "bf16"),
    ("ops", "f32"),
    ("tiny", "bf16"),
    ("tiny", "f32"),
    ("tiny", "f64"),
    ("block", "bf16"),
    ("block", "f32"),
    ("kv", "bf16"),
    ("kv", "f32"),
]


def make_context():
    """An MLIR context with the StableHLO and func dialects registered."""
    try:
        # `import jax` first, always. On Windows `jaxlib.mlir._mlir_libs._mlir`
        # is a stub that resolves against `jax_common.dll` next to it, and
        # importing it directly fails with "DLL load failed" — importing jax is
        # what puts that directory on the DLL search path.
        import jax  # noqa: F401
        import jaxlib.mlir.ir as ir  # noqa: F401
        from jax._src.interpreters import mlir as jmlir
    except ImportError as exc:  # pragma: no cover - environment-dependent
        sys.exit(f"SKIP: jaxlib/jax not importable ({exc})")
    return jmlir.make_ir_context()


def parse(ctx, text: str) -> str | None:
    """Returns None on success, or the diagnostic on failure."""
    import jaxlib.mlir.ir as ir

    try:
        with ctx:
            ir.Module.parse(text)
    except Exception as exc:  # MLIR raises a bare Exception subclass
        return str(exc).replace("\n", "\n    ")
    return None


def dump(module: str, policy: str) -> str:
    proc = subprocess.run(
        [
            "cargo", "run", "-q", "-p", "gljax",
            "--example", "dump_mlir", "--", module, policy,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"cargo run failed for {module}/{policy}:\n{proc.stderr}")
    return proc.stdout


def main() -> int:
    ctx = make_context()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        err = parse(ctx, path.read_text(encoding="utf-8"))
        if err:
            print(f"FAIL {path}\n    {err}")
            return 1
        print(f"ok   {path}")
        return 0

    failures = 0
    for module, policy in CASES:
        text = dump(module, policy)
        err = parse(ctx, text)
        label = f"{module}/{policy}"
        if err:
            failures += 1
            print(f"FAIL {label}\n    {err}")
        else:
            lines = sum(1 for line in text.splitlines() if line.strip())
            print(f"ok   {label:<12} {lines:>3} lines parsed and verified")

    print()
    if failures:
        print(f"{failures} of {len(CASES)} modules did not parse")
        return 1
    print(f"all {len(CASES)} modules parse and verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
