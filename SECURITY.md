# Security Policy

GwenLand runs untrusted model files (GGUF / safetensors) through from-scratch
parsers and an mmap loader, and talks to the NVIDIA driver over raw FFI. We
take memory-safety and parsing bugs on those paths seriously.

## Supported versions

GwenLand is pre-1.0. Only the latest state of the `main` branch (and the most
recent tagged release, when one exists) receives security fixes. Older commits
and releases are not patched retroactively.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| Anything older | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through either channel:

- **GitHub**: use *Security → Report a vulnerability* (private vulnerability
  reporting) on the repository.
- **Email**: [jinxsuperdev@gmail.com](mailto:jinxsuperdev@gmail.com) with the
  subject line starting with `[SECURITY]`.

Please include:

- a description of the issue and its impact;
- steps to reproduce — for parser bugs, a minimal crafted model file (or a
  script that generates one) is ideal;
- the commit hash you tested against, your OS, and hardware (CPU/GPU);
- any suggested fix, if you have one.

## What to expect

- **Acknowledgement within 72 hours** of your report.
- An assessment and, for confirmed issues, a fix on `main` as soon as
  practical — typically within 30 days, faster for memory-safety issues on the
  untrusted-input paths.
- Credit in the changelog/release notes if you'd like it (tell us the name to
  use, or that you prefer to stay anonymous).

This is a small open-source project: there is no bug bounty program, but
reports are genuinely appreciated and taken seriously.

## Scope

**In scope:**

- Memory-safety or logic bugs reachable from an untrusted model file — the
  GGUF/safetensors parsers, tokenizer vocab loading, mmap handling.
- Unsound `unsafe` code, including the CUDA Driver FFI boundary.
- Anything that causes GwenLand to make a network call (it is designed to make
  none — see [PRIVACY.md](PRIVACY.md)).

**Out of scope:**

- Undesirable *text output* from a model (hallucinations, offensive content).
  GwenLand executes models; it does not author them.
- Resource exhaustion from deliberately oversized models or prompts on a
  machine that cannot fit them.
- Vulnerabilities in third-party dependencies with no exploitable path through
  GwenLand (still nice to hear about — just lower priority).
