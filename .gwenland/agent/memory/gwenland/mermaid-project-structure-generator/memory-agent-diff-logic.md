### Agent Diff Logic Implementation

- The system uses a structured diff generator pattern to propose changes.
- Implementation logic relies on the standard `unified diff` format.
- Key focus is on context preservation (header, hunk range, additions/deletions).
- Current approach: `Diff.createPatch` via `diff` library.
- Future optimization: Move to AST-aware diffing to prevent syntax breakage in complex merges.
- Memory context: Stored as a structural reference for future code generation tasks.