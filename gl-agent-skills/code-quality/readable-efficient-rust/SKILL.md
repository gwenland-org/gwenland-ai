---
name: readable-efficient-rust
description: Implement or review GwenLand Rust so a reader can follow the contract, control flow, ownership, and failure behavior without tracing incidental detail, while the executed path does no accidental work. Use for new implementation, refactors, hot loops, parsers, runtime dispatch, FFI boundaries, and code or comment reviews in any `gl*` crate.
---

# Readable, Efficient Rust

Make the source explain its decisions. Make the hot path do only the work those decisions require.

## Workflow

1. State the contract: accepted input, output, ownership, failure modes, and the performance-sensitive dimension.
2. Separate validation, planning, and execution. Validate once at the boundary; choose a strategy once; keep the repeated loop focused on compute and data access.
3. Make the data path visible. A reviewer must be able to locate every clone, allocation, lock, dynamic dispatch, and synchronization point.
4. Measure production behavior before claiming an improvement. Read [measurement discipline](../../bench-skills/measurement-discipline/SKILL.md) for a performance change.

## Shape code around decisions

Prefer named, linear phases over one function that mixes parsing, policy, allocation, and compute.

```rust
fn run(request: &Request, weights: &[u8], out: &mut [f32]) -> Result<(), GlError> {
    let shape = validate_request(request, weights, out)?;
    let plan = DispatchPlan::select(request, shape)?;
    execute(plan, weights, out);
    Ok(())
}
```

Keep helpers at one abstraction level. `validate_request` answers whether the request is legal; `DispatchPlan::select` chooses a strategy; `execute` performs it. Do not hide all three behind a generic helper named `process`.

Use early returns for invalid or unsupported input. Keep the successful path left-aligned. Name a branch after the policy it represents, not after an incidental boolean. Do not introduce an abstraction just to avoid repeating two clear lines.

```rust
if !plan.supports(dtype) {
    return Err(GlError::UnsupportedDtype(dtype.to_string()));
}

let use_repacked_weights = plan.requires_q8_repack();
```

## Keep the hot path boring

In a loop executed per token, row, or element:

- Borrow slices instead of cloning buffers.
- Allocate and resize before the loop, not inside it.
- Hoist invariant checks, capability tests, and dispatch choices.
- Use direct data-oriented names (`weights`, `row`, `acc`, `out`) where their scope is small.
- Split a complicated loop only when the split does not obscure data locality or add repeated setup.

```rust
let scale = row_scale[row_index];
let row = &weights[row_start..row_end];
let mut acc = 0.0_f32;

for (&weight, &activation) in row.iter().zip(activations) {
    acc += weight * activation;
}
out[row_index] = acc * scale;
```

Do not replace a clear contiguous loop with iterator layers, trait objects, allocations, or closures merely for style. Conversely, do not write manual indexing when a slice iterator makes bounds and pairing clearer at equal cost. Confirm a material trade-off with a benchmark or generated-code inspection.

## Write comments that teach the decision

Write for a capable student first: define the local idea, state the reason, and use the domain term only when it adds precision. A professional should be able to scan the same comment without rereading it.

Comment only a non-obvious decision, invariant, boundary, numerical choice, ownership rule, or performance constraint. Let clear names and structure explain ordinary assignments, branches, and loops.

```rust
// Q8 blocks have independent scales, so integer sums cannot cross blocks.
accumulate_one_block(block, &mut out);

// SAFETY: `offset + len` was range-checked above, and the GGUF alignment
// check guarantees this cast is valid for `f32`.
let values = unsafe { cast_f32(bytes, offset, len) };
```

Avoid comments that merely translate syntax or make a broad claim without a reason.

```rust
// Increment the index.
index += 1;

// This is faster.
run_kernel();
```

For a public item, say what it does, what the caller must provide, and any meaningful failure or cost. For an internal comment, say why this implementation exists instead of an obvious alternative. Keep it adjacent to the code it constrains and update or remove it with the code.

## Make ownership and failure explicit

Pass `&[T]` and `&mut [T]` when the caller owns storage. Return an owned value only when ownership changes. Keep unsafe code in the smallest possible wrapper, state its invariants next to it, and expose a safe API; read [unsafe rules](../../rust-skills/unsafe-rules/SKILL.md) before changing an unsafe path.

Represent expected failures with the project error type and enough context to act on the error. Do not encode a normal fallback as `Option` or silently discard it; read [error handling](../../rust-skills/error-handling/SKILL.md).

## Review checklist

- Can a reviewer identify the contract and the successful execution path in one pass?
- Are validation, strategy selection, and repeated execution separated?
- Is every allocation, clone, lock, dynamic dispatch, and synchronization point intentional?
- Are names specific at boundaries and compact inside a tight local loop?
- Does every comment explain a decision or invariant that the code cannot state clearly on its own?
- Can a capable student understand the comment without external context?
- Does the refactor preserve error behavior, aliasing rules, and numerical semantics?
- Does a measured production result support any performance claim?

## Related skills

- [check existing tests](../../before-coding/check-existing-tests/SKILL.md)
- [measurement discipline](../../bench-skills/measurement-discipline/SKILL.md)
- [memory safety](../../rust-skills/memory-safety/SKILL.md)
