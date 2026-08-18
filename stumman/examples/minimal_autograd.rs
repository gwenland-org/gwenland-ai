//! Stummañ Kevskrid: minimal autograd example.
//!
//! Shows the whole M1 API in one file:
//!   1. Create tensors on the CPU backend (GlProc)
//!   2. Attach them to a tape so ops get recorded
//!   3. Forward pass: loss = mean(relu(x @ w + b))
//!   4. backward()
//!   5. Read gradients back out of the tape
//!   6. The frozen-operand case, which is the shape LoRA training takes
//!
//! Run: cargo run --example minimal_autograd

use std::sync::{Arc, Mutex};
use stumman::{GlProc, Tape, Tensor};

fn show(label: &str, data: &[f32]) {
    let cells: Vec<String> = data.iter().map(|v| format!("{v:6.3}")).collect();
    println!("  {label:<12} [{}]", cells.join(", "));
}

fn main() -> anyhow::Result<()> {
    // The tape records every op whose operands are tracked.
    let tape = Arc::new(Mutex::new(Tape::new()));

    // A batch of 2 samples, 3 features each.
    let x = Tensor::<GlProc>::from_vec(
        vec![
            1.0, 0.5, -1.0, //
            0.0, 2.0, 1.0,
        ],
        &[2, 3],
    )?
    .with_grad(tape.clone());

    // Weights: 3 features in, 2 units out.
    let w = Tensor::<GlProc>::from_vec(
        vec![
            0.1, 0.2, //
            -0.3, 0.4, //
            0.5, 0.6,
        ],
        &[3, 2],
    )?
    .with_grad(tape.clone());

    // Bias. Broadcasting is not implemented in M1, so this carries the full
    // [2,2] shape rather than a per-unit [2].
    let b = Tensor::<GlProc>::from_vec(vec![0.1, 0.1, 0.1, 0.1], &[2, 2])?
        .with_grad(tape.clone());

    // ── Forward ───────────────────────────────────────────────────────────
    let xw = x.matmul(&w)?; // [2,3] @ [3,2] -> [2,2]
    let xwb = xw.add(&b)?; // [2,2]
    let y = xwb.relu()?; // [2,2]
    let loss = y.mean()?; // [1], a scalar loss

    println!("Forward");
    show("x @ w", &xw.to_vec()?);
    show("+ b", &xwb.to_vec()?);
    show("relu", &y.to_vec()?);
    show("loss", &loss.to_vec()?);

    {
        let guard = Tape::lock(&tape);
        println!("  recorded:    {:?}", guard.op_names());
    }

    // ── Backward ──────────────────────────────────────────────────────────
    // backward() seeds the final node with ones, so this differentiates
    // sum(loss). loss is already a scalar, so that is just dL/dL = 1.
    {
        let mut guard = Tape::lock(&tape);
        guard.backward()?;

        println!("\nBackward");
        for (label, id) in [("grad x", x.id()), ("grad w", w.id()), ("grad b", b.id())] {
            match guard.grad(id) {
                Some((data, shape)) => {
                    println!("  {label} shape {shape:?}");
                    show("", data);
                }
                None => println!("  {label}: none (frozen)"),
            }
        }

        // Calling backward() twice without this would be rejected: the seed
        // accumulates onto itself and the gradient comes out tripled.
        guard.zero_grad();
    }

    // ── Frozen operands ───────────────────────────────────────────────────
    // A tensor that never joined the tape gets no gradient, and that is not an
    // error. It is how LoRA works: a frozen base weight feeding a trainable
    // activation.
    {
        let tape2 = Arc::new(Mutex::new(Tape::new()));
        let adapter = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?
            .with_grad(tape2.clone());
        let frozen_base = Tensor::<GlProc>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;

        let out = adapter.matmul(&frozen_base)?;

        let mut guard = Tape::lock(&tape2);
        guard.backward()?;

        println!("\nFrozen base weight");
        show("out", &out.to_vec()?);
        println!(
            "  adapter grad: {}",
            if guard.grad(adapter.id()).is_some() {
                "yes"
            } else {
                "no"
            }
        );
        println!(
            "  base grad:    {}   (never tracked, so never computed)",
            if guard.grad(frozen_base.id()).is_some() {
                "yes"
            } else {
                "no"
            }
        );
    }

    println!("\nM1 autograd engine: forward recorded, backward replayed.");
    Ok(())
}
