//! T1 — the StableHLO reference interpreter (ARTX12 §2.2).
//!
//! "Did we emit the MLIR we think we did?" T0 (`gljax::oracle::reference`)
//! checks the *algorithm*; T1 checks that gljax's real emitter
//! (`stablehlo::ops::emit_dot_general`, exercised here directly — not
//! reimplemented) produces StableHLO text that a spec-conformant tool
//! interprets the way gljax's own shape/op-count tests assume it does.
//!
//! ⛔ **This tier is invoked as an external binary, never linked** (ARTX12
//! §2.2's own design decision) — linking MLIR/LLVM into gljax's test build
//! would contradict ARTX01's pure-Rust, zero-heavy-dependency posture.
//! `stablehlo-translate` is looked up via `STABLEHLO_TRANSLATE`, and every
//! test here SKIPs with a clear message when it is absent — the same
//! convention every PJRT-gated test in this crate already uses (see
//! `gljax/README.md`, `runtime/mod.rs`'s "Split by what has been executed").
//!
//! ⚠️ **Honest scoping.** This environment has neither the binary nor a way
//! to install it (no network access to OpenXLA's build infra from this
//! sandbox), so the "binary present" branch below has never actually run.
//! It is written against the real `stablehlo-translate --interpret` +
//! `check` dialect contract — confirmed against
//! `stablehlo/tests/interpret/dot_general.mlir` in the openxla/stablehlo
//! repository directly, not guessed — but "written against a confirmed
//! contract" and "verified to work" are different claims, and only the first
//! one is made here.

use std::path::PathBuf;
use std::process::Command;

use gljax::oracle::reference::{dot_general_f64, TensorF64};
use gljax::stablehlo::emitter::MlirEmitter;
use gljax::stablehlo::ops::{emit_constant_dense_f32, emit_dot_general, DotDimensionNumbers};
use gljax::stablehlo::types::{DType, Shape};
use gljax::{DotAlgorithm, DotNumerics};

fn stablehlo_translate_path() -> Option<PathBuf> {
    std::env::var_os("STABLEHLO_TRANSLATE").map(PathBuf::from)
}

/// Builds a self-contained, zero-argument module: two `stablehlo.constant`
/// operands (gljax's real dense-constant emitter, not hand-written text),
/// one `stablehlo.dot_general` (gljax's real emitter), and a hand-written
/// `check.expect_eq_const` comparing the result against T0's answer for the
/// same inputs — the pattern StableHLO's own interpreter test suite uses
/// (`check.expect_eq_const %result, dense<[...]> : tensor<...>`), confirmed
/// against `stablehlo/tests/interpret/dot_general.mlir` directly.
///
/// Every value here is a small integer exactly representable in f32, so the
/// comparison can be exact equality — no need to guess at a
/// tolerance-comparison op's argument syntax for a binary this environment
/// cannot run to check against.
fn build_dot_general_check_module() -> (String, Vec<f32>) {
    let lhs_shape = Shape::new([2, 2], DType::F32);
    let rhs_shape = Shape::new([2, 2], DType::F32);
    let lhs_data = [1.0f32, 2.0, 3.0, 4.0];
    let rhs_data = [5.0f32, 6.0, 7.0, 8.0];
    let dnums = DotDimensionNumbers {
        lhs_batching: vec![],
        rhs_batching: vec![],
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
    };

    // Cross-check against T0 independently, so the expected constant below
    // is not hand-copied from the same arithmetic emit_dot_general performs.
    let lhs_f64 = TensorF64::from_data(&[2, 2], lhs_data.iter().map(|&v| v as f64).collect());
    let rhs_f64 = TensorF64::from_data(&[2, 2], rhs_data.iter().map(|&v| v as f64).collect());
    let want = dot_general_f64(&lhs_f64, &rhs_f64, &dnums);
    let want_f32: Vec<f32> = want.data().iter().map(|&v| v as f32).collect();

    let mut e = MlirEmitter::new();
    let lhs = emit_constant_dense_f32(&mut e, &lhs_data, &lhs_shape).expect("lhs constant");
    let rhs = emit_constant_dense_f32(&mut e, &rhs_data, &rhs_shape).expect("rhs constant");
    let out_shape = Shape::new([2, 2], DType::F32);
    let result = emit_dot_general(
        &mut e,
        lhs,
        rhs,
        &dnums,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
        DotNumerics::Default,
        None,
    );
    let body = e.into_body();

    let expected_literal = format!(
        "dense<[[{}, {}], [{}, {}]]> : tensor<2x2xf32>",
        want_f32[0], want_f32[1], want_f32[2], want_f32[3]
    );
    let module = format!(
        "module @t1 {{\n  func.func @main() {{\n{body}    \
         \"check.expect_eq_const\"({result}) {{value = {expected_literal}}} : (tensor<2x2xf32>) -> ()\n    \
         \"func.return\"() : () -> ()\n  }}\n}}\n"
    );
    (module, want_f32)
}

/// Runs unconditionally (no `stablehlo-translate` needed) — this is the part
/// of the harness this environment *can* verify: that
/// `build_dot_general_check_module` itself doesn't panic and produces text
/// that at least looks like the confirmed real pattern. It cannot verify a
/// real interpreter accepts the text; see this file's module docs.
#[test]
fn build_dot_general_check_module_produces_well_formed_text() {
    let (module, want) = build_dot_general_check_module();
    assert_eq!(want, vec![19.0, 22.0, 43.0, 50.0], "T0 cross-check must match the known product");
    assert!(module.contains("stablehlo.dot_general"), "{module}");
    assert!(module.contains("check.expect_eq_const"), "{module}");
    assert!(module.contains("dense<[[19, 22], [43, 50]]>"), "{module}");
    assert_eq!(
        module.matches('{').count(),
        module.matches('}').count(),
        "unbalanced braces:\n{module}"
    );
}

#[test]
fn t1_dot_general_matches_spec() {
    let Some(bin) = stablehlo_translate_path() else {
        eprintln!("SKIP t1: stablehlo-translate not found (set STABLEHLO_TRANSLATE)");
        return;
    };

    let (module, want) = build_dot_general_check_module();
    let tmp = std::env::temp_dir().join(format!("gljax_t1_dot_general_{}.mlir", std::process::id()));
    std::fs::write(&tmp, &module).expect("write temp module");

    let output = Command::new(&bin)
        .arg("--interpret")
        .arg(&tmp)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", bin.display()));

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "stablehlo-translate --interpret rejected gljax's own emitted dot_general \
         (want {want:?}):\nmodule:\n{module}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// The `algorithm`/`precision_config` mutual exclusivity A8.α relies on is a
/// StableHLO spec property, not just gljax's own convention — if the real
/// interpreter accepts a module using `algorithm` the same way it accepts
/// `precision_config`, that is independent confirmation the emitted
/// `#stablehlo.dot_algorithm<...>` spelling (verified against
/// `StablehloAttrs.td` in Wave B1, never against a real parser) is not just
/// syntactically plausible but actually accepted.
#[test]
fn t1_dot_general_with_an_explicit_algorithm_is_accepted() {
    let Some(bin) = stablehlo_translate_path() else {
        eprintln!("SKIP t1: stablehlo-translate not found (set STABLEHLO_TRANSLATE)");
        return;
    };

    let lhs_shape = Shape::new([2, 2], DType::F32);
    let rhs_shape = Shape::new([2, 2], DType::F32);
    let dnums = DotDimensionNumbers {
        lhs_batching: vec![],
        rhs_batching: vec![],
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
    };

    let mut e = MlirEmitter::new();
    let lhs =
        emit_constant_dense_f32(&mut e, &[1.0, 2.0, 3.0, 4.0], &lhs_shape).expect("lhs constant");
    let rhs =
        emit_constant_dense_f32(&mut e, &[5.0, 6.0, 7.0, 8.0], &rhs_shape).expect("rhs constant");
    let out_shape = Shape::new([2, 2], DType::F32);
    let result = emit_dot_general(
        &mut e,
        lhs,
        rhs,
        &dnums,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
        DotNumerics::Algorithm(DotAlgorithm::F32F32F32),
        None,
    );
    let body = e.into_body();
    let module = format!(
        "module @t1 {{\n  func.func @main() {{\n{body}    \
         \"check.expect_eq_const\"({result}) {{value = dense<[[19.0, 22.0], [43.0, 50.0]]> : tensor<2x2xf32>}} \
         : (tensor<2x2xf32>) -> ()\n    \"func.return\"() : () -> ()\n  }}\n}}\n"
    );

    let tmp = std::env::temp_dir().join(format!("gljax_t1_dot_algorithm_{}.mlir", std::process::id()));
    std::fs::write(&tmp, &module).expect("write temp module");
    let output = Command::new(&bin)
        .arg("--interpret")
        .arg(&tmp)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", bin.display()));
    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "stablehlo-translate rejected an explicit DotAlgorithm:\nmodule:\n{module}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
