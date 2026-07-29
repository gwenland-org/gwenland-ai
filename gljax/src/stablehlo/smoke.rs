//! The Wave A1 smoke module: `f32 + f32` on rank-0 tensors.
//!
//! This is the smallest program that exercises the whole chain — emit text,
//! compile through PJRT, upload two host buffers, execute, read one back — so
//! it is the thing that proves the FFI works before any model is involved.
//!
//! Two spellings of the same module are provided on purpose. ARTX02 claims
//! PJRT accepts both the generic and the pretty form; compiling both is how
//! that claim gets checked instead of assumed (P2).

use crate::stablehlo::emitter::MlirEmitter;
use crate::stablehlo::ops::emit_add;
use crate::stablehlo::types::{DType, Shape};

/// The pretty (sugar) form, verbatim.
///
/// gljax's emitter does not produce this shape; it exists so the smoke test
/// can confirm the MLIR parser inside the plugin accepts it, which is what
/// makes ARTX02's "PJRT accepts both" a measured statement.
pub const ADD_SCALAR_MODULE_PRETTY: &str = r#"module @smoke_test {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
"#;

/// Builds the same module in generic form, through the real emitter.
///
/// The `func.func` header, the return, and the `module` wrapper are assembled
/// here rather than inside [`MlirEmitter`] — ARTX02 §1 keeps the emitter to
/// body lines only. Wave A2's `FuncBuilder` takes this job over properly; this
/// is the minimum needed to have something to compile.
pub fn add_scalar_module() -> String {
    let scalar = Shape::scalar(DType::F32);
    let ty = scalar.mlir_type();

    let mut e = MlirEmitter::new();
    // The two function parameters occupy the first SSA names, so the emitter's
    // counter has to start past them or the body would collide with the header.
    let lhs = e.fresh();
    let rhs = e.fresh();
    let sum = emit_add(&mut e, lhs, rhs, &scalar);
    e.line(format!(r#""func.return"({sum}) : ({ty}) -> ()"#));

    let mut module = String::with_capacity(e.body().len() + 256);
    module.push_str("module @smoke_test {\n");
    module.push_str(&format!(
        "  func.func @main(%v0: {ty}, %v1: {ty}) -> {ty} {{\n"
    ));
    module.push_str(&e.into_body());
    module.push_str("  }\n");
    module.push_str("}\n");
    module
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_module_declares_both_parameters_and_returns_the_sum() {
        let m = add_scalar_module();
        assert!(m.contains("module @smoke_test"), "{m}");
        assert!(
            m.contains("func.func @main(%v0: tensor<f32>, %v1: tensor<f32>) -> tensor<f32>"),
            "{m}"
        );
        assert!(
            m.contains(r#"%v2 = "stablehlo.add"(%v0, %v1) : (tensor<f32>, tensor<f32>) -> tensor<f32>"#),
            "{m}"
        );
        assert!(
            m.contains(r#""func.return"(%v2) : (tensor<f32>) -> ()"#),
            "{m}"
        );
    }

    /// Braces are balanced and nothing was left half-emitted. Cheap structural
    /// check — the real verdict is PJRT's parser, which needs a plugin.
    #[test]
    fn generic_module_has_balanced_braces() {
        let m = add_scalar_module();
        let opens = m.matches('{').count();
        let closes = m.matches('}').count();
        assert_eq!(opens, closes, "unbalanced braces in:\n{m}");
    }
}
