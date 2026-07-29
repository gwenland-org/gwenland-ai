//! `ExecutionPlan` — a compiled program plus the ABI needed to call it.

use crate::graph::Signature;
use crate::stablehlo::types::Shape;
use crate::GlError;

/// What a compiled artifact expects and produces.
///
/// Separate from the [`crate::pjrt::LoadedExecutable`] it describes, because
/// the shapes are host-side facts that can be checked *before* any device call
/// — and a mismatch caught here names the parameter, whereas the same mismatch
/// caught inside PJRT is an opaque compile-time error.
#[derive(Debug, Clone)]
pub struct PlanSignature {
    /// Every parameter in `func.func` declaration order — the order PJRT reads
    /// `argument_lists` in.
    pub params: Vec<(String, Shape)>,
    /// How many of the leading parameters are runtime inputs. The rest are
    /// weights, uploaded once and reused (ARTX01 §8.2).
    pub outputs: Vec<Shape>,
}

impl PlanSignature {
    pub fn from_traced(sig: &Signature) -> Self {
        PlanSignature {
            params: sig
                .param_order
                .iter()
                .map(|p| (p.name.clone(), p.shape.clone()))
                .collect(),
            outputs: sig.outputs.clone(),
        }
    }

    /// Checks that a list of `(name, shape)` matches this plan exactly.
    ///
    /// ⛔ **Order matters as much as shape.** PJRT takes a flat array of
    /// buffers; nothing in the C API carries a parameter name. Two weights of
    /// the same shape swapped past this check produce a model that runs, emits
    /// text, and is wrong — P4 again. So this compares names positionally, not
    /// as a set.
    ///
    /// Returns [`GlError::Engine`] rather than [`GlError::ShapeMismatch`] for
    /// per-parameter failures: `ShapeMismatch` carries two dimension lists and
    /// no name, and "expected [896, 4864], got [4864, 896]" without saying
    /// *which* of the seven such tensors is the wrong one fails
    /// `rust-skills/error-handling.md` rule 4.
    pub fn validate(&self, provided: &[(String, Shape)]) -> Result<(), GlError> {
        if provided.len() != self.params.len() {
            return Err(GlError::Engine(format!(
                "signature mismatch: plan expects {} parameters, got {}. \
                 Expected order: {:?}",
                self.params.len(),
                provided.len(),
                self.params.iter().map(|(n, _)| n).collect::<Vec<_>>()
            )));
        }
        for (i, ((want_name, want_shape), (got_name, got_shape))) in
            self.params.iter().zip(provided).enumerate()
        {
            if want_name != got_name {
                return Err(GlError::Engine(format!(
                    "signature mismatch at parameter {i}: plan expects {want_name:?}, \
                     got {got_name:?}. PJRT matches arguments by position, so a \
                     reordering here silently binds the wrong tensor"
                )));
            }
            if want_shape != got_shape {
                return Err(GlError::Engine(format!(
                    "signature mismatch for {want_name:?}: plan expects {}, got {}",
                    want_shape.mlir_type(),
                    got_shape.mlir_type()
                )));
            }
        }
        Ok(())
    }

    /// The parameter names, in call order.
    pub fn names(&self) -> Vec<&str> {
        self.params.iter().map(|(n, _)| n.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stablehlo::types::DType;

    fn shape(dims: [usize; 2]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    fn plan() -> PlanSignature {
        PlanSignature {
            params: vec![
                ("input_ids".into(), shape([1, 8])),
                ("w.gate".into(), shape([4, 16])),
                ("w.up".into(), shape([4, 16])),
            ],
            outputs: vec![shape([1, 8])],
        }
    }

    #[test]
    fn a_matching_signature_validates() {
        let p = plan();
        let provided: Vec<_> = p.params.clone();
        p.validate(&provided).expect("identical signature");
    }

    #[test]
    fn signature_validation_rejects_shape_mismatch() {
        let p = plan();
        let mut provided = p.params.clone();
        provided[1].1 = shape([4, 17]);
        let err = p.validate(&provided).expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("w.gate"), "must name the parameter: {msg}");
        assert!(msg.contains("tensor<4x16xf32>"), "{msg}");
        assert!(msg.contains("tensor<4x17xf32>"), "{msg}");
    }

    #[test]
    fn signature_validation_rejects_a_wrong_argument_count() {
        let p = plan();
        let provided = p.params[..2].to_vec();
        let err = p.validate(&provided).expect_err("must reject");
        assert!(err.to_string().contains("expects 3 parameters, got 2"), "{err}");
    }

    /// ⛔ `w.gate` and `w.up` have identical shapes. Swapping them passes every
    /// shape check and produces a model that runs and is wrong.
    #[test]
    fn signature_validation_catches_two_same_shaped_weights_swapped() {
        let p = plan();
        let mut provided = p.params.clone();
        provided.swap(1, 2);
        let err = p.validate(&provided).expect_err("must reject a reordering");
        let msg = err.to_string();
        assert!(msg.contains("parameter 1"), "{msg}");
        assert!(msg.contains("position"), "must explain why order matters: {msg}");
    }

    #[test]
    fn plan_signature_preserves_the_traced_parameter_order() {
        use crate::model::{trace_forward, Qwen2Config};
        let cfg = Qwen2Config::tiny();
        let built =
            crate::with_policy(crate::PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
                .expect("trace");
        let plan = PlanSignature::from_traced(&built.signature);

        assert_eq!(plan.names()[0], "input_ids", "the runtime input comes first");
        assert_eq!(plan.names()[1], "model.embed_tokens.weight");
        assert_eq!(plan.params.len(), built.signature.param_order.len());
        assert_eq!(plan.outputs[0].dims, vec![1, 4, cfg.vocab]);
    }
}
