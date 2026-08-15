/*
Wave 1 audit findings for GWEN-223 (Candle 0.9.2):

- candle_nn::optim::AdamW keeps vars, step_t, first_moment, and second_moment
  private. The public surface exposes Optimizer::step plus params() and
  set_params(), but no m1/m2 accessor, so GWEN-223 needs a parallel
  MomentStore instead of reading Candle's internal optimizer state.
- candle_core::safetensors::save accepts HashMap<K, Tensor> where K: AsRef<str>
  + Ord + Display. Candle 0.9.2 has no DType::U64 or WithDType for u64; its
  supported integer tensor dtypes include U8, U32, I16, I32, and I64. Per the
  task fallback, step_t must therefore be stored without U64, with the cast
  documented at the save/load call site when implemented.
- Tensor::id() is public and returns TensorId. TensorId is Clone + Copy +
  PartialEq + Eq + Hash and is created from a process-local AtomicUsize
  monotonic counter, making it suitable for VarMap key resolution by scanning
  Vars and comparing each Var's Tensor::id() to the target Var.
*/
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor, Var};

pub(crate) type MomentStore = HashMap<String, (Tensor, Tensor)>;

pub(crate) fn varmap_key_to_adamw_prefix(varmap_key: &str) -> Option<String> {
    let parts: Vec<&str> = varmap_key.split('.').collect();
    let layer = parse_varmap_layer(parts.first().copied()?)?;

    match parts.as_slice() {
        [_, side] if is_lora_side(side) => Some(format!("layer_{layer}.{side}")),
        [_, proj, side] if is_projection_key(proj) && is_lora_side(side) => {
            Some(format!("layer_{layer}.{proj}.{side}"))
        }
        _ => None,
    }
}

pub(crate) fn adamw_prefix_to_varmap_key(prefix: &str) -> Option<String> {
    let parts: Vec<&str> = prefix.split('.').collect();
    let layer = parse_adamw_layer(parts.first().copied()?)?;

    match parts.as_slice() {
        [_, side] if is_lora_side(side) => Some(format!("l{layer}.{side}")),
        [_, proj, side] if is_projection_key(proj) && is_lora_side(side) => {
            Some(format!("l{layer}.{proj}.{side}"))
        }
        _ => None,
    }
}

pub(crate) fn varmap_key_for(var: &Var, data: &HashMap<String, Var>) -> Option<String> {
    let target_id = var.as_tensor().id();
    data.iter()
        .find(|(_, candidate)| candidate.as_tensor().id() == target_id)
        .map(|(key, _)| key.clone())
}

pub(crate) fn adamw_state_path(weight_ckpt_path: &Path) -> PathBuf {
    let stem = weight_ckpt_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("checkpoint");
    let filename = format!("{stem}_adamw.safetensors");

    match weight_ckpt_path.parent() {
        Some(parent) => parent.join(filename),
        None => PathBuf::from(filename),
    }
}

pub(crate) fn save_adamw_state(
    store: &MomentStore,
    step_t: usize,
    output_path: &Path,
    step: usize,
) -> Result<()> {
    std::fs::create_dir_all(output_path)?;
    let weight_path = output_path.join(format!("checkpoint_{step:06}.safetensors"));
    let path = adamw_state_path(&weight_path);

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (varmap_key, (m1, m2)) in store {
        let Some(prefix) = varmap_key_to_adamw_prefix(varmap_key) else {
            continue;
        };
        tensors.insert(format!("{prefix}.m1"), m1.clone());
        tensors.insert(format!("{prefix}.m2"), m2.clone());
    }

    // Candle 0.9.2 has no DType::U64. I64 is the nearest exact integer
    // safetensors dtype it supports, so save step_t as [i64] and validate the
    // narrowing cast instead of falling back to lossy F32.
    let step_i64 =
        i64::try_from(step_t).map_err(|_| anyhow!("step_t {step_t} exceeds i64::MAX"))?;
    tensors.insert(
        "step".to_string(),
        Tensor::from_vec(vec![step_i64], (1,), &Device::Cpu)?,
    );

    candle_core::safetensors::save(&tensors, &path)?;
    eprintln!("[checkpoint] AdamW state saved -> {}", path.display());
    Ok(())
}

pub(crate) fn load_adamw_state(weight_ckpt_path: &Path) -> Result<Option<(MomentStore, usize)>> {
    let path = adamw_state_path(weight_ckpt_path);
    if !path.exists() {
        return Ok(None);
    }

    let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
        .with_context(|| format!("failed to read AdamW state '{}'", path.display()))?;

    let step_values = tensors
        .get("step")
        .ok_or_else(|| anyhow!("AdamW state file '{}' missing 'step' key", path.display()))?
        .to_vec1::<i64>()
        .with_context(|| {
            format!(
                "AdamW state file '{}' has non-I64 or non-1D 'step' tensor",
                path.display()
            )
        })?;
    let step_i64 = *step_values.first().ok_or_else(|| {
        anyhow!(
            "AdamW state file '{}' has empty 'step' tensor",
            path.display()
        )
    })?;
    if step_i64 < 0 {
        return Err(anyhow!(
            "AdamW state file '{}' has negative step {step_i64}",
            path.display()
        ));
    }
    let step_t = usize::try_from(step_i64)
        .map_err(|_| anyhow!("AdamW state step {step_i64} does not fit usize"))?;

    let mut first_moments: HashMap<String, Tensor> = HashMap::new();
    let mut second_moments: HashMap<String, Tensor> = HashMap::new();
    for (key, tensor) in tensors {
        if key == "step" {
            continue;
        }
        if let Some(prefix) = key.strip_suffix(".m1") {
            if let Some(varmap_key) = adamw_prefix_to_varmap_key(prefix) {
                first_moments.insert(varmap_key, tensor);
            }
        } else if let Some(prefix) = key.strip_suffix(".m2") {
            if let Some(varmap_key) = adamw_prefix_to_varmap_key(prefix) {
                second_moments.insert(varmap_key, tensor);
            }
        }
    }

    let mut store = MomentStore::new();
    for (key, m1) in first_moments {
        if let Some(m2) = second_moments.remove(&key) {
            store.insert(key, (m1, m2));
        }
    }

    Ok(Some((store, step_t)))
}

fn parse_varmap_layer(part: &str) -> Option<&str> {
    let layer = part.strip_prefix('l')?;
    if is_layer_index(layer) {
        Some(layer)
    } else {
        None
    }
}

fn parse_adamw_layer(part: &str) -> Option<&str> {
    let layer = part.strip_prefix("layer_")?;
    if is_layer_index(layer) {
        Some(layer)
    } else {
        None
    }
}

fn is_layer_index(value: &str) -> bool {
    !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit())
}

fn is_projection_key(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn is_lora_side(value: &str) -> bool {
    matches!(value, "lora_a" | "lora_b")
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::{DType, Device};
    use candle_nn::{Init, VarMap};
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_varmap_key_to_adamw_prefix() {
        for (input, expected) in [
            ("l0.attn_q.lora_a", Some("layer_0.attn_q.lora_a")),
            ("l12.ffn_down.lora_b", Some("layer_12.ffn_down.lora_b")),
            ("l3.lora_a", Some("layer_3.lora_a")),
            ("layer_3.lora_a", None),
            ("l3.attn_q.lora_c", None),
            ("l.attn_q.lora_a", None),
            ("l3..lora_a", None),
            ("l3.attn-q.lora_a", None),
        ] {
            assert_eq!(
                varmap_key_to_adamw_prefix(input),
                expected.map(str::to_string),
                "input={input}"
            );
        }
    }

    #[test]
    fn test_adamw_prefix_to_varmap_key() {
        for (input, expected) in [
            ("layer_0.attn_q.lora_a", Some("l0.attn_q.lora_a")),
            ("layer_12.ffn_down.lora_b", Some("l12.ffn_down.lora_b")),
            ("layer_3.lora_a", Some("l3.lora_a")),
            ("l3.lora_a", None),
            ("layer_3.attn_q.lora_c", None),
            ("layer_.attn_q.lora_a", None),
            ("layer_3..lora_a", None),
            ("layer_3.attn-q.lora_a", None),
        ] {
            assert_eq!(
                adamw_prefix_to_varmap_key(input),
                expected.map(str::to_string),
                "input={input}"
            );
        }
    }

    #[test]
    fn test_varmap_key_for_resolves() {
        let varmap = VarMap::new();
        let key = "l0.attn_q.lora_a";
        let tensor = varmap
            .get((2, 3), key, Init::Const(0.0), DType::F32, &Device::Cpu)
            .expect("insert var");
        let var = Var::from_tensor(&tensor).expect("recover var");
        let data = varmap.data().lock().unwrap();

        assert_eq!(varmap_key_for(&var, &data), Some(key.to_string()));
    }

    #[test]
    fn prop_adamw_key_round_trip() {
        fn prop(layer: u8, proj_idx: u8, side_idx: bool, fallback: bool) -> bool {
            let projections = [
                "attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down",
            ];
            let side = if side_idx { "lora_a" } else { "lora_b" };
            let key = if fallback {
                format!("l{layer}.{side}")
            } else {
                let proj = projections[proj_idx as usize % projections.len()];
                format!("l{layer}.{proj}.{side}")
            };

            let Some(prefix) = varmap_key_to_adamw_prefix(&key) else {
                return false;
            };
            adamw_prefix_to_varmap_key(&prefix) == Some(key)
        }

        quickcheck::quickcheck(prop as fn(u8, u8, bool, bool) -> bool);
    }

    fn make_store() -> MomentStore {
        let mut store = MomentStore::new();
        store.insert(
            "l0.attn_q.lora_a".to_string(),
            (
                Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![0.01f32, 0.04, 0.09, 0.16], (2, 2), &Device::Cpu).unwrap(),
            ),
        );
        store.insert(
            "l1.lora_b".to_string(),
            (
                Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![1.0f32, 4.0], (1, 2), &Device::Cpu).unwrap(),
            ),
        );
        store
    }

    #[test]
    fn test_adamw_state_path_derivation() {
        let path = adamw_state_path(Path::new("out/checkpoint_000500.safetensors"));
        assert_eq!(
            path,
            PathBuf::from("out").join("checkpoint_000500_adamw.safetensors")
        );

        let path = adamw_state_path(Path::new("out/manual_checkpoint"));
        assert_eq!(
            path,
            PathBuf::from("out").join("manual_checkpoint_adamw.safetensors")
        );
    }

    #[test]
    fn test_save_adamw_state_creates_file() {
        let dir = TempDir::new().unwrap();
        save_adamw_state(&make_store(), 42, dir.path(), 500).expect("save adamw state");

        assert!(dir
            .path()
            .join("checkpoint_000500_adamw.safetensors")
            .exists());
    }

    #[test]
    fn test_save_adamw_state_filename_pattern() {
        let dir = TempDir::new().unwrap();
        save_adamw_state(&make_store(), 42, dir.path(), 1500).expect("save adamw state");

        let path = dir.path().join("checkpoint_001500_adamw.safetensors");
        assert!(path.exists());
    }

    #[test]
    fn test_save_adamw_state_contains_step_key() {
        let dir = TempDir::new().unwrap();
        save_adamw_state(&make_store(), 42, dir.path(), 500).expect("save adamw state");

        let tensors = candle_core::safetensors::load(
            dir.path().join("checkpoint_000500_adamw.safetensors"),
            &Device::Cpu,
        )
        .expect("load adamw state");
        assert!(tensors.contains_key("step"));
        assert_eq!(
            tensors["step"].to_vec1::<i64>().expect("step tensor"),
            vec![42]
        );
    }

    #[test]
    fn test_save_adamw_state_key_count() {
        let dir = TempDir::new().unwrap();
        let store = make_store();
        save_adamw_state(&store, 42, dir.path(), 500).expect("save adamw state");

        let tensors = candle_core::safetensors::load(
            dir.path().join("checkpoint_000500_adamw.safetensors"),
            &Device::Cpu,
        )
        .expect("load adamw state");
        assert_eq!(tensors.len(), 2 * store.len() + 1);
        for key in [
            "layer_0.attn_q.lora_a.m1",
            "layer_0.attn_q.lora_a.m2",
            "layer_1.lora_b.m1",
            "layer_1.lora_b.m2",
        ] {
            assert!(tensors.contains_key(key), "missing key {key}");
        }
    }

    #[test]
    fn test_save_adamw_state_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = MomentStore::new();
        save_adamw_state(&store, 7, dir.path(), 500).expect("save empty adamw state");

        let tensors = candle_core::safetensors::load(
            dir.path().join("checkpoint_000500_adamw.safetensors"),
            &Device::Cpu,
        )
        .expect("load adamw state");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["step"].to_vec1::<i64>().unwrap(), vec![7]);
    }

    fn tensor_values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn test_load_adamw_state_missing_file() {
        let dir = TempDir::new().unwrap();
        let weight_path = dir.path().join("checkpoint_000500.safetensors");

        let loaded = load_adamw_state(&weight_path).expect("load missing AdamW state");

        assert!(loaded.is_none());
    }

    #[test]
    fn test_load_adamw_state_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = make_store();
        let weight_path = dir.path().join("checkpoint_000500.safetensors");
        save_adamw_state(&store, 42, dir.path(), 500).expect("save adamw state");

        let (loaded, _) = load_adamw_state(&weight_path)
            .expect("load adamw state")
            .expect("state present");

        assert_eq!(loaded.len(), store.len());
        for (key, (expected_m1, expected_m2)) in &store {
            let (actual_m1, actual_m2) = loaded.get(key).expect("loaded moment pair");
            assert_eq!(actual_m1.dims(), expected_m1.dims());
            assert_eq!(actual_m2.dims(), expected_m2.dims());
            assert_eq!(tensor_values(actual_m1), tensor_values(expected_m1));
            assert_eq!(tensor_values(actual_m2), tensor_values(expected_m2));
        }
    }

    #[test]
    fn test_load_adamw_state_step_roundtrip() {
        let dir = TempDir::new().unwrap();
        let weight_path = dir.path().join("checkpoint_001000.safetensors");
        save_adamw_state(&make_store(), 1234, dir.path(), 1000).expect("save adamw state");

        let (_, step_t) = load_adamw_state(&weight_path)
            .expect("load adamw state")
            .expect("state present");

        assert_eq!(step_t, 1234);
    }

    #[test]
    fn test_load_adamw_state_corrupt_file() {
        let dir = TempDir::new().unwrap();
        let weight_path = dir.path().join("checkpoint_000500.safetensors");
        let adamw_path = adamw_state_path(&weight_path);
        std::fs::write(&adamw_path, b"not a safetensors file").expect("write corrupt state");

        let error = load_adamw_state(&weight_path)
            .expect_err("corrupt AdamW state should fail")
            .to_string();

        assert!(error.contains("failed to read AdamW state"), "{error}");
    }

    #[test]
    fn test_load_adamw_state_missing_step_key() {
        let dir = TempDir::new().unwrap();
        let weight_path = dir.path().join("checkpoint_000500.safetensors");
        let adamw_path = adamw_state_path(&weight_path);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "layer_0.attn_q.lora_a.m1".to_string(),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &adamw_path).expect("save missing-step state");

        let error = load_adamw_state(&weight_path)
            .expect_err("state without step should fail")
            .to_string();

        assert!(error.contains("missing 'step' key"), "{error}");
    }
}
