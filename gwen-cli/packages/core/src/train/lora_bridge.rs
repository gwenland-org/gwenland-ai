/// LoRA adapter bridge between candle training and SafeTensors export.
///
/// @INFO This module is the export side of GWEN-213. It converts trained
/// LoRA weight pairs (lora_a, lora_b) held in a candle VarMap into a
/// SafeTensors file consumable by lora_merger for GGUF merging.
use std::collections::HashMap;
use std::io::{BufWriter, Write as IoWrite};
use std::path::Path;

use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use serde_json::{json, Value};

use crate::error::GwenError;

// ── LoraAdapter ───────────────────────────────────────────────────────────────

/// A single LoRA adapter pair for one model layer.
///
/// Holds the low-rank matrices A and B that together approximate a weight delta
/// as Δ = (alpha/rank) × B @ A.
pub struct LoraAdapter {
    /// Canonical layer name, e.g. `lora_layer_0_q_proj`.
    pub layer_name: String,
    /// Shape (rank, d_in). Projected input matrix.
    pub lora_a: Tensor,
    /// Shape (d_out, rank). Projected output matrix.
    pub lora_b: Tensor,
    /// Inner dimension of the low-rank approximation.
    pub rank: usize,
    /// Scaling factor; effective scale = alpha / rank.
    /// @EDITABLE Tune per-adapter to control contribution magnitude.
    pub alpha: f32,
}

impl LoraAdapter {
    /// Compute the full weight delta Δ = (alpha/rank) × lora_b @ lora_a.
    ///
    /// @INFO The result has shape (d_out, d_in), matching the base weight
    /// tensor it will be added to during the merge step.
    /// @DANGER Panics if matmul fails due to shape incompatibility; call
    /// validate_shapes() first to surface errors cleanly.
    pub fn compute_delta(&self) -> Result<Tensor> {
        let scale = (self.alpha as f64) / (self.rank as f64);
        let delta = self.lora_b.matmul(&self.lora_a)?;
        delta.affine(scale, 0.0)
    }

    /// Validate that lora_a is (rank, d_in), lora_b is (d_out, rank),
    /// and both tensors reside on the same device.
    ///
    /// @INFO Must be called before compute_delta() or export to ensure
    /// no silent shape mismatches propagate into the merged model.
    pub fn validate_shapes(&self) -> std::result::Result<(), GwenError> {
        let a_shape = self.lora_a.dims().to_vec();
        let b_shape = self.lora_b.dims().to_vec();

        if a_shape.len() != 2 || a_shape[0] != self.rank {
            return Err(GwenError::InvalidLoraShape {
                expected: vec![self.rank, a_shape.get(1).copied().unwrap_or(0)],
                actual: a_shape,
            });
        }

        if b_shape.len() != 2 || b_shape[1] != self.rank {
            return Err(GwenError::InvalidLoraShape {
                expected: vec![b_shape.get(0).copied().unwrap_or(0), self.rank],
                actual: b_shape,
            });
        }

        if self.lora_a.device().location() != self.lora_b.device().location() {
            return Err(GwenError::CandleError(format!(
                "lora_a and lora_b for layer '{}' must be on the same device",
                self.layer_name
            )));
        }

        Ok(())
    }
}

// ── LoraConfig / LoraExporter ─────────────────────────────────────────────────

/// Configuration shared across all adapters produced by one training run.
/// @EDITABLE Adjust rank and alpha to match the training hyperparameters.
pub struct LoraConfig {
    /// LoRA inner rank. Must match the rank used during training.
    pub rank: usize,
    /// LoRA scaling factor. Must match the alpha used during training.
    pub alpha: f32,
    /// Target module suffixes that were adapted (e.g. ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    /// Sensible defaults matching the native training loop's LoRA hyperparameters.
    ///
    /// rank=8 and alpha=16.0 are the values used by `NewTrainConfig::default()`.
    /// @EDITABLE Override via `LoraConfig { rank, alpha, target_modules }` when
    /// re-exporting adapters trained with non-default hyperparameters.
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

/// Extracts trained LoRA adapters from a VarMap and serializes them to disk.
///
/// @INFO The VarMap is expected to contain keys in the candle naming convention:
/// `lora_{a|b}_layer_{N}_{proj_type}` (e.g. `lora_a_layer_0_q_proj`).
pub struct LoraExporter {
    pub config: LoraConfig,
}

impl LoraExporter {
    /// Construct a new exporter with the given training configuration.
    pub fn new(config: LoraConfig) -> Self {
        Self { config }
    }

    /// Extract all LoRA adapter pairs from the VarMap.
    ///
    /// Returns `MissingLoraPair` if any lora_a key has no matching lora_b.
    /// Keys not matching the `lora_{a|b}_layer_{N}_{proj}` pattern are silently
    /// skipped (they belong to base weights or other state).
    pub fn extract_adapters(
        &self,
        varmap: &VarMap,
    ) -> std::result::Result<Vec<LoraAdapter>, GwenError> {
        let data = varmap.data().lock().unwrap();

        let mut lora_a_map: HashMap<(usize, String), Tensor> = HashMap::new();
        let mut lora_b_map: HashMap<(usize, String), Tensor> = HashMap::new();

        for (name, var) in data.iter() {
            if let Some((side, layer_idx, proj)) = parse_lora_key(name) {
                let tensor = var.as_tensor().clone();
                match side {
                    Side::A => lora_a_map.insert((layer_idx, proj), tensor),
                    Side::B => lora_b_map.insert((layer_idx, proj), tensor),
                };
            }
        }

        let mut keys: Vec<(usize, String)> = lora_a_map.keys().cloned().collect();
        keys.sort_by_key(|(idx, proj)| (*idx, proj.clone()));

        let mut adapters: Vec<LoraAdapter> = Vec::new();
        for (layer_idx, proj) in keys {
            let lora_a = lora_a_map[&(layer_idx, proj.clone())].clone();
            let lora_b = lora_b_map
                .remove(&(layer_idx, proj.clone()))
                .ok_or(GwenError::MissingLoraPair { layer_idx })?;

            adapters.push(LoraAdapter {
                layer_name: format!("lora_layer_{}_{}_proj", layer_idx, proj),
                lora_a,
                lora_b,
                rank: self.config.rank,
                alpha: self.config.alpha,
            });
        }

        Ok(adapters)
    }

    /// Serialize all LoRA adapters from a VarMap to a SafeTensors file.
    ///
    /// Writes a spec-compliant SafeTensors file:
    ///   - 8-byte little-endian u64: header JSON length
    ///   - header JSON: map of tensor name → {dtype, shape, data_offsets}
    ///   - contiguous tensor data blobs (F32, little-endian)
    ///
    /// @INFO Tensor keys follow the pattern `{layer_name}.lora_a` /
    /// `{layer_name}.lora_b` to make each adapter's pair easy to locate.
    /// @DANGER Returns Err if any tensor is not on CPU; GPU tensors must be
    /// moved to CPU before export.
    /// @TODO Add an option to export in BF16 to halve file size.
    pub fn export_safetensors(
        &self,
        varmap: &VarMap,
        output_path: &Path,
    ) -> std::result::Result<(), GwenError> {
        let adapters = self.extract_adapters(varmap)?;

        // Flatten each lora_a / lora_b to a Vec<f32> in C-contiguous order.
        // We collect (key, shape, data) triples before writing anything.
        let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        for adapter in &adapters {
            for (suffix, tensor) in [("lora_a", &adapter.lora_a), ("lora_b", &adapter.lora_b)] {
                if !matches!(
                    tensor.device().location(),
                    candle_core::DeviceLocation::Cpu
                ) {
                    return Err(GwenError::CandleError(format!(
                        "tensor '{}.{}' must be on CPU for SafeTensors export",
                        adapter.layer_name, suffix
                    )));
                }

                let shape = tensor.dims().to_vec();
                // flatten_all() gives a 1-D view without copying unless needed.
                let flat = tensor
                    .flatten_all()
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;
                let data = flat
                    .to_vec1::<f32>()
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;

                entries.push((format!("{}.{}", adapter.layer_name, suffix), shape, data));
            }
        }

        // Compute data offsets (byte positions within the data blob).
        let mut offset: u64 = 0;
        let mut header_map: serde_json::Map<String, Value> = serde_json::Map::new();
        // Pre-compute all offsets in order so the header is complete before writing.
        let offsets: Vec<(u64, u64)> = entries
            .iter()
            .map(|(_, _, data)| {
                let start = offset;
                let end = offset + (data.len() as u64) * 4; // f32 = 4 bytes
                offset = end;
                (start, end)
            })
            .collect();

        for ((key, shape, _), (start, end)) in entries.iter().zip(offsets.iter()) {
            header_map.insert(
                key.clone(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json =
            serde_json::to_string(&Value::Object(header_map)).map_err(|e| {
                GwenError::CandleError(format!("SafeTensors header serialization failed: {e}"))
            })?;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // Write file: 8-byte LE header length + header + data blobs.
        let file = std::fs::File::create(output_path).map_err(|e| {
            GwenError::CandleError(format!(
                "failed to create {}: {e}",
                output_path.display()
            ))
        })?;
        let mut writer = BufWriter::new(file);

        writer
            .write_all(&header_len.to_le_bytes())
            .map_err(|e| GwenError::CandleError(e.to_string()))?;
        writer
            .write_all(header_bytes)
            .map_err(|e| GwenError::CandleError(e.to_string()))?;

        for (_, _, data) in &entries {
            for &v in data {
                writer
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;
            }
        }
        writer
            .flush()
            .map_err(|e| GwenError::CandleError(e.to_string()))?;

        // Validate: re-open the file and parse the header to confirm readability.
        validate_safetensors_header(output_path)?;

        Ok(())
    }
}

/// Re-open a written SafeTensors file and parse its header.
///
/// @INFO This is a lightweight sanity check — it confirms the 8-byte length
/// prefix is consistent with the actual header bytes, catching truncation or
/// write failures that BufWriter::flush() alone may not surface.
fn validate_safetensors_header(path: &Path) -> std::result::Result<(), GwenError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| {
        GwenError::CandleError(format!("validation open failed for {}: {e}", path.display()))
    })?;

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| {
        GwenError::CandleError(format!("validation read header length failed: {e}"))
    })?;

    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        GwenError::CandleError(format!("validation read header body failed: {e}"))
    })?;

    serde_json::from_slice::<Value>(&header_bytes).map_err(|e| {
        GwenError::CandleError(format!("validation header JSON parse failed: {e}"))
    })?;

    Ok(())
}

// ── internal helpers ──────────────────────────────────────────────────────────

enum Side {
    A,
    B,
}

/// Parse a VarMap key of the form `lora_{a|b}_layer_{N}_{proj_type}`.
///
/// Returns None for any key that does not match the pattern, so callers
/// can silently skip non-LoRA variables (base weights, optimizer state, etc.).
fn parse_lora_key(name: &str) -> Option<(Side, usize, String)> {
    let rest = name.strip_prefix("lora_")?;
    let (side_str, rest) = rest.split_once('_')?;
    let side = match side_str {
        "a" => Side::A,
        "b" => Side::B,
        _ => return None,
    };
    let rest = rest.strip_prefix("layer_")?;
    let (idx_str, proj_str) = rest.split_once('_')?;
    let layer_idx: usize = idx_str.parse().ok()?;
    // proj_str is e.g. "q_proj" or "gate_proj" — keep as-is
    Some((side, layer_idx, proj_str.to_string()))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Build a zero-filled LoraAdapter on CPU for shape/compute tests.
    fn make_adapter(rank: usize, d_in: usize, d_out: usize) -> LoraAdapter {
        let dev = &Device::Cpu;
        LoraAdapter {
            layer_name: format!("test_rank{}", rank),
            lora_a: Tensor::zeros((rank, d_in), DType::F32, dev).unwrap(),
            lora_b: Tensor::zeros((d_out, rank), DType::F32, dev).unwrap(),
            rank,
            alpha: rank as f32,
        }
    }

    /// Insert a matched lora_a / lora_b pair into a VarMap for testing.
    fn insert_pair(
        varmap: &VarMap,
        layer: usize,
        proj: &str,
        rank: usize,
        d_in: usize,
        d_out: usize,
    ) {
        let dev = &Device::Cpu;
        let mut data = varmap.data().lock().unwrap();
        data.insert(
            format!("lora_a_layer_{layer}_{proj}_proj"),
            candle_core::Var::from_tensor(
                &Tensor::zeros((rank, d_in), DType::F32, dev).unwrap(),
            )
            .unwrap(),
        );
        data.insert(
            format!("lora_b_layer_{layer}_{proj}_proj"),
            candle_core::Var::from_tensor(
                &Tensor::zeros((d_out, rank), DType::F32, dev).unwrap(),
            )
            .unwrap(),
        );
    }

    // ── Task 1.2: shape consistency across standard ranks ─────────────────────

    /// Verify lora_a/lora_b dimension invariants and compute_delta output shape
    /// across all standard LoRA ranks used in practice.
    #[test]
    fn lora_shape_consistency_across_ranks() {
        let d_in = 64;
        let d_out = 128;
        for &rank in &[1, 4, 8, 16, 32, 64, 128] {
            let adapter = make_adapter(rank, d_in, d_out);

            assert_eq!(adapter.lora_a.dim(0).unwrap(), rank);
            assert_eq!(adapter.lora_b.dim(1).unwrap(), rank);

            adapter.validate_shapes().unwrap_or_else(|e| {
                panic!("validate_shapes failed for rank={}: {}", rank, e)
            });

            let delta = adapter.compute_delta().unwrap();
            assert_eq!(delta.dims(), &[d_out, d_in], "rank={}", rank);
        }
    }

    /// validate_shapes must reject lora_a whose first dim != rank.
    #[test]
    fn validate_shapes_rejects_wrong_lora_a_rank() {
        let dev = &Device::Cpu;
        let adapter = LoraAdapter {
            layer_name: "bad".to_string(),
            lora_a: Tensor::zeros((8, 64), DType::F32, dev).unwrap(),
            lora_b: Tensor::zeros((128, 4), DType::F32, dev).unwrap(),
            rank: 4,
            alpha: 4.0,
        };
        assert!(matches!(
            adapter.validate_shapes(),
            Err(GwenError::InvalidLoraShape { .. })
        ));
    }

    /// validate_shapes must reject lora_b whose second dim != rank.
    #[test]
    fn validate_shapes_rejects_wrong_lora_b_rank() {
        let dev = &Device::Cpu;
        let adapter = LoraAdapter {
            layer_name: "bad".to_string(),
            lora_a: Tensor::zeros((4, 64), DType::F32, dev).unwrap(),
            lora_b: Tensor::zeros((128, 8), DType::F32, dev).unwrap(),
            rank: 4,
            alpha: 4.0,
        };
        assert!(matches!(
            adapter.validate_shapes(),
            Err(GwenError::InvalidLoraShape { .. })
        ));
    }

    /// extract_adapters must surface MissingLoraPair when lora_b is absent.
    #[test]
    fn extract_adapters_returns_missing_pair_error() {
        let varmap = VarMap::new();
        let dev = &Device::Cpu;
        varmap.data().lock().unwrap().insert(
            "lora_a_layer_0_q_proj".to_string(),
            candle_core::Var::from_tensor(&Tensor::zeros((4, 64), DType::F32, dev).unwrap())
                .unwrap(),
        );

        let exporter = LoraExporter::new(LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string()],
        });

        assert!(matches!(
            exporter.extract_adapters(&varmap),
            Err(GwenError::MissingLoraPair { layer_idx: 0 })
        ));
    }

    /// extract_adapters correctly pairs lora_a and lora_b for a single layer.
    #[test]
    fn extract_adapters_pairs_correctly() {
        let varmap = VarMap::new();
        insert_pair(&varmap, 0, "q", 4, 64, 128);

        let exporter = LoraExporter::new(LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string()],
        });

        let adapters = exporter.extract_adapters(&varmap).unwrap();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].rank, 4);
        adapters[0].validate_shapes().unwrap();
    }

    // ── Task 1.5: compute_delta with known values ─────────────────────────────

    /// Verify compute_delta numerics with identity-like matrices.
    ///
    /// A = I₂, B = 2·I₂, alpha = 2.0, rank = 2
    /// Δ = (2/2) × B @ A = [[2,0],[0,2]]
    #[test]
    fn test_compute_delta_known_values() {
        let dev = &Device::Cpu;
        // lora_a = identity (rank=2, d_in=2)
        let lora_a = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (2, 2), dev).unwrap();
        // lora_b = 2 × identity (d_out=2, rank=2)
        let lora_b = Tensor::from_slice(&[2.0f32, 0.0, 0.0, 2.0], (2, 2), dev).unwrap();

        let adapter = LoraAdapter {
            layer_name: "test".to_string(),
            lora_a,
            lora_b,
            rank: 2,
            alpha: 2.0,
        };

        let delta = adapter.compute_delta().unwrap();
        assert_eq!(delta.dims(), &[2, 2]);

        let vals = delta.to_vec2::<f32>().unwrap();
        // Expected: [[2, 0], [0, 2]]
        assert!((vals[0][0] - 2.0).abs() < 1e-5, "vals[0][0]={}", vals[0][0]);
        assert!((vals[0][1] - 0.0).abs() < 1e-5, "vals[0][1]={}", vals[0][1]);
        assert!((vals[1][0] - 0.0).abs() < 1e-5, "vals[1][0]={}", vals[1][0]);
        assert!((vals[1][1] - 2.0).abs() < 1e-5, "vals[1][1]={}", vals[1][1]);
    }

    // ── Task 1.5: SafeTensors export roundtrip ────────────────────────────────

    /// Export two adapter pairs to a temp file and parse the header back.
    ///
    /// Verifies that data_offsets are internally consistent with tensor sizes.
    #[test]
    fn test_export_safetensors_roundtrip() {
        use std::io::Read;

        let varmap = VarMap::new();
        let rank = 4;
        let d_in = 8;
        let d_out = 16;
        insert_pair(&varmap, 0, "q", rank, d_in, d_out);
        insert_pair(&varmap, 0, "v", rank, d_in, d_out);

        let exporter = LoraExporter::new(LoraConfig {
            rank,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        exporter
            .export_safetensors(&varmap, tmp.path())
            .expect("export_safetensors failed");

        // Read back and parse header.
        let mut file = std::fs::File::open(tmp.path()).unwrap();
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf).unwrap();
        let header_len = u64::from_le_bytes(len_buf) as usize;

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes).unwrap();

        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).expect("header JSON must parse");

        // Expect 4 keys: q lora_a/b + v lora_a/b
        let obj = header.as_object().unwrap();
        assert_eq!(obj.len(), 4, "expected 4 tensor keys, got {}", obj.len());

        // Verify data_offsets are plausible for each tensor.
        // Each tensor (rank × d_in or d_out × rank) is rank*d_in or d_out*rank f32 values.
        for (key, val) in obj {
            let offsets = val["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap();
            let end = offsets[1].as_u64().unwrap();
            assert!(end > start, "key {key}: end <= start");
            // size in bytes must be divisible by 4 (f32)
            assert_eq!((end - start) % 4, 0, "key {key}: byte span not f32-aligned");
        }
    }

    /// export_safetensors must propagate MissingLoraPair from extract_adapters.
    #[test]
    fn test_export_missing_pair_propagates() {
        let varmap = VarMap::new();
        let dev = &Device::Cpu;
        varmap.data().lock().unwrap().insert(
            "lora_a_layer_0_q_proj".to_string(),
            candle_core::Var::from_tensor(&Tensor::zeros((4, 64), DType::F32, dev).unwrap())
                .unwrap(),
        );

        let exporter = LoraExporter::new(LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string()],
        });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = exporter.export_safetensors(&varmap, tmp.path());
        assert!(
            matches!(result, Err(GwenError::MissingLoraPair { .. })),
            "expected MissingLoraPair, got {:?}",
            result.err()
        );
    }
}
