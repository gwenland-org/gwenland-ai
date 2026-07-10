//! Disk cache for the staged [`HostModel`].
//!
//! Building the HostModel from a GGUF ("stage") repacks Q8_0 blocks into the
//! SoA layout and dequantizes k-quants — for a 7B model that is ~30s of
//! single-threaded host work. This module serializes the staged result next to
//! the GGUF (`<model>.glcache`) so every subsequent load skips the repack and
//! just reads the bytes back. The cache is keyed on the source file's size and
//! mtime, so editing/replacing the GGUF invalidates it automatically.
//!
//! Format (little-endian; the writer and reader run on the same machine class):
//! magic, source size+mtime, config, then each staged tensor length-prefixed.
//! Best-effort: any read problem is a cache miss (rebuild), any write problem is
//! logged and ignored.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use glcore::GlError;

use crate::model::{GpuModelConfig, HostLayer, HostMat, HostModel, HostWeight, RopeStyle};

const MAGIC: &[u8; 8] = b"GLCACHE2";

/// Path of the cache file for a GGUF at `gguf_path`.
fn cache_path(gguf_path: &str) -> PathBuf {
    let mut p = PathBuf::from(gguf_path);
    let name = p.file_name().map(|n| n.to_owned()).unwrap_or_default();
    p.set_file_name(format!("{}.glcache", name.to_string_lossy()));
    p
}

/// (size, mtime_secs) identity of the source GGUF, or `None` if unavailable.
fn src_identity(gguf_path: &str) -> Option<(u64, u64)> {
    let m = std::fs::metadata(gguf_path).ok()?;
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some((m.len(), mtime))
}

/// Load a staged model from cache, or build it with `build` and cache the
/// result. Caching never changes correctness — a miss just rebuilds.
pub fn load_host_cached(
    gguf_path: &str,
    build: impl FnOnce() -> Result<HostModel, GlError>,
) -> Result<HostModel, GlError> {
    let path = cache_path(gguf_path);
    let ident = src_identity(gguf_path);

    if let Some((size, mtime)) = ident {
        if let Some(model) = read_cache(&path, size, mtime) {
            eprintln!("[glcuda] stage: loaded from cache {}", path.display());
            return Ok(model);
        }
    }

    let model = build()?;

    if let Some((size, mtime)) = ident {
        match write_cache(&path, &model, size, mtime) {
            Ok(()) => eprintln!("[glcuda] stage: wrote cache {}", path.display()),
            Err(e) => eprintln!("[glcuda] stage: cache write skipped ({e})"),
        }
    }
    Ok(model)
}

// --- read side (best-effort: any error -> None) ---

fn read_cache(path: &Path, size: u64, mtime: u64) -> Option<HostModel> {
    let f = File::open(path).ok()?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    if rd_u64(&mut r).ok()? != size || rd_u64(&mut r).ok()? != mtime {
        return None; // source changed since the cache was written
    }
    read_model(&mut r).ok()
}

fn rd_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn rd_string<R: Read>(r: &mut R) -> io::Result<String> {
    let n = rd_u64(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad utf8"))
}

fn rd_bytes<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let n = rd_u64(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn rd_vecf32<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
    let n = rd_u64(r)? as usize;
    let mut v = vec![0f32; n];
    // SAFETY: f32 has no invalid bit patterns; reading raw LE bytes into the
    // buffer is valid on the little-endian hosts this cache targets.
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
    r.read_exact(bytes)?;
    Ok(v)
}

fn rd_opt_vecf32<R: Read>(r: &mut R) -> io::Result<Option<Vec<f32>>> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    if tag[0] == 0 {
        Ok(None)
    } else {
        Ok(Some(rd_vecf32(r)?))
    }
}

fn rd_weight<R: Read>(r: &mut R) -> io::Result<HostWeight> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    Ok(match tag[0] {
        0 => HostWeight::F32(rd_vecf32(r)?),
        1 => HostWeight::Q8_0(rd_bytes(r)?),
        2 => HostWeight::Q8_0Soa { qs: rd_bytes(r)?, scales: rd_bytes(r)? },
        3 => HostWeight::Q4_0(rd_bytes(r)?),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad weight tag")),
    })
}

fn rd_mat<R: Read>(r: &mut R) -> io::Result<HostMat> {
    let out_dim = rd_u64(r)? as usize;
    let in_dim = rd_u64(r)? as usize;
    Ok(HostMat { w: rd_weight(r)?, out_dim, in_dim })
}

fn read_model<R: Read>(r: &mut R) -> io::Result<HostModel> {
    let config = GpuModelConfig {
        arch: rd_string(r)?,
        dim: rd_u64(r)? as usize,
        n_layers: rd_u64(r)? as usize,
        n_heads: rd_u64(r)? as usize,
        n_kv_heads: rd_u64(r)? as usize,
        head_dim: rd_u64(r)? as usize,
        hidden_dim: rd_u64(r)? as usize,
        vocab_size: rd_u64(r)? as usize,
        max_seq: rd_u64(r)? as usize,
        rms_eps: rd_f32(r)?,
        rope_freq_base: rd_f32(r)?,
        rope_style: if rd_u64(r)? == 0 { RopeStyle::Neox } else { RopeStyle::Norm },
    };
    let token_embd = rd_weight(r)?;
    let n = rd_u64(r)? as usize;
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        layers.push(HostLayer {
            attn_norm: rd_vecf32(r)?,
            wq: rd_mat(r)?,
            wk: rd_mat(r)?,
            wv: rd_mat(r)?,
            wo: rd_mat(r)?,
            bq: rd_opt_vecf32(r)?,
            bk: rd_opt_vecf32(r)?,
            bv: rd_opt_vecf32(r)?,
            q_norm: rd_opt_vecf32(r)?,
            k_norm: rd_opt_vecf32(r)?,
            ffn_norm: rd_vecf32(r)?,
            w_gate_up: rd_mat(r)?,
            w_down: rd_mat(r)?,
        });
    }
    let output_norm = rd_vecf32(r)?;
    let output = rd_mat(r)?;
    Ok(HostModel { config, token_embd, layers, output_norm, output })
}

// --- write side ---

fn write_cache(path: &Path, model: &HostModel, size: u64, mtime: u64) -> io::Result<()> {
    // Write to a temp file then rename, so a crash mid-write cannot leave a
    // half-written cache that reads as valid.
    let tmp = path.with_extension("glcache.tmp");
    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        w.write_all(MAGIC)?;
        wr_u64(&mut w, size)?;
        wr_u64(&mut w, mtime)?;
        write_model(&mut w, model)?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}

fn wr_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn wr_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn wr_bytes<W: Write>(w: &mut W, b: &[u8]) -> io::Result<()> {
    wr_u64(w, b.len() as u64)?;
    w.write_all(b)
}

fn wr_vecf32<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    wr_u64(w, v.len() as u64)?;
    // SAFETY: reinterpret the f32 slice as its little-endian byte view.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
    w.write_all(bytes)
}

fn wr_opt_vecf32<W: Write>(w: &mut W, v: &Option<Vec<f32>>) -> io::Result<()> {
    match v {
        None => w.write_all(&[0u8]),
        Some(v) => {
            w.write_all(&[1u8])?;
            wr_vecf32(w, v)
        }
    }
}

fn wr_weight<W: Write>(w: &mut W, weight: &HostWeight) -> io::Result<()> {
    match weight {
        HostWeight::F32(v) => {
            w.write_all(&[0u8])?;
            wr_vecf32(w, v)
        }
        HostWeight::Q8_0(b) => {
            w.write_all(&[1u8])?;
            wr_bytes(w, b)
        }
        HostWeight::Q8_0Soa { qs, scales } => {
            w.write_all(&[2u8])?;
            wr_bytes(w, qs)?;
            wr_bytes(w, scales)
        }
        HostWeight::Q4_0(b) => {
            w.write_all(&[3u8])?;
            wr_bytes(w, b)
        }
    }
}

fn wr_mat<W: Write>(w: &mut W, m: &HostMat) -> io::Result<()> {
    wr_u64(w, m.out_dim as u64)?;
    wr_u64(w, m.in_dim as u64)?;
    wr_weight(w, &m.w)
}

fn write_model<W: Write>(w: &mut W, model: &HostModel) -> io::Result<()> {
    let c = &model.config;
    wr_u64(w, c.arch.len() as u64)?;
    w.write_all(c.arch.as_bytes())?;
    for v in [c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.hidden_dim, c.vocab_size, c.max_seq] {
        wr_u64(w, v as u64)?;
    }
    wr_f32(w, c.rms_eps)?;
    wr_f32(w, c.rope_freq_base)?;
    wr_u64(w, if c.rope_style == RopeStyle::Neox { 0 } else { 1 })?;

    wr_weight(w, &model.token_embd)?;
    wr_u64(w, model.layers.len() as u64)?;
    for l in &model.layers {
        wr_vecf32(w, &l.attn_norm)?;
        wr_mat(w, &l.wq)?;
        wr_mat(w, &l.wk)?;
        wr_mat(w, &l.wv)?;
        wr_mat(w, &l.wo)?;
        wr_opt_vecf32(w, &l.bq)?;
        wr_opt_vecf32(w, &l.bk)?;
        wr_opt_vecf32(w, &l.bv)?;
        wr_opt_vecf32(w, &l.q_norm)?;
        wr_opt_vecf32(w, &l.k_norm)?;
        wr_vecf32(w, &l.ffn_norm)?;
        wr_mat(w, &l.w_gate_up)?;
        wr_mat(w, &l.w_down)?;
    }
    wr_vecf32(w, &model.output_norm)?;
    wr_mat(w, &model.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(out_dim: usize, in_dim: usize, w: HostWeight) -> HostMat {
        HostMat { w, out_dim, in_dim }
    }

    fn tiny_model() -> HostModel {
        let cfg = GpuModelConfig {
            arch: "qwen2".into(),
            dim: 4,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 2,
            hidden_dim: 8,
            vocab_size: 3,
            max_seq: 16,
            rms_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_style: RopeStyle::Neox,
        };
        let layer = HostLayer {
            attn_norm: vec![1.0, 2.0, 3.0, 4.0],
            wq: mat(4, 4, HostWeight::Q8_0Soa { qs: vec![1, 2, 3, 4], scales: vec![9, 8] }),
            wk: mat(2, 4, HostWeight::F32(vec![0.5; 8])),
            wv: mat(2, 4, HostWeight::Q4_0(vec![7; 18])),
            wo: mat(4, 4, HostWeight::F32(vec![-1.0; 16])),
            bq: Some(vec![0.1, 0.2, 0.3, 0.4]),
            bk: None,
            bv: Some(vec![9.9]),
            q_norm: None,
            k_norm: Some(vec![1.5, 2.5]),
            ffn_norm: vec![5.0, 6.0, 7.0, 8.0],
            w_gate_up: mat(16, 4, HostWeight::Q8_0Soa { qs: vec![5; 64], scales: vec![1; 4] }),
            w_down: mat(4, 8, HostWeight::F32(vec![0.25; 32])),
        };
        HostModel {
            config: cfg,
            token_embd: HostWeight::F32(vec![0.0, 1.0, 2.0]),
            layers: vec![layer],
            output_norm: vec![1.0; 4],
            output: mat(3, 4, HostWeight::F32(vec![2.0; 12])),
        }
    }

    fn weight_eq(a: &HostWeight, b: &HostWeight) -> bool {
        match (a, b) {
            (HostWeight::F32(x), HostWeight::F32(y)) => x == y,
            (HostWeight::Q8_0(x), HostWeight::Q8_0(y)) => x == y,
            (HostWeight::Q4_0(x), HostWeight::Q4_0(y)) => x == y,
            (
                HostWeight::Q8_0Soa { qs: xq, scales: xs },
                HostWeight::Q8_0Soa { qs: yq, scales: ys },
            ) => xq == yq && xs == ys,
            _ => false,
        }
    }

    #[test]
    fn round_trips_byte_exact() {
        let m = tiny_model();
        let mut buf = Vec::new();
        write_model(&mut buf, &m).unwrap();
        let back = read_model(&mut &buf[..]).unwrap();

        assert_eq!(back.config.arch, m.config.arch);
        assert_eq!(back.config.dim, m.config.dim);
        assert_eq!(back.config.hidden_dim, m.config.hidden_dim);
        assert_eq!(back.config.rope_style, m.config.rope_style);
        assert_eq!(back.config.rms_eps, m.config.rms_eps);
        assert!(weight_eq(&back.token_embd, &m.token_embd));
        assert_eq!(back.layers.len(), 1);
        let (lb, lm) = (&back.layers[0], &m.layers[0]);
        assert_eq!(lb.attn_norm, lm.attn_norm);
        assert!(weight_eq(&lb.wq.w, &lm.wq.w));
        assert!(weight_eq(&lb.wv.w, &lm.wv.w)); // Q4_0
        assert_eq!(lb.wq.out_dim, lm.wq.out_dim);
        assert_eq!(lb.bq, lm.bq);
        assert_eq!(lb.bk, lm.bk); // None
        assert_eq!(lb.k_norm, lm.k_norm);
        assert!(weight_eq(&lb.w_gate_up.w, &lm.w_gate_up.w));
        assert_eq!(back.output_norm, m.output_norm);
        assert!(weight_eq(&back.output.w, &m.output.w));
    }

    #[test]
    fn wrong_source_identity_is_a_miss() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("glcuda_cache_test_{}.glcache", std::process::id()));
        write_cache(&path, &tiny_model(), 1234, 5678).unwrap();
        assert!(read_cache(&path, 1234, 5678).is_some(), "matching identity hits");
        assert!(read_cache(&path, 9999, 5678).is_none(), "changed size misses");
        assert!(read_cache(&path, 1234, 1).is_none(), "changed mtime misses");
        let _ = std::fs::remove_file(&path);
    }
}
