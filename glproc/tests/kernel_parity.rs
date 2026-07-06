// LCG RNG from scratch
struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }
    fn fill(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.next_u32() as u8;
        }
    }
}
fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol || (g.is_nan() && w.is_nan()),
            "element {}: got {}, want {}",
            i, g, w
        );
    }
}

// Ensure the host CPU has AVX2 so we can test it
fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw")
}

#[test]
fn q4_0_avx2_matches_scalar() {
    if !has_avx2() {
        return;
    }
    let mut data = vec![0u8; 18 * 4]; // 4 blocks
    let mut rng = Lcg::new(42);
    rng.fill(&mut data[..]);
    
    // Fix scales so they are finite and reasonable
    for block in data.chunks_mut(18) {
        // e.g. 1.0 in f16 is 0x3c00
        block[0] = 0x00;
        block[1] = 0x3c;
    }
    
    let scalar = glproc::kernels::dequant::q4_0::scalar::run(&data);
    let avx2 = unsafe { glproc::kernels::dequant::q4_0::avx2::run(&data) };
    
    assert_close(&avx2, &scalar, 1e-5);
}

#[test]
fn q4_0_avx512_matches_scalar() {
    if !has_avx512() {
        return;
    }
    let mut data = vec![0u8; 18 * 4];
    let mut rng = Lcg::new(42);
    rng.fill(&mut data[..]);
    for block in data.chunks_mut(18) {
        block[0] = 0x00;
        block[1] = 0x3c;
    }
    let scalar = glproc::kernels::dequant::q4_0::scalar::run(&data);
    let avx512 = unsafe { glproc::kernels::dequant::q4_0::avx512::run(&data) };
    assert_close(&avx512, &scalar, 1e-5);
}

// ---------------------------------------------------------------------------
// M1.5 Bridge-ing parity tests
// Rule 3: scalar is ground truth; SIMD is tested against scalar, never the
// other way round. All three must pass before any kernel is considered done.
// ---------------------------------------------------------------------------

use glproc::kernels::bridge::bridge_matmul_q4k;
use glproc::kernels::dequant::q4_k;
use glproc::simd_strategy::SimdStrategy;

/// Build one 144-byte Q4_K super-block with unit scales (d=1.0, dmin=0.0,
/// all 8 sub-block scales = 1, mins = 0) so a weight equals its raw nibble.
fn q4k_block_identity(qs: &[u8; 128]) -> Vec<u8> {
    let mut block = Vec::with_capacity(144);
    block.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0 (f16)
    block.extend_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
    // 6-bit packed scales: bytes 0..4 = sub-block 0..3 scales (=1), bytes
    // 4..8 = sub-block 0..3 mins (=0), bytes 8..12 = sub-block 4..7 packed
    // (scale=1 in the low nibble, min=0 in the high nibble, high bits in
    // bytes 0..8 are zero).
    block.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    block.extend_from_slice(qs);
    block
}

/// Build a Q4_K block with arbitrary (but finite) scales and random content.
fn q4k_block_random(rng: &mut Lcg) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    rng.fill(&mut block);
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    block[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
    block
}

fn lcg_f32(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| rng.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0)
        .collect()
}

/// Test 1: dequant known values against hand-computed expectations, then
/// SIMD paths against the (now proven) scalar ground truth.
#[test]
fn dequant_q4k_known_values() {
    // qs byte i: low nibble = i % 16, high nibble = (i + 1) % 16.
    let mut qs = [0u8; 128];
    for (i, b) in qs.iter_mut().enumerate() {
        *b = ((i % 16) as u8) | ((((i + 1) % 16) as u8) << 4);
    }
    let block = q4k_block_identity(&qs);

    let mut scalar = [0f32; 256];
    q4_k::scalar::dequant_block(&block, &mut scalar);

    // GGML layout: chunk c covers weights c*64..c*64+64; the first 32 come
    // from low nibbles of qs[c*32..c*32+32], the next 32 from high nibbles.
    for c in 0..4 {
        for l in 0..32 {
            let lo = ((c * 32 + l) % 16) as f32;
            let hi = ((c * 32 + l + 1) % 16) as f32;
            assert!(
                (scalar[c * 64 + l] - lo).abs() < 1e-5,
                "chunk {c} low nibble {l}: got {}, want {lo}",
                scalar[c * 64 + l]
            );
            assert!(
                (scalar[c * 64 + 32 + l] - hi).abs() < 1e-5,
                "chunk {c} high nibble {l}: got {}, want {hi}",
                scalar[c * 64 + 32 + l]
            );
        }
    }

    if has_avx2() {
        let mut avx2 = [0f32; 256];
        // SAFETY: guarded by has_avx2().
        unsafe { q4_k::avx2::dequant_block(&block, &mut avx2) };
        assert_close(&avx2, &scalar, 1e-5);
    }
    if has_avx512() {
        let mut avx512 = [0f32; 256];
        // SAFETY: guarded by has_avx512().
        unsafe { q4_k::avx512::dequant_block(&block, &mut avx512) };
        assert_close(&avx512, &scalar, 1e-5);
    }
}

/// Test 2: dot product, scalar vs SIMD, including a length that is not a
/// multiple of the vector width (exercises the tail loop).
#[test]
fn dot_scalar_vs_simd_parity() {
    let mut rng = Lcg::new(7);
    for len in [256usize, 250, 16, 3] {
        let a = lcg_f32(&mut rng, len);
        let b = lcg_f32(&mut rng, len);
        let want = glproc::kernels::matmul::scalar::dot_f32(&a, &b);

        if has_avx2() {
            // SAFETY: guarded by has_avx2().
            let got = unsafe { glproc::kernels::matmul::avx2::dot_f32(&a, &b) };
            assert!(
                (got - want).abs() < 1e-4,
                "avx2 len {len}: got {got}, want {want}"
            );
        }
        if has_avx512() {
            // SAFETY: guarded by has_avx512().
            let got = unsafe { glproc::kernels::matmul::avx512::dot_f32(&a, &b) };
            assert!(
                (got - want).abs() < 1e-4,
                "avx512 len {len}: got {got}, want {want}"
            );
        }
    }
}

/// Test 3: Bridge-ing end-to-end — dequant → L1 buffer → dot — SIMD backends
/// vs the scalar backend over random Q4_K blocks. Relative tolerance because
/// FMA accumulation order differs between backends.
#[test]
fn bridge_matmul_q4k_parity() {
    let mut rng = Lcg::new(1234);
    let n_blocks = 4;
    let mut blocks = Vec::new();
    for _ in 0..n_blocks {
        blocks.extend_from_slice(&q4k_block_random(&mut rng));
    }
    let input = lcg_f32(&mut rng, n_blocks * 256);

    let want = bridge_matmul_q4k(&blocks, &input, n_blocks, SimdStrategy::Scalar);
    assert!(want.is_finite());
    let tol = 1e-4 * want.abs().max(1.0);

    if has_avx2() {
        let got = bridge_matmul_q4k(&blocks, &input, n_blocks, SimdStrategy::Avx2);
        assert!(
            (got - want).abs() <= tol,
            "avx2 bridge: got {got}, want {want}"
        );
    }
    if has_avx512() {
        let got = bridge_matmul_q4k(&blocks, &input, n_blocks, SimdStrategy::Avx512);
        assert!(
            (got - want).abs() <= tol,
            "avx512 bridge: got {got}, want {want}"
        );
    }

    // The bridge must also agree with the naive two-stage path:
    // full dequant to RAM, then a plain dot product.
    let dequant = q4_k::scalar::run(&blocks).unwrap();
    let naive = glproc::kernels::matmul::scalar::dot_f32(&dequant, &input);
    assert!(
        (naive - want).abs() <= tol,
        "bridge vs naive: got {want}, want {naive}"
    );
}

/// Q5_0 known values: with d=1.0 the weight is `(nibble | qh_bit<<4) - 16`.
#[test]
fn dequant_q5_0_known_values_and_simd_parity() {
    use glproc::kernels::dequant::q5_0;

    // Block: d=1.0, qh = alternating bits (weight i's 5th bit = i & 1),
    // qs byte i: low nibble = i, high nibble = 15 - i.
    let mut block = vec![0u8; 22];
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    block[2..6].copy_from_slice(&0xAAAA_AAAAu32.to_le_bytes()); // bits 1,3,5,...
    for i in 0..16u8 {
        block[6 + i as usize] = i | ((15 - i) << 4);
    }

    let mut got = [0f32; 32];
    q5_0::scalar::dequant_block(&block, &mut got);

    for i in 0..16usize {
        let bit_lo = (i & 1) as u32; // bit i of 0xAAAAAAAA
        let bit_hi = (i & 1) as u32; // bit i+16 has the same parity
        let want_lo = (i as u32 | (bit_lo << 4)) as f32 - 16.0;
        let want_hi = ((15 - i) as u32 | (bit_hi << 4)) as f32 - 16.0;
        assert!(
            (got[i] - want_lo).abs() < 1e-5,
            "weight {i}: got {}, want {want_lo}",
            got[i]
        );
        assert!(
            (got[i + 16] - want_hi).abs() < 1e-5,
            "weight {}: got {}, want {want_hi}",
            i + 16,
            got[i + 16]
        );
    }

    if has_avx2() {
        let mut rng = Lcg::new(99);
        for _ in 0..8 {
            let mut b = vec![0u8; 22];
            rng.fill(&mut b);
            b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // finite d
            let mut scalar = [0f32; 32];
            q5_0::scalar::dequant_block(&b, &mut scalar);
            let mut avx2 = [0f32; 32];
            // SAFETY: guarded by has_avx2().
            unsafe { q5_0::avx2::dequant_block(&b, &mut avx2) };
            assert_close(&avx2, &scalar, 1e-5);
        }
    }
}

/// Q8_0 block dequant must agree with the existing whole-tensor kernel.
#[test]
fn dequant_q8_0_block_matches_tensor_path() {
    use glproc::kernels::dequant::q8_0;

    let mut rng = Lcg::new(5);
    let mut data = vec![0u8; 34 * 3];
    rng.fill(&mut data);
    for block in data.chunks_mut(34) {
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // finite d
    }

    let want = q8_0::scalar::run(&data);
    for (bi, block) in data.chunks_exact(34).enumerate() {
        let mut got = [0f32; 32];
        q8_0::scalar::dequant_block(block, &mut got);
        assert_close(&got, &want[bi * 32..(bi + 1) * 32], 1e-6);

        if has_avx2() {
            let mut simd = [0f32; 32];
            // SAFETY: guarded by has_avx2().
            unsafe { q8_0::avx2::dequant_block(block, &mut simd) };
            assert_close(&simd, &got, 1e-6);
        }
    }
}

/// Q6_K spot checks against hand-derived values from the GGML layout.
#[test]
fn dequant_q6_k_known_values() {
    use glproc::kernels::dequant::q6_k;

    // d=1.0, all 16 sub-block scales = 1, qh = 0 (no high bits).
    let mut block = vec![0u8; 210];
    block[0] = 0x21; // ql[0]: low nibble 1, high nibble 2
    block[32] = 0x43; // ql[32]: low nibble 3, high nibble 4
    block[64] = 0x65; // ql[64] → first byte of half 1
    for s in &mut block[192..208] {
        *s = 1;
    }
    block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

    let mut got = [0f32; 256];
    q6_k::scalar::dequant_block(&block, &mut got);

    // Half 0, l=0: q1 = ql[0]&0xF = 1 → 1-32 = -31 at out[0];
    // q3 = ql[0]>>4 = 2 → 2-32 = -30 at out[64].
    assert_eq!(got[0], -31.0);
    assert_eq!(got[64], -30.0);
    // l=0 also reads ql[32]: q2 = 3-32 = -29 at out[32]; q4 = 4-32 = -28 at out[96].
    assert_eq!(got[32], -29.0);
    assert_eq!(got[96], -28.0);
    // Half 1 starts at out[128] and reads ql[64]: 5-32 = -27, 6-32 = -26.
    assert_eq!(got[128], -27.0);
    assert_eq!(got[192], -26.0);
    // Untouched quants are 0 → 0-32 = -32 everywhere else.
    assert_eq!(got[1], -32.0);
    assert_eq!(got[255], -32.0);

    if has_avx2() {
        let mut rng = Lcg::new(31);
        for _ in 0..4 {
            let mut b = vec![0u8; 210];
            rng.fill(&mut b);
            b[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // finite d
            let mut scalar = [0f32; 256];
            q6_k::scalar::dequant_block(&b, &mut scalar);
            let mut avx2 = [0f32; 256];
            // SAFETY: guarded by has_avx2().
            unsafe { q6_k::avx2::dequant_block(&b, &mut avx2) };
            assert_close(&avx2, &scalar, 1e-4);
        }
    }
}

/// The generalized bridge must agree with full-dequant + plain dot for
/// every supported format.
#[test]
fn bridge_row_dot_all_formats_match_naive() {
    use glproc::kernels::bridge::{bridge_row_dot, QuantFormat};
    use glproc::kernels::dequant::{q4_k, q5_0, q6_k, q8_0};

    let mut rng = Lcg::new(4242);
    let cases: [(QuantFormat, usize, usize); 4] = [
        (QuantFormat::Q4K, 256, 144),
        (QuantFormat::Q5_0, 32, 22),
        (QuantFormat::Q6K, 256, 210),
        (QuantFormat::Q8_0, 32, 34),
    ];

    for (fmt, bn, bb) in cases {
        let n_blocks = 512 / bn; // 512 weights per row for every format
        let mut row = vec![0u8; n_blocks * bb];
        rng.fill(&mut row);
        for block in row.chunks_mut(bb) {
            // Pin the f16 scale fields to finite values.
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
            match fmt {
                QuantFormat::Q4K => block[2..4].copy_from_slice(&0x3800u16.to_le_bytes()),
                QuantFormat::Q6K => block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()),
                _ => {}
            }
        }
        let x = lcg_f32(&mut rng, n_blocks * bn);

        let dequant = match fmt {
            QuantFormat::Q4K => q4_k::scalar::run(&row).unwrap(),
            QuantFormat::Q5_0 => q5_0::scalar::run(&row).unwrap(),
            QuantFormat::Q6K => q6_k::scalar::run(&row).unwrap(),
            QuantFormat::Q8_0 => q8_0::scalar::run(&row),
        };
        let naive = glproc::kernels::matmul::scalar::dot_f32(&dequant, &x);
        let tol = 1e-4 * naive.abs().max(1.0);

        let scalar = bridge_row_dot(fmt, &row, &x, SimdStrategy::Scalar);
        assert!(
            (scalar - naive).abs() <= tol,
            "{fmt:?} scalar bridge: got {scalar}, want {naive}"
        );
        if has_avx2() {
            let got = bridge_row_dot(fmt, &row, &x, SimdStrategy::Avx2);
            assert!(
                (got - naive).abs() <= tol,
                "{fmt:?} avx2 bridge: got {got}, want {naive}"
            );
        }
    }
}

/// Integer-domain fused dot vs the f32 bridge, per format.
///
/// The activation is built from whole numbers with one ±127 per 32-group,
/// so Q8 quantization is *exact* (scale = 1.0) and any disagreement is a
/// kernel bug, not quantization noise.
#[test]
fn qdot_matches_f32_bridge_exactly_quantizable() {
    use glproc::kernels::bridge::QuantFormat;
    use glproc::kernels::dequant::{q5_0, q6_k, q8_0};
    use glproc::kernels::qdot::{self, QuantizedActivation};

    let mut rng = Lcg::new(2026);
    let cases: [(QuantFormat, usize); 3] = [
        (QuantFormat::Q5_0, 22),
        (QuantFormat::Q6K, 210),
        (QuantFormat::Q8_0, 34),
    ];

    for (fmt, bb) in cases {
        let bn = fmt.block_numel();
        let n_blocks = 512 / bn;
        let mut row = vec![0u8; n_blocks * bb];
        rng.fill(&mut row);
        for block in row.chunks_mut(bb) {
            match fmt {
                QuantFormat::Q6K => block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()),
                _ => block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()),
            }
        }

        // Exactly quantizable activation: integers in [-127, 127], one 127
        // per 32-group so the group scale is exactly 1.0.
        let mut x: Vec<f32> = (0..512)
            .map(|_| (rng.next_u32() % 255) as f32 - 127.0)
            .collect();
        for g in x.chunks_mut(32) {
            g[0] = 127.0;
        }

        let mut act = QuantizedActivation::with_capacity(512);
        act.quantize(&x);
        // Quantization must be lossless for this input.
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(act.q[i] as f32, v, "activation quantization not exact at {i}");
        }

        let dequant = match fmt {
            QuantFormat::Q5_0 => q5_0::scalar::run(&row).unwrap(),
            QuantFormat::Q6K => q6_k::scalar::run(&row).unwrap(),
            QuantFormat::Q8_0 => q8_0::scalar::run(&row),
            QuantFormat::Q4K => unreachable!(),
        };
        let want = glproc::kernels::matmul::scalar::dot_f32(&dequant, &x);
        let tol = 1e-3 * want.abs().max(1.0); // f32 accumulation order only

        let scalar = qdot::row_dot_q8(fmt, &row, &act, SimdStrategy::Scalar);
        assert!(
            (scalar - want).abs() <= tol,
            "{fmt:?} scalar qdot: got {scalar}, want {want}"
        );
        if has_avx2() {
            let got = qdot::row_dot_q8(fmt, &row, &act, SimdStrategy::Avx2);
            assert!(
                (got - want).abs() <= tol,
                "{fmt:?} avx2 qdot: got {got}, want {want}"
            );
            // And scalar vs AVX2 agree tightly on the same integer inputs.
            assert!(
                (got - scalar).abs() <= 1e-3 * scalar.abs().max(1.0),
                "{fmt:?} avx2 vs scalar qdot: {got} vs {scalar}"
            );
        }
    }
}

#[test]
fn fast_exp_avx2_matches_scalar() {
    if !has_avx2() {
        return;
    }
    // Test range: [-10.0, 10.0] step 0.1
    let mut x = -10.0f32;
    while x <= 10.0 {
        let scalar = x.exp();
        let avx2 = unsafe { glproc::kernels::ops::fast_exp::avx2::run(x) };
        let rel_err = (avx2 - scalar).abs() / scalar;
        assert!(
            rel_err <= 1e-4,
            "x={}, got={}, want={}, rel_err={}", x, avx2, scalar, rel_err
        );
        x += 0.1;
    }
}

/// X5 test 9: after `warm_and_lock_model`, every weight buffer is resident
/// and readable with its original contents. Pinning itself is best-effort
/// (quota-dependent), so accessibility is the contract under test.
#[test]
fn warm_model_pages_loaded() {
    use glproc::kernels::bridge::QuantFormat;
    use glproc::loader::warm_and_lock_model;
    use glproc::model::{GlprocModel, LayerWeights, ModelConfig, RopeStyle, WeightMatrix};

    let dim = 64usize;
    let vocab = 8usize;
    let layer = LayerWeights {
        attn_norm: vec![1.0; dim],
        wq: WeightMatrix::F32(vec![0.5; dim * dim]),
        wk: WeightMatrix::F32(vec![0.5; dim * dim]),
        wv: WeightMatrix::F32(vec![0.5; dim * dim]),
        wo: WeightMatrix::F32(vec![0.5; dim * dim]),
        bq: None,
        bk: None,
        bv: None,
        q_norm: None,
        k_norm: None,
        ffn_norm: vec![1.0; dim],
        // One quantized matrix so the raw-bytes region path is exercised.
        w_gate: WeightMatrix::Quant(QuantFormat::Q8_0, vec![7u8; dim / 32 * 34 * dim]),
        w_up: WeightMatrix::F32(vec![0.5; dim * dim]),
        w_down: WeightMatrix::F32(vec![0.5; dim * dim]),
    };
    let model = GlprocModel {
        config: ModelConfig {
            arch: "test".into(),
            dim,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: dim / 4,
            hidden_dim: dim,
            vocab_size: vocab,
            max_seq: 32,
            rms_eps: 1e-5,
            rope_freq_base: 10_000.0,
            rope_style: RopeStyle::Neox,
        },
        token_embd: vec![0.25; vocab * dim],
        layers: vec![layer],
        output_norm: vec![1.0; dim],
        output: WeightMatrix::F32(vec![0.5; vocab * dim]),
    };

    warm_and_lock_model(&model);

    assert!(model.token_embd.iter().all(|&v| v == 0.25));
    assert!(model.output_norm.iter().all(|&v| v == 1.0));
    match &model.layers[0].w_gate {
        WeightMatrix::Quant(QuantFormat::Q8_0, b) => assert!(b.iter().all(|&v| v == 7)),
        _ => unreachable!("w_gate was constructed as Q8_0"),
    }
    match &model.layers[0].wq {
        WeightMatrix::F32(w) => assert!(w.iter().all(|&v| v == 0.5)),
        _ => unreachable!("wq was constructed as F32"),
    }
}

/// Q5_0 -> Q8_0 repack is bit-exact: dequantizing the repacked Q8_0 blocks
/// must reproduce the Q5_0 dequantization exactly (integer x8 / scale /8).
#[test]
fn q5_0_repack_to_q8_0_is_exact() {
    use glproc::kernels::dequant::{q5_0, q8_0};

    let mut rng = Lcg::new(77);
    let n_blocks = 64;
    let mut data = vec![0u8; n_blocks * 22];
    rng.fill(&mut data);
    // Force a normal, comfortably-sized f16 scale per block (exp > 3).
    for block in data.chunks_mut(22) {
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    }

    let want = q5_0::scalar::run(&data).unwrap();
    let repacked = q5_0::scalar::repack_to_q8_0(&data).unwrap();
    assert_eq!(repacked.len(), n_blocks * 34);
    let got = q8_0::scalar::run(&repacked);

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "repack mismatch at weight {i}");
    }
}
