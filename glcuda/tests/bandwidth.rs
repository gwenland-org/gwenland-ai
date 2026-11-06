//! The device read probe on real hardware. Skips without a CUDA device.

use glcuda::bandwidth::measure_read_gbs;
use glcuda::driver::Cuda;

#[test]
fn the_probe_reports_a_plausible_rate_and_frees_what_it_took() {
    let Ok(cuda) = Cuda::probe() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };
    let (free_before, _) = cuda.mem_get_info().unwrap();
    let gbs = measure_read_gbs(&cuda).unwrap();
    let (free_after, _) = cuda.mem_get_info().unwrap();

    eprintln!("{}: measured read {gbs:.1} GB/s", cuda.info.name);
    // Wide on purpose: this rejects a broken timer or a kernel the compiler
    // emptied, not a slow card. No GPU this engine targets reads under 20 GB/s,
    // and none reads 20 TB/s from a buffer this far past its L2.
    assert!((20.0..20_000.0).contains(&gbs), "implausible rate {gbs} GB/s");
    assert!(
        free_after + (1 << 20) >= free_before,
        "the probe leaked VRAM: {free_before} -> {free_after}"
    );
}
