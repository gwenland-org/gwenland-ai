//! Teacher-forced Q8 diagnostics: compare identical histories, avoiding
//! autoregressive amplification after the first different greedy choice.
use glcore::{format::gguf::GgufFile, tokenizer::GllmTokenizer};
use glcuda::{driver::Cuda, kernels::KernelSet, model::GpuModel};
use glproc::runner::Runner;

fn top(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: wave58_logits MODEL PROMPT [sequential]")?;
    let prompt = std::fs::read_to_string(args.get(2).ok_or("missing prompt")?)?;
    let gguf = GgufFile::open(path)?;
    let tokenizer = GllmTokenizer::from_gguf_path(path)?;
    let ids = tokenizer.encode_chat(&prompt)?.ok_or("ChatML required")?;
    let cpu = glproc::loader::load_gguf(&gguf)?;
    let mut runner = Runner::new(&cpu);
    // Match the CPU production chunk size; the diagnostic API uses a
    // preallocated 32-token workspace, not an arbitrary prompt-sized batch.
    for (index, chunk) in ids.chunks(32).enumerate() {
        runner.forward_chunk_into(chunk, index * 32)?;
    }
    let cuda = Cuda::probe()?;
    let kernels = KernelSet::load(&cuda)?;
    let mut gpu = GpuModel::upload(&cuda, glcuda::loader::load_host(&gguf)?)?;
    if args.get(3).map(String::as_str) == Some("sequential") {
        for (pos, &id) in ids.iter().enumerate() {
            gpu.step(&cuda, &kernels, id, pos, pos + 1 == ids.len())?;
        }
    } else {
        gpu.prefill_batched(&cuda, &kernels, &ids)?;
    }
    for index in 0..50 {
        let a = runner.logits();
        let b = gpu.logits_host(&cuda)?;
        if a.len() != b.len() || !a.iter().chain(b.iter()).all(|x| x.is_finite()) {
            return Err("invalid logits".into());
        }
        let at = top(a);
        let bt = top(b);
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        let rms = (a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt();
        println!("{{\"step\":{index},\"cpu_top\":{at},\"gpu_top\":{bt},\"max_abs\":{max_abs},\"rms\":{rms},\"cpu_margin_over_gpu\":{},\"gpu_margin_over_cpu\":{}}}", a[at]-a[bt], b[bt]-b[at]);
        if index < 49 {
            let pos = ids.len() + index;
            runner.forward_into(at as u32, pos)?;
            gpu.decode_step(&cuda, &kernels, at as u32, pos)?;
        }
    }
    gpu.free(&cuda)?;
    Ok(())
}
