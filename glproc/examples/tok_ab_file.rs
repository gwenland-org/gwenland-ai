//! Token-id A/B on an arbitrary text file — old `glcore::tokenizer` vs
//! `gltokenizer`, against a real model's vocabulary.
#![allow(deprecated)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model = a.next().ok_or("usage: tok_ab_file <model.gguf> <text-file>")?;
    let file = a.next().ok_or("usage: tok_ab_file <model.gguf> <text-file>")?;
    let text = std::fs::read_to_string(&file)?;

    let gguf = glcore::format::gguf::GgufFile::open(&model)?;
    let old = glcore::tokenizer::Tokenizer::from_gguf(&gguf)?;
    let new = gltokenizer::Tokenizer::from_gguf_path(&model)?;
    let add_bos = old.add_bos_default();

    let x = old.encode(&text, add_bos);
    let y = new.encode(&text, add_bos)?;
    println!("{} chars, add_bos={add_bos}", text.len());
    println!("old={} tokens  new={} tokens", x.len(), y.len());

    let n = x.len().min(y.len());
    match (0..n).find(|&i| x[i] != y[i]) {
        None if x.len() == y.len() => println!("IDENTICAL"),
        None => println!("common prefix identical; lengths differ at the tail"),
        Some(i) => {
            let (lo, hi) = (i.saturating_sub(3), (i + 4).min(n));
            println!("first divergence @{i}");
            println!("  old {:?} = {:?}", &x[lo..hi], old.decode(&x[lo..hi], false));
            println!("  new {:?} = {:?}", &y[lo..hi], new.decode(&y[lo..hi], false));
        }
    }
    Ok(())
}
