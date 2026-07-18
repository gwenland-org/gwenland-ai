use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // vergen v8 syntax
    EmitBuilder::builder()
        .git_sha(true)
        .emit()?;

    println!("cargo:rerun-if-changed=../../package.json");

    let version = std::fs::read_to_string("../../package.json")
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .and_then(|json| json["version"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GWEN_VERSION={}", version);
    Ok(())
}