//! `glbench` — the Mensura Veritatis command-line interface.
//!
//! Subcommands:
//!   glbench run     --engine <name> --model <path> [options]   run a benchmark
//!   glbench ab      --engine <name> --model <A> --model <B>    A/B N models in one command
//!   glbench compare <baseline.json> <candidate.json>           diff two runs
//!   glbench inspect <session.json>                             re-render an archive
//!   glbench export  <session.json> --format <json|md|csv>       convert an archive
//!
//! Argument parsing is hand-rolled (the crate takes zero external deps, so no
//! clap here). Parsing is intentionally small and forgiving of flag order.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glbench::comparison::runs;
use glbench::core::workload::{WorkloadKind, WorkloadSpec};
use glbench::export::{csv, markdown};
use glbench::render::text;
use glbench::runner::planner;
use glbench::storage::archive;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("ab") => cmd_ab(&args[1..]),
        Some("compare") => cmd_compare(&args[1..]),
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("glbench: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage:
  glbench run     --engine <name> --model <path> [--prompt <text>] [--tokens N]
                  [--warmup N] [--iters N] [--temperature F] [--seed N]
                  [--kind prefill|decode|end_to_end|stress] [--cot on|off]
                  [--out <file.json>]
  glbench ab      --engine <name> --model <baseline> --model <candidate> [...]
                  (same options as run; each extra --model is benchmarked with
                   the identical workload, sequentially, and diffed against the
                   first — sequential on purpose: parallel runs would contend
                   for bandwidth and corrupt every number)
  glbench compare <baseline.json> <candidate.json> [--threshold F]
  glbench inspect <session.json>
  glbench export  <session.json> --format <json|md|csv> [--out <file>]

glbench measures engine performance; it does not optimize it.";

fn print_usage() {
    println!("{USAGE}");
}

/// Flags shared by `run` and `ab`, parsed once. `models` collects every
/// `--model` in order — `run` demands exactly one, `ab` at least two.
struct RunArgs {
    spec: WorkloadSpec,
    models: Vec<String>,
    out_path: Option<PathBuf>,
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut spec = WorkloadSpec::default();
    let mut models = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut prompt_set = false;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        // Pull the next token as this flag's value, advancing the cursor.
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("flag '{flag}' needs a value"))
        };
        match flag.as_str() {
            "--engine" => spec.engine = value(&mut i)?,
            "--model" => models.push(value(&mut i)?),
            "--prompt" => {
                spec.prompt = value(&mut i)?;
                prompt_set = true;
            }
            "--tokens" => spec.max_new_tokens = parse_num(&value(&mut i)?, &flag)?,
            "--warmup" => spec.warmup_iters = parse_num(&value(&mut i)?, &flag)?,
            "--iters" => spec.measure_iters = parse_num(&value(&mut i)?, &flag)?,
            "--temperature" => spec.temperature = parse_f32(&value(&mut i)?, &flag)?,
            "--seed" => spec.seed = parse_num::<u64>(&value(&mut i)?, &flag)?,
            "--kind" => {
                let k = value(&mut i)?;
                spec.kind = WorkloadKind::from_str(&k)
                    .ok_or_else(|| format!("unknown --kind '{k}'"))?;
            }
            // Manual CoT override; unset ("auto") lets the GGUF header decide.
            "--cot" => {
                spec.cot_mode = match value(&mut i)?.as_str() {
                    "on" | "true" => Some(true),
                    "off" | "false" => Some(false),
                    other => return Err(format!("--cot takes on|off, got '{other}'")),
                };
            }
            "--out" => out_path = Some(PathBuf::from(value(&mut i)?)),
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }

    if !prompt_set {
        // A representative default prompt (~long enough to exercise prefill).
        spec.prompt = default_prompt();
    }
    Ok(RunArgs { spec, models, out_path })
}

/// Progress heartbeat to stderr so stdout stays the report.
fn progress(phase: &str, iter: usize, total: usize) {
    eprintln!("[{phase}] {}/{}", iter + 1, total.max(1));
}

/// `glbench run` — execute a benchmark and print (and optionally archive) it.
fn cmd_run(args: &[String]) -> Result<(), String> {
    let mut a = parse_run_args(args)?;
    match a.models.len() {
        0 => return Err("--model is required".into()),
        1 => a.spec.model_path = a.models.remove(0),
        _ => return Err("run takes one --model; use 'glbench ab' for several".into()),
    }

    let session = planner::run(&a.spec, &progress).map_err(|e| e.to_string())?;

    // Report to stdout.
    print!("{}", text::session(&session));

    // Archive if requested.
    if let Some(path) = a.out_path {
        archive::write(&session, &path)?;
        eprintln!("archived to {}", path.display());
    }
    Ok(())
}

/// `glbench ab` — benchmark N models under one identical workload, in one
/// command, and diff each against the first.
///
/// Runs are sequential by design: models on one machine share the memory bus,
/// so "parallel benchmark" is an oxymoron — two decodes contending for
/// bandwidth would each report a number describing neither. What v2 adds is
/// the orchestration (one command, one workload, N models, a delta table),
/// not concurrency.
fn cmd_ab(args: &[String]) -> Result<(), String> {
    let a = parse_run_args(args)?;
    if a.models.len() < 2 {
        return Err("ab needs at least two --model flags (baseline first)".into());
    }
    if a.out_path.is_some() {
        return Err("ab does not take --out; archive individual runs with 'run'".into());
    }

    // Run them all under the byte-identical spec except the model path.
    let mut sessions = Vec::new();
    for (n, model) in a.models.iter().enumerate() {
        eprintln!("=== model {}/{}: {model}", n + 1, a.models.len());
        let mut spec = a.spec.clone();
        spec.model_path = model.clone();
        sessions.push(planner::run(&spec, &progress).map_err(|e| e.to_string())?);
    }

    // Every session's full report first (the facts) ...
    for s in &sessions {
        print!("{}", text::session(s));
        println!();
    }
    // ... then each candidate against the baseline (the deltas).
    let baseline = &sessions[0];
    for candidate in &sessions[1..] {
        let report = runs::compare(baseline, candidate, 0.05);
        print!("{}", text::comparison(&report));
        println!();
    }
    Ok(())
}

/// `glbench compare` — diff two archived sessions.
fn cmd_compare(args: &[String]) -> Result<(), String> {
    let mut positional: Vec<&String> = Vec::new();
    let mut threshold = 0.05;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--threshold" => {
                i += 1;
                threshold = parse_f64(args.get(i).ok_or("--threshold needs a value")?, "--threshold")?;
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }
    if positional.len() != 2 {
        return Err("compare needs exactly two archive paths".into());
    }
    let baseline = archive::read(Path::new(positional[0]))?;
    let candidate = archive::read(Path::new(positional[1]))?;
    let report = runs::compare(&baseline, &candidate, threshold);
    print!("{}", text::comparison(&report));
    Ok(())
}

/// `glbench inspect` — re-render an archived session to the terminal.
fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("inspect needs an archive path")?;
    let session = archive::read(Path::new(path))?;
    print!("{}", text::session(&session));
    Ok(())
}

/// `glbench export` — convert an archive to json/markdown/csv.
fn cmd_export(args: &[String]) -> Result<(), String> {
    let mut input: Option<&String> = None;
    let mut format = "json".to_string();
    let mut out_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--format" => {
                i += 1;
                format = args.get(i).ok_or("--format needs a value")?.clone();
            }
            "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown flag '{other}'"));
            }
            _ => input = Some(&args[i]),
        }
        i += 1;
    }
    let input = input.ok_or("export needs an archive path")?;
    let session = archive::read(Path::new(input))?;

    let rendered = match format.as_str() {
        "json" => session.to_json().to_pretty(),
        "md" | "markdown" => markdown::render(&session),
        "csv" => csv::render(&session),
        other => return Err(format!("unknown --format '{other}' (json|md|csv)")),
    };

    match out_path {
        Some(path) => {
            std::fs::write(&path, rendered).map_err(|e| format!("writing {}: {e}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => print!("{rendered}"),
    }
    Ok(())
}

fn default_prompt() -> String {
    // ~repeated so prefill has real work; kept deterministic.
    let base = "Explain how a modern GPU executes a matrix multiplication, \
                covering threads, warps, shared memory, and coalesced loads. ";
    base.repeat(8).trim().to_string()
}

fn parse_num<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
    s.parse::<T>().map_err(|_| format!("flag '{flag}': '{s}' is not a valid integer"))
}

fn parse_f32(s: &str, flag: &str) -> Result<f32, String> {
    s.parse::<f32>().map_err(|_| format!("flag '{flag}': '{s}' is not a valid number"))
}

fn parse_f64(s: &str, flag: &str) -> Result<f64, String> {
    s.parse::<f64>().map_err(|_| format!("flag '{flag}': '{s}' is not a valid number"))
}
