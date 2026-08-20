//! `glbench` — the Mensura Veritatis command-line interface.
//!
//! Subcommands:
//!   glbench run     --engine <name> --model <path> [options]   run a benchmark
//!   glbench ab      --engine <name> --model <A> --model <B>    A/B N models in one command
//!   glbench compare <baseline.json> <candidate.json>           diff two runs
//!   glbench validate --engine <name> --model <path> --against <oracle>
//!                                                               numerical parity vs an oracle engine
//!   glbench inspect <session.json>                             re-render an archive
//!   glbench export  <session.json> --format <json|md|csv>       convert an archive
//!   glbench join    <a.json> <b.json> --out <join.json>         compare two archives into a third
//!
//! Argument parsing is hand-rolled (the crate takes zero external deps, so no
//! clap here). Parsing is intentionally small and forgiving of flag order.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use glbench::comparison::runs;
use glbench::core::workload::{WorkloadKind, WorkloadSpec};
use glbench::export::{csv, markdown};
#[cfg(feature = "gllm-bench")]
use glbench::kl_divergence::{self, KlDivArgs};
#[cfg(feature = "gllm-bench")]
use glbench::ppl::{self, PplArgs};
use glbench::quant_info::{self, QuantInfoArgs};
use glbench::render::text;
use glbench::runner::{planner, scale, thread_scale};
use glbench::numerical::scope::{self, ENBitScope};
#[cfg(feature = "train-bench")]
use glbench::training::runner::{self as train_runner, TrainArgs};
use glbench::storage::{archive, join};
use glbench::validation::availability::ENNullSemantics;
#[cfg(feature = "gllm-bench")]
use glbench::tensor_stats::{self, TensorStatsArgs};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("ab") => cmd_ab(&args[1..]),
        Some("compare") => cmd_compare(&args[1..]),
        Some("validate") => cmd_validate(&args[1..]),
        Some("scale") => cmd_scale(&args[1..]),
        Some("thread-scale") => cmd_thread_scale(&args[1..]),
        Some("accuracy-vs-perf") => cmd_accuracy_vs_perf(&args[1..]),
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
        Some("join") => cmd_join(&args[1..]),
        #[cfg(feature = "train-bench")]
        Some("train") => cmd_train(&args[1..], glbench::core::mode::ENSessionMode::TrainingOnly),
        #[cfg(feature = "train-bench")]
        Some("unified") => cmd_train(&args[1..], glbench::core::mode::ENSessionMode::Unified),
        // Recognised without the feature, so a user who read about `glbench
        // train` learns which flag they need rather than that the command does
        // not exist (design §8).
        #[cfg(not(feature = "train-bench"))]
        Some(cmd @ ("train" | "unified")) => Err(format!(
            "'{cmd}' requires a build with --features train-bench \
             (it links stumman, which a default build never compiles)"
        )),
        Some("quant-info") => cmd_quant_info(&args[1..]),
        #[cfg(feature = "gllm-bench")]
        Some("ppl") => cmd_ppl(&args[1..]),
        #[cfg(feature = "gllm-bench")]
        Some("kl-div") => cmd_kl_div(&args[1..]),
        #[cfg(feature = "gllm-bench")]
        Some("tensor-stats") => cmd_tensor_stats(&args[1..]),
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
                  [--cold-iters N] [--warmup N] [--iters N] [--temperature F]
                  [--seed N] [--kind prefill|decode|end_to_end|stress]
                  [--cot on|off] [--verify-against <oracle>] [--out <file.json>]
                  (--verify-against loads a second engine and cross-checks the
                   first 50 generated tokens against it, folding the result into
                   the validation report; skipped, not just trivial, when the
                   oracle equals --engine)
  glbench ab      --engine <name> --model <baseline> --model <candidate> [...]
                  (same options as run; each extra --model is benchmarked with
                   the identical workload, sequentially, and diffed against the
                   first — sequential on purpose: parallel runs would contend
                   for bandwidth and corrupt every number)
  glbench compare <baseline.json> <candidate.json> [--threshold F]
  glbench validate --engine <name> --model <path> --against <oracle>
                  [--prompt <text>] [--tokens N]
                  (runs <engine> and <oracle> on the identical prompt under
                   greedy decoding and reports the matching token prefix;
                   default oracle is 'glproc')
  glbench validate --availability <archive.json>
                  (a different check: the D-10 invariant over an archive that
                   already exists — every null carries an availability status,
                   and no status sits on a field that has a value)
  glbench scale   --engine <name> --model <path> --sweep N,N,N,...
                  (runs the identical prompt at each token budget in --sweep,
                   sequentially, and classifies how decode throughput scales)
  glbench thread-scale --engine glproc --model <path> --sweep N,N,N,...
                  (runs the identical prompt at each GLPROC_THREADS value in
                   --sweep, sequentially, and reports speedup/efficiency
                   relative to the lowest thread count — glproc only, since
                   GLPROC_THREADS is glproc's own env override)
  glbench inspect <session.json> [--no-verify]
                  (verifies the archive's content digest by default; a
                   mismatch is reported and the session is still rendered,
                   because refusing to show a modified archive is useless
                   exactly when you need to see what changed)
  glbench export  <session.json> --format <json|md|csv> [--out <file>]
  glbench join    <a.json> <b.json> --out <join.json> [--label <name>]
                  [--threshold F] [--no-verify]
                  (compares two archives into a third file that records each
                   source's content digest; neither source is opened for
                   writing, and 'glbench inspect <join.json>' re-checks them
                   so a source edited afterwards shows up as drift.
                   Exactly two sources in v3)
  glbench accuracy-vs-perf <run.json> <accuracy.json>
                  (joins a `run` archive's throughput with a `kl-div` or
                   `ppl` archive's numerical-accuracy figures, side by side —
                   no new measurement, both archives must already exist)
  glbench quant-info --model <package-dir> [--out <file.json>]
                  (static inspection of a .gllm package: dtype tally and
                   quantization coverage, no inference, no model load;
                   --model names the package directory, e.g. the folder
                   containing gllm.json — ZIP-archive .gllm files are not
                   yet readable, see glictus-caliburni ARTX06)

training observation (needs --features train-bench):
  glbench train   [--d-in N] [--d-out N] [--rank N] [--samples N]
                  [--epochs N] [--lr F] [--seed N] [--dataset-seed N]
                  [--step-sample N] [--target-loss F] [--bit-scope <list>]
                  [--label <name>] [--out <file.json>]
                  (runs a LoRA fine-tune on stumman under observation and
                   archives it as a training_only v2 session. glbench does not
                   drive the loop — it installs an observer and calls
                   stumman's own Trainer::train)
  glbench unified [same options as train]
                  (a training run with inference roles labelled either side of
                   it: the outer envelope is pre_training, training.post_eval
                   is post_training)

                  There is deliberately no --model or --dataset: stumman M2
                  generates its frozen base weight from a seed and builds its
                  dataset in memory, so neither flag has a subject. The shape
                  and seed flags above fully determine the run, which makes it
                  reproducible in a way a path would not.

                  --target-loss has NO default. Time-to-target needs someone to
                  say what good means; without it, steps_to_target is archived
                  as absent rather than guessed.

bit profiling (run):
  --profile bits              profile the model's weight tensors at the bit
                              level after the benchmark (GLBitProf). Static —
                              it reads the model file, not the run, so it
                              cannot perturb the timings.
  --bit-scope <scope>         weights (default) | gradients | optimizer.
                              weights needs --features gllm-bench; the two
                              training scopes are Wave 4 and say so.

archive options (run, ab, scale, ...):
  --null-semantics strict|lenient
                  (strict, the default, refuses to write a session with an
                   unexplained null or a mode/content disagreement; lenient
                   downgrades both to warnings and writes anyway)

glbench measures engine performance; it does not optimize it.";

/// Only present when built with `--features gllm-bench` — appended to
/// [`USAGE`] at print time so a default build's usage text never mentions a
/// command it does not have.
#[cfg(feature = "gllm-bench")]
const PPL_USAGE: &str = "\
  glbench ppl     --model <package-dir> --gguf <original.gguf>
                  [--context N] [--stride N] [--out <file.json>]
                  (perplexity over an embedded WikiText-2 sample via
                   teacher-forced log-probs; --gguf supplies the tokenizer,
                   since .gllm packages do not embed one yet, ARTX1 OQ3;
                   default context 512, stride 256 — the known garbage-output
                   bug this was diagnostic-only for is fixed, but this number
                   is not yet re-validated, see glbench ppl's own output)
";

#[cfg(feature = "gllm-bench")]
const KL_DIV_USAGE: &str = "\
  glbench kl-div  --model <package-dir> --gguf <original.gguf>
                  [--tokens N] [--out <file.json>]
                  (per-position KL-divergence between the .gllm package and
                   glproc::runner::Runner's logits, teacher-forced over the
                   same embedded WikiText-2 sample as ppl; default 64 tokens
                   — see glbench::kl_divergence module docs for why this
                   command exists)
";

#[cfg(feature = "gllm-bench")]
const TENSOR_STATS_USAGE: &str = "\
  glbench tensor-stats --model <package-dir> [--out <file.json>]
                  [--full] [--norm-only]
                  (decode every tensor in the package and flag NaN / Inf /
                   zero-variance tensors — a fast structural sanity check,
                   see glbench::tensor_stats module docs; --full adds a
                   per-tensor mean/std/min/max distribution to the output;
                   --norm-only restricts the scan to *norm.weight tensors,
                   the RMSNorm gamma weights, for --full --norm-only together)
";

fn full_usage() -> String {
    #[cfg(feature = "gllm-bench")]
    {
        format!("{USAGE}\n{PPL_USAGE}\n{KL_DIV_USAGE}\n{TENSOR_STATS_USAGE}")
    }
    #[cfg(not(feature = "gllm-bench"))]
    {
        USAGE.to_string()
    }
}

fn print_usage() {
    println!("{}", full_usage());
}

/// Flags shared by `run` and `ab`, parsed once. `models` collects every
/// `--model` in order — `run` demands exactly one, `ab` at least two.
struct RunArgs {
    spec: WorkloadSpec,
    models: Vec<String>,
    out_path: Option<PathBuf>,
    /// How a D-10 violation is treated when the archive is written.
    null_semantics: ENNullSemantics,
    /// Which tensor family to bit-profile, if `--profile bits` was given.
    bit_scope: Option<ENBitScope>,
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut spec = WorkloadSpec::default();
    let mut models = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut prompt_set = false;
    let mut null_semantics = ENNullSemantics::default();
    // `--profile bits` and `--bit-scope` are separate flags so the scope can be
    // named before or after the profile is asked for, in either order.
    let mut profile_bits = false;
    let mut bit_scope: Option<ENBitScope> = None;

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
            "--cold-iters" => spec.cold_iters = parse_num(&value(&mut i)?, &flag)?,
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
            "--verify-against" => spec.verify_against = Some(value(&mut i)?),
            "--null-semantics" => {
                let v = value(&mut i)?;
                null_semantics = ENNullSemantics::from_str(&v)
                    .ok_or_else(|| format!("--null-semantics takes strict|lenient, got '{v}'"))?;
            }
            "--profile" => {
                let v = value(&mut i)?;
                match v.as_str() {
                    "bits" => profile_bits = true,
                    other => return Err(format!("--profile takes 'bits', got '{other}'")),
                }
            }
            "--bit-scope" => {
                let v = value(&mut i)?;
                let scope = ENBitScope::from_str(&v).ok_or_else(|| {
                    format!("--bit-scope takes weights|gradients|optimizer, got '{v}'")
                })?;
                // Refuse a scope this build cannot collect at parse time, not
                // after a full benchmark run has already been paid for.
                scope.availability()?;
                bit_scope = Some(scope);
            }
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }

    // `--profile bits` with no `--bit-scope` means weights (Wave 2's only
    // collectable scope); `--bit-scope` on its own implies `--profile bits`,
    // because naming a scope is already asking for the profile.
    let bit_scope = match (profile_bits, bit_scope) {
        (_, Some(scope)) => Some(scope),
        (true, None) => {
            let scope = ENBitScope::Weights;
            scope.availability()?;
            Some(scope)
        }
        (false, None) => None,
    };

    if !prompt_set {
        // A representative default prompt (~long enough to exercise prefill).
        spec.prompt = default_prompt();
    }
    Ok(RunArgs { spec, models, out_path, null_semantics, bit_scope })
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

    // GLBitProf, if asked for. Runs after the measured iterations and reads the
    // model file, not the run — profiling is static, so it cannot perturb the
    // timings printed above.
    if let Some(bit_scope) = a.bit_scope {
        print!("{}", render_bit_profile(&session, bit_scope)?);
    }

    // Archive if requested.
    if let Some(path) = a.out_path {
        let report = archive::write_with_policy(&session, &path, a.null_semantics)?;
        for finding in &report.findings {
            eprintln!("glbench: [{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
        }
        eprintln!("archived to {}", path.display());
    }
    Ok(())
}

/// Render the GLBitProf summary for a finished session.
///
/// Human-readable only, deliberately: Wave 2 wires the math and the CLI
/// surface, and the archive projection of `VLBitProfile` lands with the rest of
/// the training plumbing in Wave 4. Printing a number the archive cannot yet
/// carry is better than half a schema.
fn render_bit_profile(
    session: &glbench::BenchmarkSession,
    bit_scope: ENBitScope,
) -> Result<String, String> {
    use std::fmt::Write as _;

    let scopes = scope::scope_weights_for_session(session)?;
    let mut out = String::new();
    let _ = writeln!(out, "\nGLBitProf — scope {}", bit_scope.as_str());
    let _ = writeln!(out, "{}", "\u{2500}".repeat(78));

    if scopes.is_empty() {
        let _ = writeln!(out, "no tensor in this model could be decoded to f32.");
        return Ok(out);
    }

    let _ = writeln!(
        out,
        "{:<34} {:>10} {:>6} {:>9} {:>7} {:>7}",
        "tensor", "count", "sign", "exp range", "dyn", "mantissa"
    );
    for entry in &scopes {
        let p = &entry.profile;
        let mantissa = match p.mantissa_entropy_bits {
            // The skipped case prints "skipped", never 0.0 — a tensor whose
            // mantissa was never profiled has no entropy, which is a different
            // statement from having none.
            None => "skipped".to_string(),
            Some(bits) => format!("{bits:.2}b"),
        };
        let _ = writeln!(
            out,
            "{:<34} {:>10} {:>5.1}% {:>4}..{:<4} {:>7.4} {:>7}",
            truncate_name(&entry.tensor_name, 34),
            p.count,
            p.sign_set_ratio * 100.0,
            p.exponent_min,
            p.exponent_max,
            p.dynamic_range_used,
            mantissa
        );
    }

    // Per-position entropy is the axis a structured bug shows up on, so the
    // extremes are named rather than left in a 32-element array nobody reads.
    let (mut max_h, mut max_at, mut min_h, mut min_at) = (f64::MIN, 0usize, f64::MAX, 0usize);
    for entry in &scopes {
        for (i, &h) in entry.profile.bit_entropy.iter().enumerate() {
            if h > max_h {
                max_h = h;
                max_at = i;
            }
            if h < min_h {
                min_h = h;
                min_at = i;
            }
        }
    }
    let _ = writeln!(
        out,
        "\nbit entropy across {} tensors: max {max_h:.4} at bit {max_at}, min {min_h:.4} at bit {min_at}",
        scopes.len()
    );
    let skipped = scopes.iter().filter(|s| s.profile.mantissa_sparse_skipped).count();
    let _ = writeln!(
        out,
        "mantissa map: {} of {} tensors profiled at full 23-bit resolution \
         ({skipped} over the {}-element cap)",
        scopes.len() - skipped,
        scopes.len(),
        glbench::numerical::bitprof::MANTISSA_SPARSE_CAP
    );
    Ok(out)
}

/// Shorten a tensor name from the left, keeping the distinguishing tail.
fn truncate_name(name: &str, width: usize) -> String {
    if name.len() <= width {
        return name.to_string();
    }
    format!("...{}", &name[name.len() - (width - 3)..])
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

/// `glbench validate` — numerical parity of `--engine` against `--against`
/// (default `glproc`, DESIGN.md's oracle) on the identical prompt.
fn cmd_validate(args: &[String]) -> Result<(), String> {
    let mut against = "glproc".to_string();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            // A different check entirely: the D-10 invariant over an archive
            // that already exists, for files written by an older build.
            "--availability" => {
                i += 1;
                let path = args.get(i).ok_or("--availability needs an archive path")?;
                return cmd_validate_availability(Path::new(path));
            }
            "--against" => {
                i += 1;
                against = args.get(i).ok_or("--against needs a value")?.clone();
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    let mut a = parse_run_args(&rest)?;
    let candidate = match a.models.len() {
        0 => return Err("--model is required".into()),
        1 => a.models.remove(0),
        _ => return Err("validate takes one --model".into()),
    };
    a.spec.model_path = candidate;

    if a.spec.engine == against {
        return Err(format!("--engine and --against are both '{against}'"));
    }

    let report = glbench::validation::validate_against_oracle(&a.spec, &against, &a.spec.engine)
        .map_err(|e| e.to_string())?;
    print!("{}", text::parity(&report));

    if !report.passed() {
        return Err("numerical parity check failed".into());
    }
    Ok(())
}

/// `glbench validate --availability` — the D-10 invariant over an archive.
fn cmd_validate_availability(path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let report = glbench::validation::availability::check(&text);
    if report.findings.is_empty() {
        println!("{}: every null carries an availability status.", path.display());
        return Ok(());
    }
    for finding in &report.findings {
        println!("[{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
    }
    if !report.passed() {
        return Err(format!("{}: null-semantics check failed", path.display()));
    }
    Ok(())
}

/// `glbench scale` — decode-throughput sweep over `--sweep`'s token budgets.
fn cmd_scale(args: &[String]) -> Result<(), String> {
    let mut sweep_arg: Option<String> = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--sweep" => {
                i += 1;
                sweep_arg = Some(args.get(i).ok_or("--sweep needs a value")?.clone());
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    let sweep_arg = sweep_arg.ok_or("scale needs --sweep N,N,N,...")?;
    let mut axis_values = Vec::new();
    for part in sweep_arg.split(',') {
        axis_values.push(parse_num::<usize>(part.trim(), "--sweep")?);
    }
    if axis_values.len() < 2 {
        return Err("--sweep needs at least two values to say anything about scaling".into());
    }

    let mut a = parse_run_args(&rest)?;
    match a.models.len() {
        0 => return Err("--model is required".into()),
        1 => a.spec.model_path = a.models.remove(0),
        _ => return Err("scale takes one --model".into()),
    }

    let report = scale::run_sweep(&a.spec, &axis_values, &progress).map_err(|e| e.to_string())?;
    print!("{}", text::sweep(&report));
    Ok(())
}

/// `glbench thread-scale` — decode-throughput sweep over `--sweep`'s
/// `GLPROC_THREADS` values. See [`thread_scale`]'s module docs for why this
/// is `glproc`-only.
fn cmd_thread_scale(args: &[String]) -> Result<(), String> {
    let mut sweep_arg: Option<String> = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--sweep" => {
                i += 1;
                sweep_arg = Some(args.get(i).ok_or("--sweep needs a value")?.clone());
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }

    let sweep_arg = sweep_arg.ok_or("thread-scale needs --sweep N,N,N,...")?;
    let mut thread_counts = Vec::new();
    for part in sweep_arg.split(',') {
        thread_counts.push(parse_num::<usize>(part.trim(), "--sweep")?);
    }
    if thread_counts.len() < 2 {
        return Err("--sweep needs at least two values to say anything about scaling".into());
    }

    let mut a = parse_run_args(&rest)?;
    match a.models.len() {
        0 => return Err("--model is required".into()),
        1 => a.spec.model_path = a.models.remove(0),
        _ => return Err("thread-scale takes one --model".into()),
    }
    if a.spec.engine != "glproc" {
        return Err(format!(
            "thread-scale only supports --engine glproc (GLPROC_THREADS has no effect on '{}'); \
             see thread_scale module docs for why other engines are not silently swept",
            a.spec.engine
        ));
    }

    let report = thread_scale::run_thread_sweep(&a.spec, &thread_counts, &progress).map_err(|e| e.to_string())?;
    print!("{}", text::thread_sweep(&report));
    Ok(())
}

/// `glbench train` / `glbench unified` — observe a stumman training run.
///
/// D-05: glbench never drives the loop. It builds a `Trainer`, installs a
/// collector, and calls `Trainer::train`; every number archived is something
/// stumman reported.
#[cfg(feature = "train-bench")]
fn cmd_train(args: &[String], mode: glbench::core::mode::ENSessionMode) -> Result<(), String> {
    let mut a = TrainArgs { mode, ..TrainArgs::default() };
    let mut null_semantics = ENNullSemantics::default();

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("flag '{flag}' needs a value"))
        };
        match flag.as_str() {
            "--d-in" => a.d_in = parse_num(&value(&mut i)?, &flag)?,
            "--d-out" => a.d_out = parse_num(&value(&mut i)?, &flag)?,
            "--rank" => a.rank = parse_num(&value(&mut i)?, &flag)?,
            "--samples" => a.samples = parse_num(&value(&mut i)?, &flag)?,
            "--epochs" => a.epochs = parse_num(&value(&mut i)?, &flag)?,
            "--lr" => a.lr = parse_f64(&value(&mut i)?, &flag)?,
            "--seed" => a.seed = parse_num(&value(&mut i)?, &flag)?,
            "--dataset-seed" => a.dataset_seed = parse_num(&value(&mut i)?, &flag)?,
            "--step-sample" => a.step_sample_n = parse_num(&value(&mut i)?, &flag)?,
            "--target-loss" => a.target_loss = Some(parse_f32(&value(&mut i)?, &flag)?),
            "--label" => a.label = Some(value(&mut i)?),
            "--out" => a.out_path = Some(PathBuf::from(value(&mut i)?)),
            "--profile" => {
                let v = value(&mut i)?;
                if v != "bits" {
                    return Err(format!("--profile takes 'bits', got '{v}'"));
                }
                if a.bit_scopes.is_empty() {
                    a.bit_scopes.push(ENBitScope::Gradients);
                }
            }
            "--bit-scope" => {
                a.bit_scopes.clear();
                for name in value(&mut i)?.split(',') {
                    let name = name.trim();
                    let scope = ENBitScope::from_str(name).ok_or_else(|| {
                        format!("--bit-scope takes weights|gradients|optimizer, got '{name}'")
                    })?;
                    if scope == ENBitScope::Weights {
                        return Err(
                            "--bit-scope weights profiles a .gllm package, which a training \
                             run does not have; use gradients and/or optimizer here"
                                .to_string(),
                        );
                    }
                    a.bit_scopes.push(scope);
                }
            }
            "--null-semantics" => {
                let v = value(&mut i)?;
                null_semantics = ENNullSemantics::from_str(&v)
                    .ok_or_else(|| format!("--null-semantics takes strict|lenient, got '{v}'"))?;
            }
            // Named explicitly so the error explains the design decision rather
            // than reporting an unknown flag.
            "--model" | "--dataset" => {
                return Err(format!(
                    "'{flag}' has no subject at stumman M2: the frozen base weight is \
                     generated from --seed and the dataset is built in memory from \
                     --samples/--dataset-seed. See `glbench help`."
                ))
            }
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }

    eprintln!(
        "training: lora r={} over {}x{}, {} samples x {} epochs",
        a.rank, a.d_in, a.d_out, a.samples, a.epochs
    );
    let session = train_runner::run(&a)?;
    print!("{}", render_training(&session));

    if let Some(path) = &a.out_path {
        let report = archive::write_with_policy(&session, path, null_semantics)?;
        for finding in &report.findings {
            eprintln!("glbench: [{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
        }
        eprintln!("archived to {}", path.display());
    }
    Ok(())
}

/// Human-readable summary of a finished training session.
#[cfg(feature = "train-bench")]
fn render_training(session: &glbench::BenchmarkSession) -> String {
    use std::fmt::Write as _;

    let Some(t) = session.training.as_ref() else {
        return String::new();
    };
    let mut out = String::new();
    let rule = "\u{2500}".repeat(60);
    let _ = writeln!(out, "\nglbench {} :: {}", session.metadata.session_mode.as_str(), session.metadata.label);
    let _ = writeln!(out, "{rule}");

    if let Some(adapter) = &t.adapter {
        let _ = writeln!(
            out,
            "adapter {} r={} alpha={:.1} over {}x{} | {} trainable / {} base ({:.2}%)",
            adapter.kind,
            adapter.rank,
            adapter.alpha,
            adapter.d_in,
            adapter.d_out,
            adapter.trainable_parameters,
            adapter.base_parameters,
            adapter.parameter_ratio * 100.0
        );
    }
    let _ = writeln!(
        out,
        "optimizer {} | {} epochs | {} steps observed, {} archived (sample N={})",
        t.optimizer, t.epochs, t.steps_observed, t.steps_archived, t.step_sample_n
    );

    if let Some(c) = &t.convergence {
        let _ = writeln!(out, "\nconvergence");
        let _ = writeln!(
            out,
            "  loss {:.6} -> {:.6} (best {:.6} at step {})",
            c.first_loss, c.final_loss, c.best_loss, c.best_step
        );
        let _ = writeln!(
            out,
            "  slope {:+.3e}/step | EMA {:.6} (alpha {:.2})",
            c.slope_per_step, c.ema_final, c.ema_alpha
        );
        // Window and threshold travel with the verdict; a plateau claim without
        // them is an opinion (research §14).
        let _ = writeln!(
            out,
            "  plateau {} over {} steps at threshold {:.1e} | CV {:.4}",
            if c.plateau_detected { "detected" } else { "not detected" },
            c.plateau_window,
            c.plateau_threshold,
            c.cv
        );
        match (c.target_loss, c.steps_to_target) {
            (None, _) => {
                let _ = writeln!(out, "  target: none given (--target-loss has no default)");
            }
            (Some(target), Some(step)) => {
                let _ = writeln!(out, "  target {target:.6} reached at step {step}");
            }
            (Some(target), None) => {
                let _ = writeln!(out, "  target {target:.6} NOT reached in this run");
            }
        }
    }

    if let Some(a) = &t.attribution {
        let _ = writeln!(out, "\nstep time ({:.3} ms mean over {} steps)", a.mean_step_ms, a.steps);
        let _ = writeln!(
            out,
            "  forward {:.1}% | backward {:.1}% | optimizer {:.1}% | unattributed {:.3} ms",
            a.forward_share * 100.0,
            a.backward_share * 100.0,
            a.optimizer_share * 100.0,
            a.unattributed_ms
        );
    }

    if let Some(m) = &t.memory {
        let _ = writeln!(out, "\nmemory");
        let _ = writeln!(out, "  trainable parameters: {} bytes", m.parameter_bytes);
        match m.optimizer_state_bytes {
            Some(b) => {
                let _ = writeln!(out, "  optimizer state: {b} bytes");
            }
            None => {
                let _ = writeln!(out, "  optimizer state: not read (no --bit-scope asked for it)");
            }
        }
        match m.peak_rss_bytes {
            Some(b) => {
                let _ = writeln!(out, "  peak RSS: {b} bytes");
            }
            None => {
                let _ = writeln!(out, "  peak RSS: not available on this platform");
            }
        }
    }

    if !t.bit_profiles.is_empty() {
        let _ = writeln!(out, "\nGLBitProf — {} tensors profiled", t.bit_profiles.len());
        for entry in t.bit_profiles.iter().take(8) {
            let p = &entry.scope.profile;
            let _ = writeln!(
                out,
                "  step {:>5}  {:<24} {:>8} elems  sign {:>5.1}%  exp {}..{}",
                entry.step_index,
                entry.scope.tensor_name,
                p.count,
                p.sign_set_ratio * 100.0,
                p.exponent_min,
                p.exponent_max
            );
        }
        if t.bit_profiles.len() > 8 {
            let _ = writeln!(out, "  ... {} more", t.bit_profiles.len() - 8);
        }
    }
    out
}

/// `glbench inspect` — re-render an archived session, or re-check a join.
///
/// Digest verification is on by default. A mismatch prints as an error finding
/// and the archive is still rendered: refusing to show a modified archive would
/// make the tool useless exactly when a user most needs to see what changed.
fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let mut path: Option<&String> = None;
    let mut verify = true;
    for arg in args {
        match arg.as_str() {
            "--no-verify" => verify = false,
            other if other.starts_with("--") => return Err(format!("unknown flag '{other}'")),
            _ => path = Some(arg),
        }
    }
    let path = Path::new(path.ok_or("inspect needs an archive path")?);

    // A join manifest is a different top-level type, distinguished by the one
    // key a session never has.
    let text_in = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let value = glbench::export::json::parse(&text_in)
        .map_err(|e| format!("parsing {}: {e}", path.display()))?;
    if join::looks_like_join(&value) {
        return inspect_join(path);
    }

    let (session, report) = archive::read_verified(path, verify)?;
    for finding in &report.findings {
        eprintln!("glbench: [{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
    }
    print!("{}", text::session(&session));
    if !report.passed() {
        return Err("archive failed integrity verification".into());
    }
    Ok(())
}

/// `glbench inspect <join.json>` — re-verify the sources a join recorded.
fn inspect_join(path: &Path) -> Result<(), String> {
    let manifest = join::read(path)?;
    println!("join: {}", manifest.label);
    for source in &manifest.sources {
        println!(
            "  {} ({}) digest={}",
            source.path,
            source.label,
            source.digest.as_deref().unwrap_or("none (v1 archive)")
        );
    }
    println!();
    print!("{}", text::comparison(&manifest.comparison));

    let report = join::verify_sources(&manifest);
    for finding in &report.findings {
        eprintln!("glbench: [{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
    }
    if !report.passed() {
        return Err("a join source has changed since the join was written".into());
    }
    Ok(())
}

/// `glbench join` — compare two archives into a third file.
///
/// Neither source is opened for writing. The join records each source's content
/// digest so `glbench inspect` can later tell whether one has moved underneath
/// it; that check is the whole reason the digests are stored.
fn cmd_join(args: &[String]) -> Result<(), String> {
    let mut positional: Vec<&String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut label: Option<String> = None;
    let mut threshold = 0.05;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            "--label" => {
                i += 1;
                label = Some(args.get(i).ok_or("--label needs a value")?.clone());
            }
            "--threshold" => {
                i += 1;
                threshold = parse_f64(args.get(i).ok_or("--threshold needs a value")?, "--threshold")?;
            }
            // Accepted and ignored: a join always verifies its sources, since
            // recording an unverified digest would make the drift check a lie.
            "--no-verify" => {}
            other if other.starts_with("--") => return Err(format!("unknown flag '{other}'")),
            _ => positional.push(&args[i]),
        }
        i += 1;
    }

    if positional.len() != join::V3_SOURCE_COUNT {
        return Err(format!(
            "join takes exactly {} archive paths, got {}. v3 joins two sessions; \
             `sources` is a list so an N-way join stays schema-compatible later, \
             but nothing reads more than two yet",
            join::V3_SOURCE_COUNT,
            positional.len()
        ));
    }
    let out = out.ok_or("join needs --out <join.json>")?;

    let (manifest, report) = join::build(
        Path::new(positional[0]),
        Path::new(positional[1]),
        label.as_deref(),
        threshold,
    )?;
    for finding in &report.findings {
        eprintln!("glbench: [{}] {}: {}", finding.severity.as_str(), finding.check, finding.message);
    }
    if !report.passed() {
        return Err("refusing to write a join over sources that failed verification".into());
    }

    join::write(&manifest, &out)?;
    print!("{}", text::comparison(&manifest.comparison));
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// `glbench accuracy-vs-perf` — join a `run` archive with a `kl-div`/`ppl`
/// archive. See [`glbench::comparison::accuracy`]'s module docs.
fn cmd_accuracy_vs_perf(args: &[String]) -> Result<(), String> {
    let run_path = args.first().ok_or("accuracy-vs-perf needs a run.json path")?;
    let accuracy_path = args.get(1).ok_or("accuracy-vs-perf needs a second, accuracy.json path")?;

    let run = archive::read(Path::new(run_path))?;
    let accuracy_text =
        std::fs::read_to_string(accuracy_path).map_err(|e| format!("reading {accuracy_path}: {e}"))?;
    let accuracy_json =
        glbench::export::json::parse(&accuracy_text).map_err(|e| format!("parsing {accuracy_path}: {e}"))?;

    let joined = glbench::comparison::accuracy::join(&run, &accuracy_json);
    print!("{}", text::accuracy_vs_perf(&joined));
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

/// `glbench quant-info` — static dtype/coverage inspection of a `.gllm`
/// package directory. No inference, no model load.
fn cmd_quant_info(args: &[String]) -> Result<(), String> {
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(args.get(i).ok_or("--model needs a value")?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }
    let model = model.ok_or("quant-info needs --model <package-dir>")?;
    quant_info::run_quant_info(QuantInfoArgs { model, out })
}

/// `glbench ppl` — perplexity of a `.gllm` package on the embedded
/// WikiText-2 sample. Gated behind `gllm-bench`: see [`ppl`]'s module docs
/// for why the number this prints is diagnostic-only today.
#[cfg(feature = "gllm-bench")]
fn cmd_ppl(args: &[String]) -> Result<(), String> {
    let mut a = PplArgs::default();
    let mut model: Option<PathBuf> = None;
    let mut gguf: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(args.get(i).ok_or("--model needs a value")?));
            }
            "--gguf" => {
                i += 1;
                gguf = Some(PathBuf::from(args.get(i).ok_or("--gguf needs a value")?));
            }
            "--context" => {
                i += 1;
                a.context = parse_num(args.get(i).ok_or("--context needs a value")?, "--context")?;
            }
            "--stride" => {
                i += 1;
                a.stride = parse_num(args.get(i).ok_or("--stride needs a value")?, "--stride")?;
            }
            "--out" => {
                i += 1;
                a.out = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }
    a.model = model.ok_or("ppl needs --model <package-dir>")?;
    a.gguf = gguf.ok_or("ppl needs --gguf <original.gguf> (packages don't embed a tokenizer yet)")?;
    ppl::run_ppl(a)
}

/// `glbench kl-div` — per-position KL-divergence between a `.gllm` package
/// and `glproc::runner::Runner` (the oracle), teacher-forced over the
/// embedded WikiText-2 sample. See [`kl_divergence`]'s module docs for the
/// full research vetting behind this command (`RESEARCH_REQUIREMENTS.md`'s
/// 8 mandatory questions) and why it exists alongside `ppl`.
#[cfg(feature = "gllm-bench")]
fn cmd_kl_div(args: &[String]) -> Result<(), String> {
    let mut a = KlDivArgs::default();
    let mut model: Option<PathBuf> = None;
    let mut gguf: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(args.get(i).ok_or("--model needs a value")?));
            }
            "--gguf" => {
                i += 1;
                gguf = Some(PathBuf::from(args.get(i).ok_or("--gguf needs a value")?));
            }
            "--tokens" => {
                i += 1;
                a.tokens = parse_num(args.get(i).ok_or("--tokens needs a value")?, "--tokens")?;
            }
            "--out" => {
                i += 1;
                a.out = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }
    a.model = model.ok_or("kl-div needs --model <package-dir>")?;
    a.gguf = gguf.ok_or("kl-div needs --gguf <original.gguf> (packages don't embed a tokenizer yet)")?;
    kl_divergence::run_kl_divergence(a)
}

/// `glbench tensor-stats` — decode every tensor in a `.gllm` package and
/// flag NaN/Inf/zero-variance. See [`tensor_stats`]'s module docs for the
/// full research vetting.
#[cfg(feature = "gllm-bench")]
fn cmd_tensor_stats(args: &[String]) -> Result<(), String> {
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut full = false;
    let mut norm_only = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(args.get(i).ok_or("--model needs a value")?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).ok_or("--out needs a value")?));
            }
            "--full" => full = true,
            "--norm-only" => norm_only = true,
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }
    let model = model.ok_or("tensor-stats needs --model <package-dir>")?;
    tensor_stats::run_tensor_stats(TensorStatsArgs { model, out, full, norm_only })
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
