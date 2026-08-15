//! CLI entry point (ARTX16 §8's `main.rs`: "CLI parse -> Config -> build pool
//! -> serve", simplified to this crate's scheduler-less v1: CLI parse ->
//! build one backend -> serve).
//!
//! ```text
//! glserve --model /path/to/qwen2-0.5b-hf-dir \
//!         --plugin /path/to/libpjrt_cpu.so \
//!         --port 1136
//!
//! glserve --fake --port 1136   # no model/plugin — FakeBackend, for trying
//!                               # the HTTP API without a checkpoint
//! ```
//!
//! ⛔ `--model` takes a **HuggingFace directory** (`config.json` +
//! `tokenizer.json` + `model.safetensors`), not a `.gllm` package — that is
//! `glictus-caliburni`'s format, a different converter pipeline
//! `gljax::runtime::CachedSession::from_hf_dir` does not read (see
//! `gljax/src/runtime/hf.rs`'s own module docs on exactly this point). No
//! `clap` dependency: the CLI surface is four flags, parsed by hand.

use std::path::PathBuf;
use std::sync::Arc;

use glserve::backend::{FakeBackend, GljaxBackend, InferenceBackend};
use glserve::{build_router, AppState};

/// GwenLand's established convention (legacy `packages/tui` serve command,
/// the Tauri GUI's SSE endpoint, `general.default_port` in the config
/// schema) — inherited, not reinvented (ARTX16's own note).
const DEFAULT_PORT: u16 = 1136;

/// Loopback by default. This server has **no authentication of any kind**, so
/// the bind address is the only thing standing between a loaded model and
/// everyone on the network — binding all interfaces must be an explicit,
/// typed-out decision (`--host 0.0.0.0`), never what you get by not passing a
/// flag.
const DEFAULT_HOST: &str = "127.0.0.1";

struct Args {
    model: Option<PathBuf>,
    plugin: Option<PathBuf>,
    host: String,
    port: u16,
    window: usize,
    fake: bool,
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(std::env::args().skip(1))
}

/// The real parser, taking the argument sequence rather than reading the
/// process environment — so the defaults (notably the loopback bind) are
/// testable without spawning a server.
fn parse_args_from(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut model = None;
    let mut plugin = None;
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;
    let mut window = 1024usize;
    let mut fake = false;

    let mut args = args;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--plugin" => plugin = Some(PathBuf::from(args.next().ok_or("--plugin needs a path")?)),
            "--host" => host = args.next().ok_or("--host needs an address")?,
            "--port" => {
                port = args.next().ok_or("--port needs a number")?.parse().map_err(|_| "--port must be a number".to_string())?
            }
            "--window" => {
                window = args
                    .next()
                    .ok_or("--window needs a number")?
                    .parse()
                    .map_err(|_| "--window must be a number".to_string())?
            }
            "--fake" => fake = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    if !fake && (model.is_none() || plugin.is_none()) {
        return Err("either pass --fake, or both --model <hf_dir> and --plugin <path>".to_string());
    }

    Ok(Args { model, plugin, host, port, window, fake })
}

fn print_help() {
    println!(
        "glserve — OpenAI-compatible HTTP serving for gljax\n\n\
         USAGE:\n    \
         glserve --model <hf_dir> --plugin <path> [--port {DEFAULT_PORT}] [--window 1024]\n    \
         glserve --fake [--port {DEFAULT_PORT}]\n\n\
         FLAGS:\n    \
         --model <dir>     HuggingFace directory (config.json + tokenizer.json + model.safetensors)\n    \
         --plugin <path>   PJRT plugin shared library\n    \
         --host <addr>     listen address (default {DEFAULT_HOST} — loopback only;\n                      \
         pass 0.0.0.0 to expose on the network, which is UNAUTHENTICATED)\n    \
         --port <n>        listen port (default {DEFAULT_PORT})\n    \
         --window <n>      compiled sequence window (default 1024)\n    \
         --fake            serve FakeBackend instead of a real model — no --model/--plugin needed\n"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger_init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n");
            print_help();
            std::process::exit(2);
        }
    };

    let backend: Arc<dyn InferenceBackend> = if args.fake {
        log::warn!("--fake: serving FakeBackend, no model loaded");
        Arc::new(FakeBackend::new("fake-model"))
    } else {
        let model_dir = args.model.expect("checked in parse_args");
        let plugin_path = args.plugin.expect("checked in parse_args");
        let model_id = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string());
        log::info!("loading {} via {}", model_dir.display(), plugin_path.display());
        Arc::new(GljaxBackend::spawn(model_id, plugin_path, model_dir, args.window)?)
    };

    let state = AppState { backend };
    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    if args.host != DEFAULT_HOST {
        log::warn!(
            "binding {} — glserve has no authentication, so this exposes the loaded model to \
             every host that can reach this address",
            args.host
        );
    }
    log::info!("glserve listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Minimal `RUST_LOG` handling, matching gljax's own examples
/// (`bench_kv_cache.rs`) rather than adding an `env_logger` dependency for
/// four log levels.
fn env_logger_init() {
    struct Stderr(log::LevelFilter);
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= self.0
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{}] {}", r.level(), r.args());
            }
        }
        fn flush(&self) {}
    }
    let level = match std::env::var("RUST_LOG").as_deref() {
        Ok("trace") => log::LevelFilter::Trace,
        Ok("debug") => log::LevelFilter::Debug,
        Ok("warn") => log::LevelFilter::Warn,
        Ok("error") => log::LevelFilter::Error,
        Ok("off") => log::LevelFilter::Off,
        _ => log::LevelFilter::Info,
    };
    let logger: &'static Stderr = Box::leak(Box::new(Stderr(level)));
    let _ = log::set_logger(logger).map(|()| log::set_max_level(level));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args_from(args.iter().map(|s| s.to_string()))
    }

    /// The security-relevant default: with no `--host`, glserve must bind
    /// loopback. This server has no authentication, so a default of
    /// `0.0.0.0` would expose a loaded model to the whole network the moment
    /// someone runs it.
    #[test]
    fn default_bind_is_loopback_not_all_interfaces() {
        let args = parse(&["--fake"]).expect("--fake alone is valid");
        assert_eq!(args.host, "127.0.0.1");
        assert_ne!(args.host, "0.0.0.0");
        assert_eq!(DEFAULT_HOST, "127.0.0.1");
    }

    /// Exposing the server stays possible — it just has to be typed out.
    #[test]
    fn host_flag_overrides_the_default() {
        let args = parse(&["--fake", "--host", "0.0.0.0"]).expect("explicit host is valid");
        assert_eq!(args.host, "0.0.0.0");
    }

    #[test]
    fn port_defaults_and_overrides() {
        assert_eq!(parse(&["--fake"]).unwrap().port, DEFAULT_PORT);
        assert_eq!(parse(&["--fake", "--port", "9999"]).unwrap().port, 9999);
    }

    /// Error case: a real model needs both `--model` and `--plugin`; half a
    /// pair must be refused rather than silently serving nothing.
    #[test]
    fn model_without_plugin_is_rejected() {
        assert!(parse(&["--model", "/tmp/m"]).is_err());
        assert!(parse(&["--plugin", "/tmp/p"]).is_err());
    }

    #[test]
    fn unknown_flag_is_rejected() {
        assert!(parse(&["--fake", "--nonsense"]).is_err());
        assert!(parse(&["--port"]).is_err(), "--port with no value must fail");
    }
}
