//! `gwen config` — read and write user configuration.
//!
//! # Why this module exists
//!
//! GwenLand exposes user-facing settings (theme, default_model, etc.) through
//! a JSON file at `~/.gwenland/config/config.json`.  This module owns:
//!   - loading that file (creating it with defaults when missing)
//!   - typed get/set via a flat key namespace (e.g. `"theme"`)
//!   - pretty-printed JSON I/O so the file stays human-editable
//!   - ANSI colour output that matches GwenLand's brand orange
//!
//! # Why merge-on-save
//!
//! The Rust core also stores nested engine sections in this JSON file.  This
//! command owns only the flat user-facing keys, so writes merge those keys into
//! the existing object instead of replacing unrelated sections.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ── colour constant ───────────────────────────────────────────────────────────

/// Gwen brand orange as an ANSI SGR escape sequence.
/// Used for key names in `list` output to match the TUI colour scheme.
const ORANGE: &str = "\x1b[38;2;255;140;66m";
const RESET: &str = "\x1b[0m";

// ── config struct ─────────────────────────────────────────────────────────────

/// The user-facing configuration persisted at `~/.gwenland/config/config.json`.
///
/// Every field is `Option` or has a default so that missing keys in an
/// existing file are filled in rather than causing a parse error — important
/// for forward-compatibility when new fields are added in future releases.
///
/// # Field semantics
///
/// - `theme`           — TUI colour theme name (default `"gwen-noir"`)
/// - `default_model`   — HF model ID used when no `-m` flag is provided
/// - `blocks_dir`      — Override path for the user's blocks/plugins directory
/// - `package_manager` — Package manager hint for doctor/setup (`"auto"` | `"pip"` | `"uv"`)
/// - `telemetry`       — Opt-in anonymous usage statistics (default `false`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    #[serde(default = "default_theme")]
    pub theme: String,

    #[serde(default)]
    pub default_model: Option<String>,

    #[serde(default)]
    pub blocks_dir: Option<String>,

    #[serde(default = "default_package_manager")]
    pub package_manager: String,

    #[serde(default)]
    pub telemetry: bool,
}

fn default_theme() -> String {
    "gwen-noir".to_string()
}

fn default_package_manager() -> String {
    "auto".to_string()
}

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            theme:           default_theme(),
            default_model:   None,
            blocks_dir:      None,
            package_manager: default_package_manager(),
            telemetry:       false,
        }
    }
}

// ── path resolution ───────────────────────────────────────────────────────────

/// Resolve the shared GwenLand JSON config path.
fn config_path() -> std::path::PathBuf {
    gwenland_core::storage::paths::GwenPaths::config_file()
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

/// Load `UserConfig` from disk.
///
/// If the file is missing, writes the default and returns it — this means
/// `gwen config get` on a fresh install never errors with "file not found".
///
/// If the file exists but is malformed JSON, returns an error rather than
/// silently overwriting user edits.
fn load_or_create() -> Result<UserConfig> {
    let path = config_path();

    if !path.exists() {
        let cfg = UserConfig::default();
        save(&cfg)?;
        return Ok(cfg);
    }

    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read config file at {}", path.display()))?;

    serde_json::from_str::<UserConfig>(&raw)
        .with_context(|| format!("config file at {} contains invalid JSON", path.display()))
}

/// Persist `UserConfig` to disk with 2-space pretty-printing.
///
/// Why pretty-print: the file is designed to be hand-edited by users.
/// Compact JSON would make diffs illegible and discourage manual edits.
fn save(cfg: &UserConfig) -> Result<()> {
    let path = config_path();

    // Create parent directory if it doesn't exist yet (fresh install).
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create config directory at {}", parent.display()))?;
    }

    let mut root = read_json_object(&path);
    let user = serde_json::to_value(cfg).context("cannot serialise UserConfig")?;
    if let Value::Object(user_map) = user {
        for (key, value) in user_map {
            root.insert(key, value);
        }
    }

    let json = serde_json::to_string_pretty(&Value::Object(root))
        .context("cannot serialise config JSON")?;
    std::fs::write(&path, json)
        .with_context(|| format!("cannot write config file at {}", path.display()))?;
    Ok(())
}

fn read_json_object(path: &std::path::Path) -> Map<String, Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| match value {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default()
}

// ── key helpers ───────────────────────────────────────────────────────────────

/// Read a config value by flat key name.
///
/// Returns the value as a String so the caller can print it without
/// caring about the underlying type.
fn get_key(cfg: &UserConfig, key: &str) -> Result<String> {
    match key {
        "theme"           => Ok(cfg.theme.clone()),
        "default_model"   => Ok(cfg.default_model.clone().unwrap_or_else(|| "null".to_string())),
        "blocks_dir"      => Ok(cfg.blocks_dir.clone().unwrap_or_else(|| "null".to_string())),
        "package_manager" => Ok(cfg.package_manager.clone()),
        "telemetry"       => Ok(cfg.telemetry.to_string()),
        _ => bail!(
            "unknown config key '{}'\n\
             valid keys: theme, default_model, blocks_dir, package_manager, telemetry",
            key
        ),
    }
}

/// Write a config value by flat key name, parsing the string to the correct type.
///
/// Bool fields accept "true"/"false" (case-insensitive to be user-friendly).
/// Optional string fields accept "null" / "" to clear the value.
fn set_key(cfg: &mut UserConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "theme" => {
            cfg.theme = value.to_string();
        }
        "default_model" => {
            // Allow "null" or empty string to clear the field.
            cfg.default_model = if value.is_empty() || value.eq_ignore_ascii_case("null") {
                None
            } else {
                Some(value.to_string())
            };
        }
        "blocks_dir" => {
            cfg.blocks_dir = if value.is_empty() || value.eq_ignore_ascii_case("null") {
                None
            } else {
                Some(value.to_string())
            };
        }
        "package_manager" => {
            // Validate against known values so a typo doesn't silently break
            // the doctor/setup commands that read this field.
            match value {
                "auto" | "pip" | "uv" | "conda" => cfg.package_manager = value.to_string(),
                _ => bail!(
                    "invalid package_manager '{}'; valid values: auto, pip, uv, conda",
                    value
                ),
            }
        }
        "telemetry" => {
            cfg.telemetry = value
                .to_ascii_lowercase()
                .parse::<bool>()
                .with_context(|| {
                    format!("telemetry must be 'true' or 'false', got '{}'", value)
                })?;
        }
        _ => bail!(
            "unknown config key '{}'\n\
             valid keys: theme, default_model, blocks_dir, package_manager, telemetry",
            key
        ),
    }
    Ok(())
}

// ── Clap args ─────────────────────────────────────────────────────────────────

/// Top-level args for `gwen config`.
///
/// Why `Args` not `Parser`: this struct is registered as a subcommand in
/// `main.rs`'s top-level `Commands` enum.  Using `Args` keeps it consistent
/// with every other command in the codebase.
#[derive(Args, Debug)]
#[command(
    about = "Manage GwenLand user configuration",
    long_about = "Read and write GwenLand user settings stored at\n\
                  ~/.gwenland/config/config.json.\n\n\
                  Subcommands:\n  \
                    gwen config get <key>        Print value of a config key\n  \
                    gwen config set <key> <val>  Update a config key\n  \
                    gwen config list             Pretty-print all config as JSON\n  \
                    gwen config reset            Reset config to defaults\n\n\
                  Keys: theme, default_model, blocks_dir, package_manager, telemetry"
)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub subcommand: ConfigSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ConfigSubcommand {
    /// Print the value of a single config key
    Get {
        /// The config key to read (e.g. "theme")
        key: String,
    },
    /// Set a config key to a new value
    Set {
        /// The config key to update (e.g. "theme")
        key: String,
        /// The new value (e.g. "gwen-light")
        value: String,
    },
    /// Pretty-print all config keys and their current values
    List,
    /// Reset all config to factory defaults
    Reset,
}

// ── command entry point ───────────────────────────────────────────────────────

/// Entry point called from `main.rs`.
///
/// Exits with code 1 on any error so callers and scripts can detect failure
/// without parsing stderr.
pub fn run_config_cmd(args: ConfigArgs) {
    if let Err(e) = run_config_inner(args) {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

fn run_config_inner(args: ConfigArgs) -> Result<()> {
    match args.subcommand {
        ConfigSubcommand::Get { key } => {
            let cfg = load_or_create()?;
            let value = get_key(&cfg, &key)?;
            // Print key in brand orange, value in plain white — matches TUI style.
            println!("{}{}{} = {}", ORANGE, key, RESET, value);
        }

        ConfigSubcommand::Set { key, value } => {
            let mut cfg = load_or_create()?;
            set_key(&mut cfg, &key, &value)?;
            save(&cfg)?;
            println!("{}{}{}  ←  {}", ORANGE, key, RESET, value);
        }

        ConfigSubcommand::List => {
            let cfg = load_or_create()?;
            print_list(&cfg);
        }

        ConfigSubcommand::Reset => {
            let cfg = UserConfig::default();
            save(&cfg)?;
            println!("Config reset to defaults.");
            print_list(&cfg);
        }
    }
    Ok(())
}

/// Pretty-print all config keys with brand orange key names.
///
/// We hand-roll this rather than calling `serde_json::to_string_pretty`
/// directly on the struct so we can apply per-key ANSI colouring.
fn print_list(cfg: &UserConfig) {
    let path = config_path().display().to_string();

    println!("{}GwenLand config{} — {}", ORANGE, RESET, path);
    println!();

    let fields: &[(&str, String)] = &[
        ("theme",           cfg.theme.clone()),
        ("default_model",   cfg.default_model.as_deref().unwrap_or("null").to_string()),
        ("blocks_dir",      cfg.blocks_dir.as_deref().unwrap_or("null").to_string()),
        ("package_manager", cfg.package_manager.clone()),
        ("telemetry",       cfg.telemetry.to_string()),
    ];

    // Align values by the longest key name for readability.
    let max_key_len = fields.iter().map(|(k, _)| k.len()).max().unwrap_or(0);

    for (key, value) in fields {
        println!(
            "  {}{:<width$}{}  {}",
            ORANGE,
            key,
            RESET,
            value,
            width = max_key_len,
        );
    }
    println!();
}
