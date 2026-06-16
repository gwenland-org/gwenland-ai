//! `gwen update` — self-update the GwenLand binary.
//!
//! # Why this module exists
//!
//! GwenLand ships as a single self-contained binary.  Asking users to re-run
//! `cargo install` or visit GitHub releases every time defeats the "zero-ops"
//! design goal.  This command checks the latest GitHub release, asks for
//! confirmation, downloads the platform-correct asset, and replaces the
//! running binary atomically.
//!
//! # Update strategy
//!
//! ## Unix (Linux / macOS)
//! 1. Download new binary to `<current_binary_path>.new.tmp`.
//! 2. `chmod +x` the temp file.
//! 3. `std::fs::rename(tmp, current_path)` — atomic on POSIX; the kernel swaps
//!    the inode so the currently-running process is unaffected.
//!
//! ## Windows
//! Windows locks the EXE of a running process; you cannot delete or overwrite
//! it while it is executing.  The workaround:
//! 1. Rename the running EXE to `<name>.exe.old` (allowed because Windows only
//!    locks the inode, not the directory entry, as of Vista+).
//! 2. Write the new EXE to the original path.
//! 3. Record the `.old` path in `~/.gwenland/update.old` so the next
//!    invocation can clean it up.
//!
//! # Asset naming convention
//!
//! Release assets follow: `gwenland-{os}-{arch}[.tar.gz|.zip]`
//!   - `gwenland-linux-x86_64.tar.gz`
//!   - `gwenland-linux-aarch64.tar.gz`
//!   - `gwenland-macos-x86_64.tar.gz`
//!   - `gwenland-macos-aarch64.tar.gz`
//!   - `gwenland-windows-x86_64.zip`
//!
//! OS names come from `std::env::consts::OS` (`linux`, `macos`, `windows`).
//! Arch names come from `std::env::consts::ARCH` (`x86_64`, `aarch64`).
//!
//! # Why `reqwest::blocking` and not async
//!
//! `run_update_cmd` is called from a `tokio::block_on` context.  Using the
//! blocking client inside a `std::thread::spawn` avoids nesting runtimes while
//! keeping the download loop readable as straight-line imperative code.  The
//! async stream API would require `futures_util::StreamExt` and a pin-project
//! dance for no ergonomic benefit here.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

// ── colour constants ──────────────────────────────────────────────────────────

const ORANGE: &str = "\x1b[38;2;255;140;66m";
const GREEN:  &str = "\x1b[32m";
const RESET:  &str = "\x1b[0m";

// ── GitHub API types ──────────────────────────────────────────────────────────

/// Minimal subset of the GitHub Releases API response.
///
/// We only deserialize fields we actually use; unknown fields are ignored
/// by serde so future API additions won't break the client.
#[derive(Debug, Deserialize)]
struct GhRelease {
    /// The git tag for this release (e.g. `"v0.9.2"`).
    tag_name: String,
    /// List of downloadable assets attached to the release.
    assets: Vec<GhAsset>,
}

/// One downloadable file attached to a GitHub release.
#[derive(Debug, Deserialize)]
struct GhAsset {
    /// Filename as it appears on the release page (e.g. `"gwenland-linux-x86_64.tar.gz"`).
    name: String,
    /// Direct download URL (the `browser_download_url` field in the API).
    #[serde(rename = "browser_download_url")]
    download_url: String,
}

// ── version helpers ───────────────────────────────────────────────────────────

/// Strip a leading `v` from a version tag so `"v0.9.2"` == `"0.9.2"`.
///
/// GitHub conventionally tags releases as `v{semver}`, while `CARGO_PKG_VERSION`
/// is plain `{semver}`.  Normalising both sides prevents a false "update
/// available" report on every run.
fn strip_v(s: &str) -> &str {
    s.strip_prefix('v').unwrap_or(s)
}

// ── asset name helpers ────────────────────────────────────────────────────────

/// Build the expected asset filename for the current platform.
///
/// The archive extension is `.zip` on Windows (because `tar` is not universally
/// available there), `.tar.gz` everywhere else.
fn expected_asset_name() -> String {
    let os   = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    // Map Rust's OS constant to the convention used in release asset names.
    // `macos` stays `macos`; `linux` stays `linux`; `windows` → `windows`.
    let ext = if os == "windows" { "zip" } else { "tar.gz" };

    format!("gwenland-{}-{}.{}", os, arch, ext)
}

// ── current binary path ───────────────────────────────────────────────────────

/// Resolve the absolute path of the currently-running binary.
///
/// `std::env::current_exe()` returns the canonical path after following
/// symlinks, which is what we need for the replace-in-place strategy.
fn current_exe_path() -> Result<PathBuf> {
    std::env::current_exe()
        .context("cannot determine path of current executable")
}

// ── cleanup .old files left by previous Windows update ───────────────────────

/// On Windows, the previous binary is renamed to `.exe.old` during an update
/// because the OS holds a lock on the running EXE.  This function removes any
/// `.exe.old` left behind from the last successful update.
///
/// We call this at the start of every `gwen update` run so the cleanup happens
/// without needing a separate command.  Failure is non-fatal: the `.old` file
/// is inert and takes no disk space beyond one binary-sized copy.
fn cleanup_windows_old_binary() {
    let marker_path = gwenland_core::storage::paths::GwenPaths::root_dir().join("update.old");

    if let Ok(old_path_str) = std::fs::read_to_string(&marker_path) {
        let old_path = PathBuf::from(old_path_str.trim());
        if old_path.exists() {
            // Best-effort removal; ignore errors (file might still be locked
            // if the user opened two terminals).
            let _ = std::fs::remove_file(&old_path);
        }
    }
    // Remove the marker regardless of whether deletion succeeded.
    let _ = std::fs::remove_file(&marker_path);
}

// ── download with progress ────────────────────────────────────────────────────

/// Download `url` into `dest` and print a simple KB counter.
///
/// Why a manual byte loop rather than `reqwest::get().bytes()`:
/// `bytes()` buffers the entire response before returning, which on a 10 MB
/// binary would hold 10 MB in RAM with no progress feedback.  Reading in
/// chunks lets us print a live counter without an async stream.
///
/// The "total" may be unknown if the server omits `Content-Length`; in that
/// case we show `{kb} KB` without a denominator.
fn download_with_progress(url: &str, dest: &mut dyn Write) -> Result<()> {
    let mut response = reqwest::blocking::Client::builder()
        // GitHub requires a User-Agent; an empty UA returns 403.
        .user_agent(format!("gwenland/{}", env!("GWEN_VERSION")))
        .build()
        .context("failed to build reqwest client")?
        .get(url)
        .send()
        .with_context(|| format!("failed to download {}", url))?;

    if !response.status().is_success() {
        bail!("download failed: HTTP {}", response.status());
    }

    let total_bytes = response.content_length();

    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 65_536]; // 64 KiB read buffer

    loop {
        use std::io::Read;
        let n = response
            .read(&mut buf)
            .context("error reading download stream")?;
        if n == 0 {
            break;
        }
        dest.write_all(&buf[..n])
            .context("error writing download to disk")?;
        downloaded += n as u64;

        // Print progress on the same line by returning carriage position.
        match total_bytes {
            Some(total) => print!(
                "\r  Downloading... {}/{} KB",
                downloaded / 1024,
                total / 1024
            ),
            None => print!("\r  Downloading... {} KB", downloaded / 1024),
        }
        // Flush stdout immediately so the counter updates mid-line rather
        // than buffering until the download finishes.
        let _ = std::io::stdout().flush();
    }

    // Newline after the progress line so subsequent output appears on a
    // fresh line rather than overwriting the counter.
    println!();
    Ok(())
}

// ── archive extraction ────────────────────────────────────────────────────────

/// Extract the binary from a `.tar.gz` archive (Unix) or `.zip` (Windows).
///
/// We extract only the first file whose name matches `gwenland` (or
/// `gwenland.exe` on Windows) to avoid accidentally overwriting other files
/// if the asset contains extras (README, licence, etc.).
///
/// Why temp file → rename rather than writing directly:
/// If extraction fails mid-way, a partial write would corrupt the running
/// binary.  Writing to a temp file and renaming only on success is an
/// atomic-commit pattern that guarantees the binary is either untouched or
/// fully replaced.
fn extract_binary(archive_path: &std::path::Path, _os: &str) -> Result<Vec<u8>> {
    let ext = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    if ext.ends_with(".tar.gz") || ext.ends_with(".tgz") {
        extract_from_tar_gz(archive_path)
    } else if ext.ends_with(".zip") {
        extract_from_zip(archive_path)
    } else {
        bail!("unsupported archive format: {}", archive_path.display())
    }
}

/// Read the binary bytes from a `.tar.gz` archive.
fn extract_from_tar_gz(archive_path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;

    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("cannot open archive {}", archive_path.display()))?;

    // flate2 is a transitive dep of reqwest/tokio — no new dep needed.
    // If it isn't available at link time this will produce a clear compile
    // error; we handle that case with a build note in the PR.
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);

    for entry in tar.entries().context("cannot read tar entries")? {
        let mut entry = entry.context("corrupt tar entry")?;
        let path = entry.path().context("cannot read entry path")?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Accept `gwenland` (Unix) or `gwenland.exe` (Windows cross-build).
        if name == "gwenland" || name == "gwenland.exe" {
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).context("cannot read binary from tar")?;
            return Ok(bytes);
        }
    }

    bail!("no `gwenland` binary found in archive")
}

/// Read the binary bytes from a `.zip` archive (Windows releases).
fn extract_from_zip(archive_path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;

    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("cannot open zip {}", archive_path.display()))?;

    let mut zip = zip::ZipArchive::new(file).context("cannot open zip archive")?;

    // zip crate uses indexed access; iterate to find the binary entry.
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).context("cannot read zip entry")?;
        let name = entry.name().to_string();

        if name == "gwenland.exe" || name == "gwenland" {
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).context("cannot read binary from zip")?;
            return Ok(bytes);
        }
    }

    bail!("no `gwenland` binary found in zip")
}

// ── binary replacement ────────────────────────────────────────────────────────

/// Replace the running binary with new bytes.
///
/// The strategy differs by platform — see the module-level doc for rationale.
fn replace_binary(new_bytes: &[u8]) -> Result<()> {
    let current = current_exe_path()?;

    #[cfg(windows)]
    {
        replace_binary_windows(&current, new_bytes)?;
    }

    #[cfg(not(windows))]
    {
        replace_binary_unix(&current, new_bytes)?;
    }

    Ok(())
}

/// Unix atomic replacement via temp-file + rename.
///
/// `rename()` on the same filesystem is atomic at the kernel level: observers
/// see either the old binary or the new one, never a partial file.  We write
/// to a `.new.tmp` suffix rather than the target directly so that a crash
/// during write leaves the original intact.
#[cfg(not(windows))]
fn replace_binary_unix(current: &std::path::Path, new_bytes: &[u8]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    // Write to a temp file in the same directory so the rename stays on the
    // same filesystem (cross-filesystem renames are not atomic on Linux).
    let tmp_path = current.with_extension("new.tmp");

    std::fs::write(&tmp_path, new_bytes)
        .with_context(|| format!("cannot write new binary to {}", tmp_path.display()))?;

    // Mark executable — required on Unix; without this the new binary would
    // fail with "Permission denied" on the next invocation.
    std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o755))
        .context("cannot chmod new binary")?;

    // Atomic rename: this replaces `current` even if it is currently running.
    // The running process holds an open file descriptor to the old inode; the
    // kernel keeps the inode alive until that fd is closed, so the swap is safe.
    std::fs::rename(&tmp_path, current)
        .with_context(|| format!("cannot rename {} → {}", tmp_path.display(), current.display()))?;

    Ok(())
}

/// Windows binary replacement via rename-old + write-new.
///
/// Windows holds a share lock on the executing EXE's inode but does NOT
/// lock the directory entry, so we can rename the running EXE to a `.old`
/// path.  After the rename, we write the new bytes to the original path.
/// The `.old` path is recorded in `~/.gwenland/update.old` so the next
/// run of `gwen update` can delete it.
#[cfg(windows)]
fn replace_binary_windows(current: &std::path::Path, new_bytes: &[u8]) -> Result<()> {
    let old_path = current.with_extension("exe.old");

    // Step 1: rename the running EXE out of the way.
    // This succeeds because Windows only lock-prevents deletion of the inode,
    // not renaming the directory entry.  If this fails (e.g. permissions),
    // we surface the error immediately rather than touching the binary.
    std::fs::rename(current, &old_path)
        .with_context(|| {
            format!(
                "cannot rename {} → {}; do you have write access to the binary directory?",
                current.display(),
                old_path.display()
            )
        })?;

    // Step 2: write the new binary to the original path.
    // If this fails we attempt to roll back by renaming `.old` back.  The
    // rollback may fail too (e.g. out of disk), so we still return an error.
    if let Err(write_err) = std::fs::write(current, new_bytes) {
        // Best-effort rollback — don't mask the original error.
        let _ = std::fs::rename(&old_path, current);
        return Err(write_err).with_context(|| {
            format!("cannot write new binary to {}", current.display())
        });
    }

    // Step 3: record the `.old` path so the next run can clean it up.
    // We use a marker file rather than a registry key so the cleanup path
    // works the same way on all Windows editions.
    let marker_dir = gwenland_core::storage::paths::GwenPaths::root_dir();
    let _ = std::fs::create_dir_all(&marker_dir);
    let marker = marker_dir.join("update.old");
    let _ = std::fs::write(&marker, old_path.to_string_lossy().as_bytes());

    Ok(())
}

// ── confirmation prompt ───────────────────────────────────────────────────────

/// Read a single line from stdin and return true if it starts with 'y' or 'Y'.
///
/// Default is NO — an empty Enter press or 'n' will abort the update.
/// This matches the standard `[y/N]` convention where the capital letter
/// is the default.
fn confirm_prompt(current_ver: &str, latest_ver: &str) -> bool {
    println!(
        "\n  Current version : {}{}{}",
        ORANGE, current_ver, RESET
    );
    println!(
        "  Latest version  : {}{}{}\n",
        GREEN, latest_ver, RESET
    );
    print!("  Update now? [y/N] ");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

// ── command entry point ───────────────────────────────────────────────────────

/// Entry point called from `main.rs`.
///
/// Exits 1 on any unrecoverable error.  Runs synchronously in a `std::thread`
/// to keep the blocking HTTP calls off the tokio runtime thread pool.
pub fn run_update_cmd() {
    // Clean up any `.exe.old` left over from the previous Windows update
    // before doing anything else.  Non-fatal; errors are silently swallowed.
    #[cfg(windows)]
    cleanup_windows_old_binary();

    let result = std::thread::spawn(run_update_inner)
        .join()
        .unwrap_or_else(|_| Err(anyhow::anyhow!("update thread panicked")));

    if let Err(e) = result {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

/// Core update logic, run on a dedicated thread to keep blocking I/O off tokio.
fn run_update_inner() -> Result<()> {
    let current_ver = env!("GWEN_VERSION");
    println!(
        "{}GwenLand updater{}  (current: v{})",
        ORANGE, RESET, current_ver
    );
    println!("  Checking GitHub for the latest release…");

    // ── 1. fetch latest release metadata ─────────────────────────────────────

    let client = reqwest::blocking::Client::builder()
        // GitHub blocks requests without a User-Agent.
        .user_agent(format!("gwenland/{}", current_ver))
        .build()
        .context("failed to build HTTP client")?;

    let release: GhRelease = client
        .get("https://api.github.com/repos/jinxsuper/gwenland/releases/latest")
        .send()
        .context("failed to reach GitHub API — check your network connection")?
        .json()
        .context("GitHub API response was not valid JSON — the repo may have no releases yet")?;

    let latest_ver = strip_v(&release.tag_name);

    // ── 2. version comparison ─────────────────────────────────────────────────

    if strip_v(current_ver) == latest_ver {
        println!(
            "\n  {}✓ GwenLand is up to date (v{}){}",
            GREEN, current_ver, RESET
        );
        return Ok(());
    }

    // ── 3. find the matching asset ────────────────────────────────────────────

    let asset_name = expected_asset_name();
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .with_context(|| {
            format!(
                "no release asset named '{}' found for v{}\n\
                 Available assets: {}",
                asset_name,
                latest_ver,
                release
                    .assets
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    // ── 4. confirm with user ──────────────────────────────────────────────────

    if !confirm_prompt(current_ver, latest_ver) {
        println!("  Update cancelled.");
        return Ok(());
    }

    println!("\n  Downloading {}…", asset.name);

    // ── 5. download archive to a temp file ────────────────────────────────────
    //
    // We download to a temp file rather than memory because release binaries
    // can be 5–20 MB; holding that in RAM while also running the TUI could
    // cause OOM on memory-constrained systems.

    let tmp_dir = gwenland_core::storage::paths::GwenPaths::tmp_dir();
    std::fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("cannot create temp dir {}", tmp_dir.display()))?;

    let archive_path = tmp_dir.join(&asset.name);
    {
        let mut archive_file = std::fs::File::create(&archive_path)
            .with_context(|| format!("cannot create temp file {}", archive_path.display()))?;
        download_with_progress(&asset.download_url, &mut archive_file)?;
    }

    println!("  Extracting {}…", asset.name);

    // ── 6. extract binary from archive ───────────────────────────────────────

    let new_bytes = extract_binary(&archive_path, std::env::consts::OS)
        .with_context(|| format!("failed to extract binary from {}", archive_path.display()))?;

    // Clean up the archive immediately — we have the bytes in memory now.
    let _ = std::fs::remove_file(&archive_path);

    println!("  Installing…");

    // ── 7. replace running binary ─────────────────────────────────────────────

    replace_binary(&new_bytes)?;

    println!(
        "\n  {}✓ Updated to v{}.{}  Restart gwen to use the new version.",
        GREEN, latest_ver, RESET
    );

    Ok(())
}
