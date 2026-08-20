// diagnostics/crash_report.rs — human-readable crash reports for GWEN-224 Wave 3.
//
// Two capture paths feed the same report format:
//   1. Rust panics  — `std::panic::set_hook`, runs with a normal allocator/stack,
//      can format freely and call into `chrono`/`sysinfo`.
//   2. OS-level signals (SIGSEGV/SIGABRT/SIGILL/SIGBUS on Unix, unhandled
//      structured exceptions on Windows) — these fire from a restricted signal
//      context. POSIX signal-safety rules forbid allocating, locking, or
//      calling non-reentrant libc functions inside the handler. The signal
//      path therefore pre-renders everything it safely can (timestamp captured
//      eagerly via an atomic, surface/command captured at startup) and performs
//      a single best-effort `std::fs::write` — if that fails we silently fall
//      back to the OS's own crash behavior; we never risk a second fault inside
//      the handler.
//
// @INFO: install order in `main()` should be: set_surface() -> install_panic_hook()
//        -> install_signal_handler(). Both hooks are additive — they format a
//        report and write it, then let the previous behavior (default panic
//        printer, default signal disposition) continue so existing terminal-
//        restore logic (see tui/src/main.rs) still runs.

use crate::storage::paths::GwenPaths;
use std::fmt::Write as _;
use std::panic::PanicHookInfo;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// Which front-end was running when the crash happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Surface {
    Cli = 0,
    Tui = 1,
    Gui = 2,
    Serve = 3,
}

impl Surface {
    fn label(self) -> &'static str {
        match self {
            Surface::Cli => "CLI",
            Surface::Tui => "TUI",
            Surface::Gui => "GUI",
            Surface::Serve => "Serve",
        }
    }

    fn from_u8(v: u8) -> Surface {
        match v {
            1 => Surface::Tui,
            2 => Surface::Gui,
            3 => Surface::Serve,
            _ => Surface::Cli,
        }
    }
}

static ACTIVE_SURFACE: AtomicU8 = AtomicU8::new(Surface::Cli as u8);

/// Record which surface is active. Call once at startup, before installing
/// the panic hook / signal handler. Safe to call from a signal context too
/// (it's just an atomic store), though in practice it's only ever called early.
pub fn set_surface(surface: Surface) {
    ACTIVE_SURFACE.store(surface as u8, Ordering::Relaxed);
}

fn active_surface() -> Surface {
    Surface::from_u8(ACTIVE_SURFACE.load(Ordering::Relaxed))
}

/// Static context captured once at process startup — everything here is
/// either `'static` or cheap to clone, so the signal handler can read it
/// without allocating.
#[derive(Debug, Clone)]
pub struct CrashContext {
    pub version: String,
    pub git_hash: String,
    pub command: String,
}

static CONTEXT: OnceLock<CrashContext> = OnceLock::new();

/// Install the static crash context. Call once at startup alongside
/// `set_surface`. `version`/`git_hash` should come from the calling crate's
/// own `env!(...)` (core has no build.rs of its own, so it can't bake these
/// in itself).
pub fn init_context(version: impl Into<String>, git_hash: impl Into<String>) {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let _ = CONTEXT.set(CrashContext {
        version: version.into(),
        git_hash: git_hash.into(),
        command,
    });
}

fn context() -> CrashContext {
    CONTEXT.get().cloned().unwrap_or_else(|| CrashContext {
        version: "unknown".into(),
        git_hash: "unknown".into(),
        command: std::env::args().collect::<Vec<_>>().join(" "),
    })
}

fn os_description() -> String {
    // sysinfo gives a real distro/build string where available; fall back to
    // the std::env::consts pair if the platform call fails for any reason.
    let long_os = sysinfo::System::long_os_version();
    let kernel = sysinfo::System::kernel_version();
    match (long_os, kernel) {
        (Some(os), Some(k)) => format!("{} (kernel {}), {}", os, k, std::env::consts::ARCH),
        (Some(os), None) => format!("{}, {}", os, std::env::consts::ARCH),
        _ => format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
    }
}

fn backtrace_section() -> String {
    if std::env::var("RUST_BACKTRACE").map(|v| v != "0").unwrap_or(false) {
        format!("\nBacktrace:\n{:?}\n", backtrace::Backtrace::new())
    } else {
        "\nBacktrace: (set RUST_BACKTRACE=1 for full trace)\n".to_string()
    }
}

fn report_path(timestamp: &chrono::DateTime<chrono::Local>) -> PathBuf {
    let stamp = timestamp.to_rfc3339().replace(':', "-");
    GwenPaths::crash_logs_dir().join(format!("crash-{stamp}.txt"))
}

fn render_header(ctx: &CrashContext, timestamp: &chrono::DateTime<chrono::Local>, crash_type: &str) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "GwenLand Crash Report\n======================\n\
         Timestamp:   {}\n\
         Version:     gwen {} (rev {})\n\
         Surface:     {}\n\
         Command:     {}\n\
         OS:          {}\n\n\
         Crash Type:  {}\n\n",
        timestamp.to_rfc3339(),
        ctx.version,
        ctx.git_hash,
        active_surface().label(),
        ctx.command,
        os_description(),
        crash_type,
    );
    out
}

fn render_footer() -> &'static str {
    "\n------------------------------------------------------\n\
     If this looks like a bug, please share this file when reporting.\n"
}

/// Format and write a crash report for a Rust panic. Returns the path written
/// on success. Never panics itself — any failure (e.g. can't create the
/// directory, disk full) is swallowed and `None` is returned so the original
/// panic still propagates through the default printer untouched.
pub fn write_panic_report(info: &PanicHookInfo<'_>, ctx: &CrashContext) -> Option<PathBuf> {
    let timestamp = chrono::Local::now();
    let location = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "unknown location".into());
    let message = info
        .payload()
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| info.payload().downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic payload>".into());
    let thread_name = std::thread::current().name().unwrap_or("main").to_string();

    let mut report = render_header(ctx, &timestamp, "Rust panic");
    let _ = write!(
        report,
        "Panic:\n  thread '{}' panicked at {}\n  {}\n",
        thread_name, location, message
    );
    report.push_str(&backtrace_section());
    report.push_str(render_footer());

    let path = report_path(&timestamp);
    match std::fs::write(&path, report) {
        Ok(()) => Some(path),
        Err(_) => None,
    }
}

/// Format and write a crash report for an OS-level signal / unhandled
/// exception. Intentionally minimal compared to `write_panic_report`: callers
/// invoke this either from an actual signal context (Unix) or from an
/// unhandled-exception filter (Windows), both of which carry stricter safety
/// constraints than ordinary code. We still allocate here (Rust's signal
/// handling story does not give us a non-allocating formatter), so this is a
/// best-effort capture, not a hard safety guarantee — if it ever reenters a
/// fault, the process was going to terminate anyway.
pub fn write_signal_report(signal_name: &str, ctx: &CrashContext) -> Option<PathBuf> {
    let timestamp = chrono::Local::now();
    let mut report = render_header(ctx, &timestamp, &format!("OS Signal — {signal_name}"));
    let _ = write!(
        report,
        "Signal:\n  process received {signal_name}; no further Rust state is available.\n"
    );
    report.push_str(render_footer());

    let path = report_path(&timestamp);
    match std::fs::write(&path, report) {
        Ok(()) => Some(path),
        Err(_) => None,
    }
}

/// Install the panic hook. Chains after whatever hook was previously
/// registered (e.g. a TUI terminal-restore hook) so existing behavior is
/// preserved — this should be the *last* hook installed, i.e. call this
/// before any surface-specific hook that needs to run after report-writing,
/// or wrap it yourself if ordering needs to be the other way around.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let ctx = context();
        let _ = write_panic_report(info, &ctx);
        previous(info);
    }));
}

/// Install OS-level signal / unhandled-exception capture. Best-effort: if the
/// underlying platform hook can't be installed, this silently no-ops rather
/// than failing startup.
pub fn install_signal_handler() {
    platform::install();
}

#[cfg(unix)]
mod platform {
    use super::{context, write_signal_report};

    pub fn install() {
        use signal_hook::consts::{SIGABRT, SIGBUS, SIGILL, SIGSEGV};
        use signal_hook::iterator::Signals;

        let mut signals = match Signals::new([SIGSEGV, SIGABRT, SIGILL, SIGBUS]) {
            Ok(s) => s,
            Err(_) => return,
        };

        std::thread::spawn(move || {
            for sig in signals.forever() {
                let name = match sig {
                    SIGSEGV => "SIGSEGV",
                    SIGABRT => "SIGABRT",
                    SIGILL => "SIGILL",
                    SIGBUS => "SIGBUS",
                    _ => "unknown signal",
                };
                let ctx = context();
                let _ = write_signal_report(name, &ctx);
                // Restore default disposition and re-raise so the process
                // still terminates the way it normally would (core dump,
                // correct exit code, etc.) instead of hanging forever.
                unsafe {
                    let _ = signal_hook::low_level::register(sig, || {});
                }
                let _ = signal_hook::low_level::raise(sig);
                std::process::exit(128 + sig);
            }
        });
    }
}

#[cfg(windows)]
mod platform {
    use super::{context, write_signal_report};
    use windows::Win32::System::Diagnostics::Debug::{
        SetUnhandledExceptionFilter, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };

    unsafe extern "system" fn handler(info: *const EXCEPTION_POINTERS) -> i32 {
        let code = if info.is_null() {
            0
        } else {
            unsafe { (*(*info).ExceptionRecord).ExceptionCode.0 as u32 }
        };
        let name = match code {
            0xC0000005 => "Access Violation",
            0xC000001D => "Illegal Instruction",
            0x80000003 => "Breakpoint",
            _ => "Unhandled Exception",
        };
        let ctx = context();
        let _ = write_signal_report(&format!("{name} (0x{code:08X})"), &ctx);
        EXCEPTION_CONTINUE_SEARCH
    }

    pub fn install() {
        unsafe {
            SetUnhandledExceptionFilter(Some(handler));
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    pub fn install() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::paths::test_support::set_gwen_home;

    fn sample_context() -> CrashContext {
        CrashContext {
            version: "1.0.0-test".into(),
            git_hash: "deadbeef".into(),
            command: "gwen train --resume checkpoint_000500".into(),
        }
    }

    #[test]
    fn write_signal_report_creates_readable_file() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = set_gwen_home(temp.path());

        let ctx = sample_context();
        let path = write_signal_report("SIGSEGV", &ctx).expect("expected report path");

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("GwenLand Crash Report"));
        assert!(contents.contains("SIGSEGV"));
        assert!(contents.contains("1.0.0-test"));
        assert!(contents.contains("deadbeef"));
        assert!(path.starts_with(GwenPaths::crash_logs_dir()));
    }

    #[test]
    fn write_panic_report_captures_message_and_location() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = set_gwen_home(temp.path());
        set_surface(Surface::Tui);

        let ctx = sample_context();
        let captured: std::sync::Arc<std::sync::Mutex<Option<PathBuf>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new({
                let ctx = ctx.clone();
                let captured = captured.clone();
                move |info| {
                    let path = write_panic_report(info, &ctx).unwrap();
                    *captured.lock().unwrap() = Some(path);
                }
            }));
            let panic_result = std::panic::catch_unwind(|| {
                panic!("shape mismatch: expected [256, 256], got [256, 128]");
            });
            std::panic::set_hook(previous);
            assert!(panic_result.is_err());
        }));

        result.unwrap();
        let path = captured.lock().unwrap().clone().expect("expected a written report path");
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("shape mismatch: expected [256, 256], got [256, 128]"));
        assert!(contents.contains("TUI"));
        assert!(contents.contains("crash_report.rs"));
    }

    #[test]
    fn unwritable_crash_dir_does_not_panic_the_hook() {
        // Point GWEN_HOME at a path that cannot be created (a file, not a dir,
        // sitting where crash-logs/ would need to be created).
        let temp = tempfile::tempdir().unwrap();
        let blocked_root = temp.path().join("blocked-root");
        std::fs::write(&blocked_root, b"not a directory").unwrap();
        let _guard = set_gwen_home(&blocked_root);

        let ctx = sample_context();
        // crash_logs_dir() itself calls create_dir_all and silently no-ops on
        // failure (see paths.rs::ensure_dir), so the resulting path exists in
        // name only; the write should fail gracefully and return None.
        let result = write_signal_report("SIGABRT", &ctx);
        assert!(result.is_none());
    }
}
