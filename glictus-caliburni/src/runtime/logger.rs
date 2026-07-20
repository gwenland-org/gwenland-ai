//! Structured runtime logging (ARTX05 §"Logging and Diagnostics").
//!
//! Entries are captured in memory rather than printed, for three reasons:
//! the crate takes no logging dependency, tests can assert on what was
//! recorded, and a caller can forward entries to whatever logger it already
//! uses. `emit_to_stderr` opts into printing.

use std::sync::{Arc, Mutex};

use crate::runtime::types::RuntimeLogLevel;

/// One recorded log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Severity of this entry.
    pub level: RuntimeLogLevel,
    /// Human-readable message.
    pub message: String,
    /// Optional structured context (layer index, path, byte counts).
    pub context: Option<String>,
}

impl LogEntry {
    /// Render as `LEVEL: message (context)`.
    pub fn render(&self) -> String {
        match &self.context {
            Some(ctx) => format!("{}: {} ({})", self.level.as_str(), self.message, ctx),
            None => format!("{}: {}", self.level.as_str(), self.message),
        }
    }
}

/// Collects [`LogEntry`] values at or above a minimum severity.
///
/// Cheap to clone and shared across runtime components via [`Arc`]; the
/// interior [`Mutex`] makes it `Send + Sync`.
#[derive(Debug)]
pub struct RuntimeLogger {
    min_level: RuntimeLogLevel,
    emit_to_stderr: bool,
    entries: Mutex<Vec<LogEntry>>,
}

impl RuntimeLogger {
    /// Logger that records entries and also prints them to stderr.
    pub fn new(min_level: RuntimeLogLevel) -> Self {
        RuntimeLogger {
            min_level,
            emit_to_stderr: true,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Logger that records entries without printing. Use in tests and
    /// libraries that render their own output.
    pub fn silent(min_level: RuntimeLogLevel) -> Self {
        RuntimeLogger {
            min_level,
            emit_to_stderr: false,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// A silent logger wrapped in an [`Arc`], ready to share.
    pub fn shared_silent(min_level: RuntimeLogLevel) -> Arc<Self> {
        Arc::new(Self::silent(min_level))
    }

    /// Minimum severity this logger records.
    pub fn min_level(&self) -> RuntimeLogLevel {
        self.min_level
    }

    /// Record a message if `level` passes the minimum-severity filter.
    ///
    /// `RuntimeLogLevel` sorts by severity (Error lowest), so "at or above
    /// severity" is `level <= min_level`.
    pub fn log(&self, level: RuntimeLogLevel, message: impl Into<String>, context: Option<String>) {
        if level > self.min_level {
            return;
        }
        let entry = LogEntry {
            level,
            message: message.into(),
            context,
        };
        if self.emit_to_stderr {
            eprintln!("{}", entry.render());
        }
        // A poisoned lock means another thread panicked mid-log. Losing log
        // entries must never escalate into a panic in the runtime, so recover
        // the guard and carry on.
        match self.entries.lock() {
            Ok(mut g) => g.push(entry),
            Err(poisoned) => poisoned.into_inner().push(entry),
        }
    }

    /// Record at ERROR.
    pub fn error(&self, message: impl Into<String>, context: Option<String>) {
        self.log(RuntimeLogLevel::Error, message, context);
    }

    /// Record at WARN.
    pub fn warn(&self, message: impl Into<String>, context: Option<String>) {
        self.log(RuntimeLogLevel::Warn, message, context);
    }

    /// Record at INFO.
    pub fn info(&self, message: impl Into<String>, context: Option<String>) {
        self.log(RuntimeLogLevel::Info, message, context);
    }

    /// Record at DEBUG.
    pub fn debug(&self, message: impl Into<String>, context: Option<String>) {
        self.log(RuntimeLogLevel::Debug, message, context);
    }

    /// Snapshot of everything recorded so far.
    pub fn entries(&self) -> Vec<LogEntry> {
        match self.entries.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// How many entries were recorded at exactly `level`.
    pub fn count_at(&self, level: RuntimeLogLevel) -> usize {
        self.entries().iter().filter(|e| e.level == level).count()
    }

    /// Total entries recorded.
    pub fn len(&self) -> usize {
        self.entries().len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all recorded entries (used by
    /// [`GllmRuntime::reset`](crate::runtime::GllmRuntime::reset)).
    pub fn clear(&self) {
        match self.entries.lock() {
            Ok(mut g) => g.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }
}

impl Default for RuntimeLogger {
    fn default() -> Self {
        Self::new(RuntimeLogLevel::Info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_entries_below_the_minimum_level() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Warn);
        log.error("fatal", None);
        log.warn("suboptimal", None);
        log.info("progress", None);
        log.debug("detail", None);

        assert_eq!(log.len(), 2, "Info and Debug must be filtered out");
        assert_eq!(log.count_at(RuntimeLogLevel::Error), 1);
        assert_eq!(log.count_at(RuntimeLogLevel::Warn), 1);
        assert_eq!(log.count_at(RuntimeLogLevel::Info), 0);
        assert_eq!(log.count_at(RuntimeLogLevel::Debug), 0);
    }

    #[test]
    fn debug_level_records_everything() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Debug);
        log.error("e", None);
        log.warn("w", None);
        log.info("i", None);
        log.debug("d", None);
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn error_level_records_only_errors() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Error);
        log.error("e", None);
        log.warn("w", None);
        log.info("i", None);
        assert_eq!(log.len(), 1);
        assert_eq!(log.count_at(RuntimeLogLevel::Error), 1);
    }

    #[test]
    fn context_is_preserved_and_rendered() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Info);
        log.info("mapped layer", Some("layer=3 bytes=1024".into()));

        let entries = log.entries();
        assert_eq!(entries[0].context.as_deref(), Some("layer=3 bytes=1024"));
        assert_eq!(entries[0].render(), "INFO: mapped layer (layer=3 bytes=1024)");
    }

    #[test]
    fn renders_without_context() {
        let e = LogEntry {
            level: RuntimeLogLevel::Warn,
            message: "cpu fallback".into(),
            context: None,
        };
        assert_eq!(e.render(), "WARN: cpu fallback");
    }

    #[test]
    fn silent_logger_records_but_does_not_print() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Debug);
        assert!(!log.emit_to_stderr);
        log.info("quiet", None);
        assert_eq!(log.len(), 1, "silent still records");
    }

    #[test]
    fn clear_drops_all_entries() {
        let log = RuntimeLogger::silent(RuntimeLogLevel::Info);
        log.info("a", None);
        log.info("b", None);
        assert_eq!(log.len(), 2);
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn is_shareable_across_threads() {
        let log = RuntimeLogger::shared_silent(RuntimeLogLevel::Info);
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let log = Arc::clone(&log);
                std::thread::spawn(move || log.info(format!("from {i}"), None))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(log.len(), 4);
    }
}
