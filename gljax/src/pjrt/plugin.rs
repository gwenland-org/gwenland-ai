//! Dynamic PJRT plugin loading (ARTX01 §1.3, §5.2).
//!
//! gljax never links a PJRT plugin at build time. It `dlopen`s / `LoadLibrary`s
//! the plugin binary, resolves `GetPjrtApi`, and calls it for the vtable.

use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};

use crate::pjrt::error::{check, missing_slot};
use crate::sys::ffi::GetPjrtApiFn;
use crate::sys::types::*;
use crate::{struct_size, GlError};

/// Environment variable naming the CPU plugin binary. ARTX01 §5.4.
pub const ENV_PLUGIN_CPU: &str = "PJRT_PLUGIN_CPU";

/// Alias accepted for the CPU plugin path.
///
/// The ARTX01–05 sprint brief names this variable; ARTX01 §5.4 names
/// [`ENV_PLUGIN_CPU`]. Both are honoured so neither document is silently
/// wrong, with `PJRT_PLUGIN_CPU` taking precedence.
pub const ENV_PLUGIN_CPU_ALIAS: &str = "PJRT_CPU_PLUGIN_PATH";

/// Resolves the CPU plugin path from the environment.
///
/// Returns `None` when neither variable is set — the caller decides whether
/// that is a skip (tests) or an error (production).
pub fn cpu_plugin_path() -> Option<PathBuf> {
    for key in [ENV_PLUGIN_CPU, ENV_PLUGIN_CPU_ALIAS] {
        if let Ok(v) = std::env::var(key) {
            if !v.trim().is_empty() {
                return Some(PathBuf::from(v));
            }
        }
    }
    None
}

/// A loaded PJRT plugin: the library handle plus its vtable.
///
/// The [`Library`] must outlive every pointer derived from it — dropping it
/// unloads the binary and turns the whole vtable into dangling code pointers.
/// That is why it is held here and why [`PjrtPlugin`] is what everything else
/// borrows from.
pub struct PjrtPlugin {
    // Field order matters for drop order: `api` is derived from `_lib`, so the
    // library must be dropped last.
    api: *const PjrtApi,
    path: PathBuf,
    _lib: Library,
}

// SAFETY: `PJRT_Api` is a table of C function pointers that the plugin
// populates once, before `GetPjrtApi` returns, and never mutates afterwards.
// ARTX01 §1.7 states `PJRT_Client` is thread-safe for concurrent compile and
// execute; the vtable itself is immutable shared data. Note this says nothing
// about `PJRT_Buffer`, which is *not* thread-safe and is not covered here.
unsafe impl Send for PjrtPlugin {}
// SAFETY: as above — shared references only ever read immutable function
// pointers.
unsafe impl Sync for PjrtPlugin {}

impl PjrtPlugin {
    /// Loads a PJRT plugin binary and validates its API version.
    ///
    /// The version check has two parts, and they answer different questions:
    ///
    /// * **Major version** must match exactly. A major bump means arguments or
    ///   field order changed, which `struct_size` cannot detect — the only
    ///   safe response is to refuse (P5).
    /// * **`struct_size`** must reach past the last vtable slot gljax calls.
    ///   This is the real compatibility test, and it is why gljax does *not*
    ///   follow ARTX01 §9.1's advice to refuse on any minor divergence: that
    ///   would reject every plugin that is not this exact build, including
    ///   newer ones that are fully compatible.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GlError> {
        let path = path.as_ref().to_path_buf();

        // SAFETY: loading an arbitrary shared library runs its initializers.
        // The path comes from operator configuration (an env var or an
        // explicit call), which is the same trust level as the binary itself.
        let lib = unsafe { Library::new(&path) }.map_err(|e| {
            GlError::Engine(format!(
                "PJRT plugin load failed for {}: {e}",
                path.display()
            ))
        })?;

        let api = {
            // SAFETY: `GetPjrtApi` is the symbol every PJRT plugin exports,
            // with this exact signature (ARTX01 §1.3). A binary that exports
            // it with another signature is not a PJRT plugin.
            let get_api: Symbol<GetPjrtApiFn> =
                unsafe { lib.get(b"GetPjrtApi\0") }.map_err(|e| {
                    GlError::Engine(format!(
                        "{} exports no GetPjrtApi — not a PJRT plugin ({e})",
                        path.display()
                    ))
                })?;
            // SAFETY: resolved symbol with the correct signature; the returned
            // pointer is owned by the plugin and lives as long as `lib`.
            unsafe { get_api() }
        };

        if api.is_null() {
            return Err(GlError::Engine(format!(
                "GetPjrtApi returned null for {}",
                path.display()
            )));
        }

        let plugin = PjrtPlugin {
            api,
            path,
            _lib: lib,
        };
        plugin.check_version()?;
        plugin.initialize()?;
        Ok(plugin)
    }

    /// Loads the CPU plugin named by the environment.
    pub fn load_cpu_from_env() -> Result<Self, GlError> {
        let path = cpu_plugin_path().ok_or_else(|| {
            GlError::Engine(format!(
                "no PJRT CPU plugin configured — set {ENV_PLUGIN_CPU} (or {ENV_PLUGIN_CPU_ALIAS})"
            ))
        })?;
        Self::load(path)
    }

    /// The vtable. Callers must not outlive the plugin.
    pub(crate) fn api(&self) -> *const PjrtApi {
        self.api
    }

    /// Path the plugin was loaded from — the useful half of any error message.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `(major, minor)` as reported by the plugin.
    pub fn api_version(&self) -> (i32, i32) {
        // SAFETY: `self.api` is non-null and the library is alive.
        let v = unsafe { (*self.api).pjrt_api_version };
        (v.major_version, v.minor_version)
    }

    fn check_version(&self) -> Result<(), GlError> {
        let (major, minor) = self.api_version();
        if major != PJRT_API_MAJOR {
            return Err(GlError::Engine(format!(
                "PJRT major version mismatch: plugin {} reports {major}.{minor}, \
                 gljax was bound against {PJRT_API_MAJOR}.{PJRT_API_MINOR}. \
                 A major bump reorders or retypes the vtable; refusing rather than guessing",
                self.path.display()
            )));
        }

        // The reachability check. `PJRT_Buffer_ToHostBuffer` is the
        // highest-indexed slot gljax calls (slot 71 of 138), so a plugin whose
        // struct reaches it carries every slot gljax needs.
        // SAFETY: `self.api` is non-null and the library is alive.
        let reported = unsafe { (*self.api).struct_size };
        let required = struct_size!(
            PjrtApi,
            PJRT_Buffer_ToHostBuffer: Option<crate::sys::ffi::PjrtBufferToHostBufferFn>
        );
        if reported < required {
            return Err(GlError::Engine(format!(
                "PJRT plugin {} reports struct_size {reported}, but gljax calls slots up to \
                 PJRT_Buffer_ToHostBuffer at offset {required}. Plugin API {major}.{minor} is \
                 older than gljax needs",
                self.path.display()
            )));
        }

        if minor != PJRT_API_MINOR {
            log::info!(
                "PJRT plugin {} is API {major}.{minor}; gljax was bound against \
                 {PJRT_API_MAJOR}.{PJRT_API_MINOR}. Compatible: every slot gljax calls is present.",
                self.path.display()
            );
        }
        Ok(())
    }

    /// `PJRT_Plugin_Initialize` — "must be called before any other functions".
    fn initialize(&self) -> Result<(), GlError> {
        // SAFETY: `self.api` is non-null and the library is alive.
        let Some(f) = (unsafe { (*self.api).PJRT_Plugin_Initialize }) else {
            return Err(missing_slot("PJRT_Plugin_Initialize"));
        };
        let mut args = PjrtPluginInitializeArgs {
            struct_size: struct_size!(PjrtPluginInitializeArgs, extension_start: *mut PjrtExtensionBase),
            extension_start: core::ptr::null_mut(),
        };
        // SAFETY: `args` is fully initialized with a correct `struct_size`.
        let err = unsafe { f(&mut args) };
        // SAFETY: `err` is the owned error this call just returned.
        unsafe { check(self.api, err, "Plugin_Initialize") }
    }
}
