#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // GWEN-224 Wave 3: crash reporting for the GUI surface. Tauri's release
    // profile sets `panic = "abort"`, so the hook still runs (it fires before
    // the abort), it just can't unwind past main() — that's fine, we only
    // need it to format and write the report.
    gwenland_core::diagnostics::crash_report::set_surface(
        gwenland_core::diagnostics::crash_report::Surface::Gui,
    );
    gwenland_core::diagnostics::crash_report::init_context(
        env!("CARGO_PKG_VERSION"),
        option_env!("VERGEN_GIT_SHA").unwrap_or("unknown"),
    );
    gwenland_core::diagnostics::crash_report::install_panic_hook();
    gwenland_core::diagnostics::crash_report::install_signal_handler();

    gwen_gui_lib::run();
}
