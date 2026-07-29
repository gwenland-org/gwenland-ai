//! Wave A1 — PJRT FFI bring-up.
//!
//! # Two kinds of test in this file
//!
//! * **Host-only** tests exercise the StableHLO emitter and need nothing.
//! * **Plugin** tests need a PJRT plugin binary and SKIP loudly without one,
//!   the glcuda convention (`rust-skills/testing-standards.md` rule 5).
//!
//! ⛔ A SKIP is **not** a pass. The plugin tests below are the ones that decide
//! whether the FFI in `gljax::sys` is right, and on a machine without a plugin
//! they decide nothing. Point `PJRT_PLUGIN_CPU` (or `PJRT_CPU_PLUGIN_PATH`) at
//! **`libpjrt_cpu.so`** from a `zml/pjrt-artifacts` release to run them for
//! real — see `gljax/README.md` for the pinned tag and its SHA-256.
//!
//! ⚠️ That filename is not what ARTX01 §1.4 says (`pjrt_c_api_cpu_plugin.so`);
//! zml does not ship that name. `.github/workflows/gljax-pjrt.yml` runs these
//! on `ubuntu-latest` and **fails the build if any of them skips**, so a green
//! CI run cannot be mistaken for coverage.

use std::rc::Rc;

use gljax::pjrt::{cpu_plugin_path, PjrtClientHandle, PjrtPlugin};
use gljax::stablehlo::{smoke, DType, MlirEmitter, Shape};

/// f32 round-trips exactly through PJRT for values this small, so the
/// tolerance covers transfer/format slop only, not arithmetic error.
const TOL_F32: f32 = 1e-6;

/// Loads the CPU plugin, or prints a SKIP and returns `None`.
fn load_cpu_plugin_or_skip(test: &str) -> Option<PjrtPlugin> {
    let Some(path) = cpu_plugin_path() else {
        eprintln!(
            "SKIP {test}: no PJRT plugin configured \
             (set PJRT_PLUGIN_CPU or PJRT_CPU_PLUGIN_PATH)"
        );
        return None;
    };
    if !path.exists() {
        eprintln!(
            "SKIP {test}: PJRT plugin path {} does not exist",
            path.display()
        );
        return None;
    }
    match PjrtPlugin::load(&path) {
        Ok(p) => Some(p),
        Err(e) => panic!(
            "PJRT plugin at {} is configured but failed to load: {e}",
            path.display()
        ),
    }
}

// ---------------------------------------------------------------------------
// Plugin tests
// ---------------------------------------------------------------------------

#[test]
fn pjrt_cpu_plugin_loads_without_panic() {
    let Some(plugin) = load_cpu_plugin_or_skip("pjrt_cpu_plugin_loads_without_panic") else {
        return;
    };
    let (major, minor) = plugin.api_version();
    eprintln!(
        "PJRT plugin {} reports API {major}.{minor}",
        plugin.path().display()
    );
    assert_eq!(
        major,
        gljax::PJRT_API_VERSION_BOUND.0,
        "major version must match the bindings; load() should already have refused"
    );

    let client = PjrtClientHandle::create(Rc::new(plugin)).expect("client create failed");
    let platform = client.platform_name().expect("platform_name failed");
    let version = client.platform_version().expect("platform_version failed");
    let devices = client
        .addressable_devices()
        .expect("addressable_devices failed");
    eprintln!("PJRT platform {platform:?} version {version:?}, {} device(s)", devices.len());

    assert!(!platform.is_empty(), "platform name must not be empty");
    assert!(
        !devices.is_empty(),
        "a CPU plugin must expose at least one addressable device"
    );
}

#[test]
fn stablehlo_add_scalar_compiles_and_executes() {
    let Some(plugin) = load_cpu_plugin_or_skip("stablehlo_add_scalar_compiles_and_executes") else {
        return;
    };
    let client = PjrtClientHandle::create(Rc::new(plugin)).expect("client create failed");
    let device = client.default_device().expect("no addressable device");

    let module = smoke::add_scalar_module();
    let program = client
        .compile(&module)
        .unwrap_or_else(|e| panic!("compile failed for module:\n{module}\nerror: {e}"));

    assert_eq!(
        program.num_outputs().expect("num_outputs failed"),
        1,
        "the smoke module returns a single scalar"
    );

    let scalar = Shape::scalar(DType::F32);
    let lhs = client
        .buffer_from_host_f32(&[2.0], &scalar, &device)
        .expect("upload lhs failed");
    let rhs = client
        .buffer_from_host_f32(&[3.0], &scalar, &device)
        .expect("upload rhs failed");

    let outputs = program.execute(&[&lhs, &rhs]).expect("execute failed");
    assert_eq!(outputs.len(), 1);

    let host = outputs[0].to_host_f32().expect("transfer back failed");
    assert_eq!(host.len(), 1, "expected one scalar back, got {host:?}");
    let err = (host[0] - 5.0f32).abs();
    assert!(
        err <= TOL_F32,
        "2.0 + 3.0 came back as {} (|err| = {err:e})",
        host[0]
    );
}

/// ARTX02's opening claim is that the MLIR parser accepts both the generic and
/// the pretty op syntax. gljax emits generic; this compiles the pretty form so
/// the claim is measured rather than assumed (P2).
#[test]
fn pjrt_accepts_the_pretty_syntax_form_too() {
    let Some(plugin) = load_cpu_plugin_or_skip("pjrt_accepts_the_pretty_syntax_form_too") else {
        return;
    };
    let client = PjrtClientHandle::create(Rc::new(plugin)).expect("client create failed");
    let device = client.default_device().expect("no addressable device");

    let program = client
        .compile(smoke::ADD_SCALAR_MODULE_PRETTY)
        .unwrap_or_else(|e| {
            panic!(
                "PJRT rejected the pretty form — ARTX02's \"both forms\" claim is wrong: {e}\n{}",
                smoke::ADD_SCALAR_MODULE_PRETTY
            )
        });

    let scalar = Shape::scalar(DType::F32);
    let lhs = client
        .buffer_from_host_f32(&[2.0], &scalar, &device)
        .expect("upload lhs failed");
    let rhs = client
        .buffer_from_host_f32(&[3.0], &scalar, &device)
        .expect("upload rhs failed");
    let outputs = program.execute(&[&lhs, &rhs]).expect("execute failed");
    let host = outputs[0].to_host_f32().expect("transfer back failed");
    assert!((host[0] - 5.0f32).abs() <= TOL_F32, "got {}", host[0]);
}

// ---------------------------------------------------------------------------
// Host-only tests
// ---------------------------------------------------------------------------

#[test]
fn mlir_emitter_produces_valid_utf8() {
    let mut e = MlirEmitter::new();
    e.line("module @test {");
    e.push_indent();
    e.line("// comment");
    e.pop_indent();
    e.line("}");
    let s = e.into_body();

    assert!(s.contains("module @test"));
    assert!(s.contains("// comment"));
    // A `String` is UTF-8 by construction; the meaningful check is that the
    // bytes survive the round trip, i.e. nothing was written through a path
    // that could produce a lone surrogate or a truncated sequence.
    assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    assert_eq!(std::str::from_utf8(s.as_bytes()).unwrap(), s);
}

#[test]
fn smoke_module_is_ascii_and_newline_terminated() {
    // PJRT receives `(ptr, len)`, not a NUL-terminated string, so embedded
    // NULs or a missing trailing newline would not be caught by the C side.
    for module in [
        smoke::add_scalar_module(),
        smoke::ADD_SCALAR_MODULE_PRETTY.to_owned(),
    ] {
        assert!(module.is_ascii(), "MLIR text must be ASCII:\n{module}");
        assert!(!module.contains('\0'), "MLIR text must not contain NUL");
        assert!(module.ends_with('\n'), "MLIR text must end with a newline");
    }
}

/// The plugin-path resolution has to agree with what the SKIP messages tell
/// the user to set, or a configured plugin silently goes unused.
#[test]
fn plugin_path_resolves_from_either_documented_env_var() {
    // Read-only: whichever variable this machine has (or neither) is fine.
    match cpu_plugin_path() {
        Some(p) => eprintln!("PJRT plugin configured at {}", p.display()),
        None => eprintln!(
            "no PJRT plugin configured — {} and {} are both unset",
            gljax::pjrt::ENV_PLUGIN_CPU,
            gljax::pjrt::ENV_PLUGIN_CPU_ALIAS
        ),
    }
    assert_eq!(gljax::pjrt::ENV_PLUGIN_CPU, "PJRT_PLUGIN_CPU");
    assert_eq!(gljax::pjrt::ENV_PLUGIN_CPU_ALIAS, "PJRT_CPU_PLUGIN_PATH");
}
