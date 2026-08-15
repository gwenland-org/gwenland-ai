//! `--help` must exit 0. Spawns the real binary rather than calling clap
//! in-process: the exit *code* is what a shell script or CI step sees, and
//! that is produced by `main`, not by the parser.

use std::process::Command;

#[test]
fn help_exits_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_gwen")).arg("--help").output().expect("run gwen --help");
    assert!(out.status.success(), "--help exited with {:?}", out.status.code());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("run"), "help should list the run subcommand");
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let out = Command::new(env!("CARGO_BIN_EXE_gwen")).arg("nonsense").output().expect("run gwen");
    assert!(!out.status.success(), "an unknown subcommand must not exit 0");
}
