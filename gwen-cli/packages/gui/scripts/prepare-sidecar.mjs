// Builds the `gwen` CLI (gwenland-tui) in release mode and copies it next to the
// Tauri app as a sidecar binary named `gwen-<target-triple>(.exe)`, which is what
// Tauri's `bundle.externalBin: ["binaries/gwen"]` resolves and ships inside the
// installer. Cross-platform (Windows/Linux/macOS) — the target triple is read
// from `rustc -vV` so the same script works locally and in CI.
//
// Wired into tauri.conf.json `beforeBuildCommand`, so `pnpm tauri build` always
// has a fresh sidecar. The cargo build is incremental, so it's cheap after the
// first compile (and shares gwenland-core with the GUI's own build).

import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const guiDir = join(here, "..");            // packages/gui
const workspace = join(guiDir, "..", ".."); // gwen-cli (cargo workspace root)
const binDir = join(guiDir, "src-tauri", "binaries");

// Host target triple, e.g. x86_64-pc-windows-msvc / x86_64-unknown-linux-gnu.
const hostLine = execSync("rustc -vV", { encoding: "utf8" })
  .split("\n")
  .find((l) => l.startsWith("host:"));
if (!hostLine) throw new Error("could not determine host target triple from `rustc -vV`");
const triple = hostLine.split(":")[1].trim();
const ext = process.platform === "win32" ? ".exe" : "";

console.log(`[sidecar] building gwen CLI (release) for ${triple} …`);
execSync("cargo build --release -p gwenland-tui", { cwd: workspace, stdio: "inherit" });

const src = join(workspace, "target", "release", `gwenland${ext}`);
mkdirSync(binDir, { recursive: true });
const dest = join(binDir, `gwen-${triple}${ext}`);
copyFileSync(src, dest);
console.log(`[sidecar] copied ${src} -> ${dest}`);
