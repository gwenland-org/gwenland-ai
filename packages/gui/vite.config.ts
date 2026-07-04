import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Vite config tuned for Tauri.
//
//  - server.port 1420 / strictPort: must match `build.devUrl` in
//    tauri.conf.json. Tauri loads the dev server from this exact URL, so a
//    fallback port would silently break `tauri dev`.
//  - build.outDir "dist": resolves to packages/gui/dist, which is what
//    `build.frontendDist` ("../dist", relative to src-tauri) points at.
//  - envPrefix: also expose TAURI_* vars (Topbar reads import.meta.env
//    .TAURI_APP_VERSION) on top of the default VITE_ prefix.
//  - @tailwindcss/vite: the CSS entry uses Tailwind v4 (`@import "tailwindcss"`).

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  envPrefix: ["VITE_", "TAURI_"],
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    watch: {
      // Don't let Vite watch the Rust side — cargo handles that.
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Tauri targets a modern webview (Chromium on Windows, WebKit elsewhere),
    // so we can ship modern JS without legacy transpilation.
    target: "esnext",
    minify: "esbuild",
    sourcemap: false,
  },
});
