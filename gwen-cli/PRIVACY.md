# Privacy

GwenLand runs locally. Nothing you do with it is sent to a remote server unless you explicitly ask for it — and the only thing that reaches out is downloading or pushing models and datasets through `gwen fetch` and `gwen hub`. Your conversations, your files, and your config never leave the machine.

## What's stored, and where

Everything lives under `~/.gwenland/`:

- **Conversation history** is written as JSONL, one file per session. It's loaded locally when you start up and goes nowhere else.
- **Session files** hold things like error messages and warnings from a run, a snapshot of the TUI state (active pane, scroll position, input buffer), and — if the process crashed — the crash report.
- **Config** is a single `config/config.json`.

None of these are transmitted anywhere. Delete any of them whenever you want; GwenLand recreates what it needs on the next run, and `gwen doctor` will tell you where the folders are and whether they're writable.

## Files you bring into a chat

When you reference a file in a query, GwenLand reads it locally and hands the content to the in-process inference engine running on your machine. Depending on your settings it may send only the relevant slices of a file rather than the whole thing, but either way the content stays local — it's never sent off the machine.

## Crashes

If GwenLand crashes, it writes a readable report to `~/.gwenland/crash-logs/`. The report includes the version, which part of the app was running, the command line, some OS details, and the error itself. It stays on disk for you to read or share if you're filing a bug — it isn't uploaded.
