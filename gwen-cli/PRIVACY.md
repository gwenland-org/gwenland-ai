# GwenLand Privacy

GwenLand is designed to be **fully local**. No data is sent to any remote server by default.

## What stays on your machine

| Data | Location | Transmitted? |
|---|---|---|
| Conversation history | `~/.gwen/history/` | Never |
| Session error logs | `~/.gwen/session/` | Never |
| File contents (context injection) | Read at inference time | Only to local mistral.rs |
| Config | `~/.gwen/config.json` | Never |

## File context (JIN-164 — Relevance Windowing)

When you include files in a chat query, GwenLand reads their content locally.

- With `compression.enabled = true`: only **relevant line windows** (not the full file)
  are sent to the local mistral.rs process.
- With `compression.enabled = false` (default): the full file is sent to the local
  mistral.rs process.
- In both cases, file content is **never transmitted beyond your local machine**.

## Session logs (`~/.gwen/session/`)

`session_<timestamp>.txt` files contain:
- Error messages and warnings logged during the session
- A TUI state snapshot (active pane, scroll position, input buffer)
- Crash information if the process panicked (formatted stack trace)

These files are never sent anywhere. You can delete them at any time:

```
gwenland session clear
```

List all session files with their status:

```
gwenland session list
```

## History (`~/.gwen/history/`)

Conversation history is stored as JSONL, one file per session.
It is loaded locally on startup and never leaves your machine.
