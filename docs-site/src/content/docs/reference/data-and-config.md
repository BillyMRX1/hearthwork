---
title: Data folders and config
description: Where Hearthwork keeps your settings, llama.cpp builds, caches and results on each operating system.
---

Settings, llama.cpp, caches and results are stored per user, outside the install.

| OS | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\hearthwork` |
| macOS | `~/Library/Application Support/hearthwork` |
| Linux | `$XDG_DATA_HOME/hearthwork` (default `~/.local/share/hearthwork`) |

Set `HEARTHWORK_HOME` to use another folder. `hearthwork --version` prints the folder in use.

## What is inside

| Path | Contents |
|---|---|
| `config.json` | Your settings (including the `server` section and budget). |
| `bin/` | llama.cpp builds. |
| `cache/slots/` | Saved prompt caches. |
| `bench/` | Benchmark results (`results.jsonl`) and each run's folder (`runs/`). |
| `doctor/` | Doctor reports (JSON). |
| `profiles/` | Setup profiles. |
| `logs/` | `requests.jsonl` (numbers only) and the host log. |
| `templates/` | Chat-template overrides. |
| `server.json`, `server.log` | The running model server, and its log. |

## Config

Settings live in `config.json`. The `server` section holds what the model server uses (such as `context`, prompt batch size, VRAM headroom and KV cache type). `hearthwork setup` recalculates them for your hardware, and `hearthwork start --model X` or `--context N` override a single run.

Prefer the commands (`hearthwork context`, `budget`, `idle`, `profile`) over editing the file by hand.

Hearthwork never edits your agents' own configuration. Files it generates for them (for example `claude-settings-<model>.json`) are written in this folder.
