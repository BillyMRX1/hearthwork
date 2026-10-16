---
title: Ports and environment variables
description: The network ports Hearthwork uses and the environment variables that change its behaviour.
---

## Ports

| Port | Used by | Notes |
|---|---|---|
| 8470 | The Hearthwork host (relay and queue) | Listens on `127.0.0.1` only. |
| 8080 | `hearthwork api` | This computer only. Change with `--port N`. |
| 8484 | `hearthwork share` (TCP) | Change with `--port N`. Connect to it with `hearthwork connect <ip>[:8484]`. |
| 8485 | Sharing discovery (UDP) | Used by `hearthwork connect` to find a host on the local network. |

The model server (llama-server) listens on its own local port, which is set under `server.port` in `config.json`.

## Environment variables

| Variable | Effect |
|---|---|
| `HEARTHWORK_HOME` | Use another [data folder](/reference/data-and-config/). |
| `HF_TOKEN` | Hugging Face token for gated or private repositories. |
| `HEARTHWORK_CLAUDE_OVERRIDES` | Path to a settings file merged on top of the generated Claude Code settings (default `~/.claude/settings-hearthwork.json`). |
| `HEARTHWORK_TOOL_REPAIR=0` | Turn off [tool-call repair](/how-it-works/relay/). |
| `HEARTHWORK_RELAY_DUMP=<folder>` | Write every request before and after cleanup, for debugging. |
| `HEARTHWORK_CAPTURE=1` | Also write full request and response bodies to `logs/capture.jsonl`. Off by default. |
| `XDG_DATA_HOME` | Linux: base folder of the data folder. |

The agents' own variables, such as `OPENCODE_CONFIG_CONTENT`, `QWEN_CODE_SYSTEM_SETTINGS_PATH`, `CODEX_HOME` and `CLAUDE_CONFIG_DIR`, are set by Hearthwork for the processes it starts. You normally never set them yourself.
