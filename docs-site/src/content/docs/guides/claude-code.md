---
title: Claude Code
description: Run Claude Code on your local model with hearthwork claude.
---

| | |
|---|---|
| Command | `hearthwork claude` |
| Install | https://code.claude.com |
| API used | Anthropic Messages |

Run `hearthwork claude` from the project you want it to work on. A model is started first if none is running. Arguments after the command go to Claude Code, for example `hearthwork claude -p "explain this repo"`.

## Your settings are not changed

`hearthwork claude` writes `claude-settings-<model>.json` in the [data folder](/reference/data-and-config/) and starts Claude Code with `--settings` pointing at it. Your `~/.claude/settings.json` is not changed, and plain `claude` stays your normal cloud Claude.

The generated settings:

- Set `permissions.defaultMode: "default"`, so Claude Code asks before risky actions. In auto mode the local model would check every command first, which is slow and can time out.
- Add a status line such as `local · <model> · 64K`.

If `~/.claude/settings-hearthwork.json` exists, it is merged on top. Set `HEARTHWORK_CLAUDE_OVERRIDES` to point to another file. `--max-output N` caps each reply (default 4096 tokens).

## Tell it what the model can do

Claude Code is told the real context window and output limit of the running (or shared) model. The relay also keeps the start of the prompt stable (it removes Claude Code's per-session billing-header line and token counter), so llama.cpp can reuse its cache.

## Inside another agent

`hearthwork claude` needs a terminal. If you run it from inside another agent without one, it exits with code 2 and suggests `hearthwork task --agent claude "..."` instead. See [Subagents](/guides/subagents/).
