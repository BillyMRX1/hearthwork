---
title: Codex
description: Run OpenAI Codex CLI on your local model with hearthwork codex.
---

| | |
|---|---|
| Command | `hearthwork codex` |
| Install | `npm i -g @openai/codex` |
| API used | OpenAI Responses |

Run `hearthwork codex` from your project folder. Extra arguments go to Codex: `hearthwork codex exec "fix the failing test"`.

## Your settings are not changed

Codex gets `-c` overrides on the command line. Your own Codex configuration is not edited, and Codex still writes its history where it always does.

## Preview

```sh
hearthwork codex --dry-run
```

prints the exact command and environment without starting anything.

## Used as a subagent

In `hearthwork task`, Codex runs in its `workspace-write` sandbox: it edits inside the folder and runs commands there. The `--allow` option has no effect for Codex. Over MCP, some clients (Codex among them) cancel a tool call after about 60 seconds, so use `local_task_start` and `local_task_result`. See [Subagents](/guides/subagents/).
