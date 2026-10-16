---
title: OpenCode
description: Run OpenCode on your local model with hearthwork opencode.
---

| | |
|---|---|
| Command | `hearthwork opencode` |
| Install | `npm i -g opencode-ai` |
| API used | OpenAI Chat Completions |

Run `hearthwork opencode` from your project folder. Extra arguments go to OpenCode.

## Your settings are not changed

OpenCode is configured through the `OPENCODE_CONFIG_CONTENT` environment variable, so your own OpenCode configuration is not edited. OpenCode still writes its history where it always does.

Preview exactly what would run with `hearthwork opencode --dry-run`.

## Used as a subagent

In `hearthwork task`, OpenCode can read, search and edit files, and run only commands matching an `--allow` pattern. See [Subagents](/guides/subagents/).
