---
title: Local agents as subagents
description: Let any agent hand a coding task to a local agent, by command (hearthwork task) or by MCP.
---

Any agent harness (cloud Claude Code, Codex, Cursor, OpenCode and others) can hand a coding task to a local agent running on the local model, or on the computer you `hearthwork connect` to, and get the result back. There are two ways.

## 1. The `task` command

From the other agent's shell tool:

```sh
hearthwork task [--agent claude|codex|opencode|aider|qwen] [--allow "pytest *"]... [--cwd DIR] [--timeout 900] [--json] [--dry-run] "TASK"
echo "TASK" | hearthwork task -        # "-" reads the task from stdin: no shell quoting problems
```

It starts the model if none is running, runs the agent non-interactively, and prints a report on stdout: agent, model, host, duration, status, files created, modified and deleted, and the agent's final message. Progress goes to stderr.

Exit codes: **0** success, **124** timeout (the agent and its children are killed), **1** agent error.

| Agent | What it may do in a task |
|---|---|
| Claude Code, OpenCode, Qwen Code | Read, search and edit files, and run only commands matching an `--allow` pattern (`--allow "python *"`). Everything else is denied. |
| Codex | Runs in its `workspace-write` sandbox: edits inside the folder and runs commands there. `--allow` has no effect. |
| Aider | Only edits files. Shell commands are switched off. |

## 2. The MCP server

So the other agent gets tools instead of a shell command:

```sh
hearthwork mcp install claude   # claude mcp add --scope user hearthwork -- <path to hearthwork> mcp
hearthwork mcp install codex    # codex mcp add hearthwork -- <path to hearthwork> mcp
hearthwork mcp install print    # JSON for Cursor, OpenCode, Claude Desktop
```

Tools:

- `local_task_start` returns an id, and `local_task_result` waits up to 55 seconds per call.
- `local_task` blocks until the task is done.
- `local_model_status` reports the model.

Prefer start plus result: some clients, Codex among them, cancel an MCP tool call after about 60 seconds. For the blocking `local_task` in Codex, set `tool_timeout_sec = 900` under `[mcp_servers.hearthwork]` in `~/.codex/config.toml`.

## Tips

Give small, precise tasks (file names, interfaces, the test command) and review the result, because the local model is usually weaker than the agent calling it. Parallel tasks are limited by the model's slots, so give them different files.
