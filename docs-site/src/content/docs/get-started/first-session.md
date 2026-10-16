---
title: Your first agent session
description: Start an agent on the local model from your project folder.
---

Go to the project you want the agent to work on, then start an agent:

```sh
cd my-project
hearthwork claude      # also: codex, opencode, qwen, aider
```

If no model is running, Hearthwork starts one first. Run `hearthwork` alone for a menu with the same choices.

## What happens

- A **host** starts in the background and owns the model and a single request queue. The model keeps running between agent sessions, and nothing opens a window. See [How it works](/how-it-works/).
- The agent gets its connection settings through environment variables, flags or a generated file in the data folder. **Your own agent configuration is never edited.**
- Extra arguments go to the agent: `hearthwork codex exec "..."`, `hearthwork aider --message "..."`.

## Look before you run

Add `--dry-run` as the first argument to print the exact command, the environment variables Hearthwork sets (keys masked) and the files it generates, without starting anything:

```sh
hearthwork opencode --dry-run
```

## Stop and check

```sh
hearthwork status    # host, model, context, slots, requests per client, budget
hearthwork stop      # save the prompt cache, then stop the model and the host
```

`hearthwork agents` lists which agents are installed, the API each uses, and the install command for the missing ones. For per-agent notes see the guides: [Claude Code](/guides/claude-code/), [Codex](/guides/codex/), [OpenCode](/guides/opencode/), [Qwen Code](/guides/qwen-code/), [Aider](/guides/aider/).

If something misbehaves, start with [Troubleshooting](/troubleshooting/).
