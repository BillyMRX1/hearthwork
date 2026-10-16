---
title: How it works
description: One host per computer owns the model and a single queue; a relay sits between your agents and llama.cpp.
---

![How Hearthwork works: menu, first-run setup, model folder, llama-server using GPU and RAM, and a relay between the agents and the model](/how-it-works.svg)

## One host per computer

`hearthwork start`, or any command that needs the model, starts the Hearthwork host in the background. The host owns llama-server and the single relay and request queue that every client goes through: agents, `task` and MCP, `hearthwork api` and `hearthwork share`.

- It listens on `127.0.0.1:8470`. Each agent session identifies itself with a token like `local-claude-ab12`.
- The model keeps running between agent sessions, and nothing opens a window.
- A model started by an older Hearthwork is adopted, not restarted.

## The pieces

- **[The relay](/how-it-works/relay/):** the queue, request cleanup and tool-call repair.
- **[Context and models](/how-it-works/context-and-models/):** how context is sized, how layers are placed on the GPU, and how models are recommended.
- **Faster first messages.** Each agent's processed system prompt is saved to `cache/slots/` when a session ends or the server stops, and restored when the server starts.
- **Settings** live in the `server` section of `config.json`. `hearthwork setup` recalculates them, and `hearthwork start --model X` or `--context N` override a single run.
- **Recovery** keeps the model server running. See [Recovery](/guides/recovery/).
