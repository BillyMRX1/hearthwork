---
title: Commands at a glance
description: One line for every Hearthwork command. The full flags are in the command reference.
---

For every flag and its exact help text, see the generated [command reference](/reference/cli/).

| Command | What it does |
|---|---|
| `hearthwork` | The menu. |
| `hearthwork claude\|codex\|opencode\|qwen\|aider [args]` | Run that agent on the local model. Add `--dry-run` first to show what would run. |
| `hearthwork agents` | Supported agents: installed, version, API, how each connects. |
| `hearthwork task "TASK"` | One coding task for another agent to delegate (no terminal needed). |
| `hearthwork mcp [install claude\|codex\|print]` | MCP server so any agent can use the local model as a subagent. |
| `hearthwork start [--model X] [--context N] [--force] [--over-budget]` | Start the model under the host, or switch it. |
| `hearthwork stop [--force]` | Save the prompt cache, then stop the model and the host. |
| `hearthwork serve [--model X]` | Run the host in this terminal with its log. Ctrl+C stops it. |
| `hearthwork idle [30\|off]` | Unload the model after N idle minutes (off by default). |
| `hearthwork budget [show\|use dedicated\|balanced\|work\|set ...\|reset]` | Resource budget. |
| `hearthwork context [model] [N\|auto]` | Context range per model; save a size for a model. |
| `hearthwork status [--json]` | Host, model, context, slots, requests per client, budget, switch, idle and recovery state. |
| `hearthwork logs [--server] [-f]` | The host's log. `--server` shows llama-server's output. |
| `hearthwork doctor [--agent X] [--quick] [--json]` | Compatibility check. Says which layer is at fault. |
| `hearthwork diagnose [--last N \| --since 30m] [--client X] [--json] [--export FILE]` | What limited your sessions, from measured numbers. |
| `hearthwork check [--json]` | Is this computer suited, and which models fit. |
| `hearthwork model <link>` | Download a GGUF model from Hugging Face. |
| `hearthwork profile save\|list\|show\|check\|apply\|export\|import` | Verified setup profiles. |
| `hearthwork share [--port N] [--allow-public] [--tailscale]` | Share the running model with other computers. |
| `hearthwork connect [host[:port]]` | Use the model of a computer that is sharing. |
| `hearthwork disconnect` | Back to local models. |
| `hearthwork devices [remove <name>]` | Trusted devices. `remove` revokes one. |
| `hearthwork api [--port N] [--show]` | OpenAI + Anthropic compatible endpoint on this computer. |
| `hearthwork alias add\|list\|remove` | Short commands for agents. |
| `hearthwork bench [--agent X] [--all] [--warmup] [--show]` | Graded coding tasks through an agent, with a scoreboard. |
| `hearthwork runtime [install <name>\|use <name>\|update\|rollback]` | llama.cpp engines (CUDA, ROCm, Vulkan, SYCL, Metal, CPU) side by side. |
| `hearthwork setup [--update-llama] [--reset] [--import FOLDER]` | Hardware check, llama.cpp download or update, models folder. |
| `hearthwork update` | Update Hearthwork itself. |
| `hearthwork --version` | Version and data folder. |
