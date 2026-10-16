---
title: Common errors
description: The exact messages Hearthwork prints, what they mean and how to fix them.
---

The messages below are copied from Hearthwork, so you can search this page for the text you see.

## Starting and stopping

### The model server failed: ...

> The model server failed: `<reason>`. See `hearthwork logs --server` for its output; `hearthwork start` tries again.

The host restarted llama-server three times in 15 minutes and gave up. Requests get this as an HTTP 503. Read `hearthwork logs --server` for the cause (often out of memory), free memory or pick a smaller model or context, then run `hearthwork start`. See [Recovery](/guides/recovery/).

### Port ... is in use but no model answers

> Port `<port>` is in use but no model answers. Stop it with `hearthwork stop` first.

Something is holding the model server's port. Run `hearthwork stop`. If it persists, close the program using that port.

### Another Hearthwork host is already running

> Another Hearthwork host is already running; this one exits without touching the model.

Only one host runs per computer. Use `hearthwork status` to see it. `hearthwork serve` prints a similar message and exits with code 1 when a host is already running.

### The model was not ready after N s

> The model was not ready after `<N>` s (see `hearthwork logs --server`).

Large models take a while to load. Check `hearthwork logs --server` to see whether it is still loading or has failed.

### Not stopping / Not switching: N request(s) are running or waiting

> Wait for them, or use `hearthwork stop --force`.

Hearthwork will not interrupt running requests unless you ask. Wait, or add `--force` (for a switch, `--force` cancels requests after 10 seconds).

## Memory and budget

### Not starting ... with the '...' budget

> Not starting `<model>` with the '`<preset>`' budget

The model would not fit in your configured budget. The message lists the numbers and options: a smaller quantization, an explicit smaller context (`hearthwork start --context 96K`), or another preset. `hearthwork start --over-budget` starts anyway. See [Budgets](/guides/budgets/).

## Requests

### The request doesn't fit the model's context

> `<N>` tokens don't fit the model's `<M>` context (per session): raise it with `hearthwork context <model> <N>` or start a new conversation.

The conversation outgrew the context. Hearthwork never truncates requests. Raise the context for that model or start a new conversation.

### The model is busy and N requests are already waiting

> The model is busy and N requests are already waiting. Try again in a minute.

The queue holds 16 waiting requests. Callers get an overload error in their own API format (Anthropic `overloaded_error`, HTTP 529; OpenAI-style 503 with `Retry-After`). Reduce parallel agents or raise the number of slots with a [budget](/guides/budgets/).

### The request waited N s for a free slot of the model

> The request waited N s for a free slot of the model. Try again later.

A request waited the maximum time (10 minutes) in the queue.

### The Hearthwork host is shutting down

The host was stopped while the request was waiting. Start it again with any agent command or `hearthwork start`.

### The model is still starting; try again in a moment

The model is loading. Wait for `hearthwork status` to show it ready.

## Agents

### ... needs a terminal to talk to you

> `<agent>` needs a terminal to talk to you. Running inside another agent? Use: `hearthwork task --agent <agent> "what to do"`

You started an interactive agent without a terminal. Use [`hearthwork task`](/guides/subagents/) or the MCP server.

### ... is not installed

> `<Agent>` is not installed. Install: `<command>`

Install the agent. `hearthwork agents` lists the install command for each one.

## Sharing and pairing

### Not sharing: this computer is on a Public network

> Not sharing: this computer is on a Public network (...). Sharing uses plain HTTP, so it is only for your own home network.

Mark the network as Private (Windows Settings > Network & internet > your network > Network profile type > Private network), or run `hearthwork share --allow-public` if you are sure. With `--tailscale`, only Tailscale computers can connect on a Public network.

### Wrong PIN

> Wrong PIN.

You have 3 tries per pairing request. The PIN lasts 2 minutes.

### Pairing is locked for a few minutes after too many wrong PINs

> Pairing is locked for a few minutes after too many wrong PINs.

Wait 10 minutes. Then run `hearthwork connect` again. See [Use the model from another computer](/guides/other-computer/).

### Discovery finds nothing

Discovery uses UDP port 8485. Use `hearthwork connect <ip>[:8484]` with the host's address. Check the host's firewall allows Python on private networks.

## Still stuck

Run [`hearthwork doctor`](/troubleshooting/doctor/) and look at [`hearthwork logs`](/troubleshooting/logs/).
