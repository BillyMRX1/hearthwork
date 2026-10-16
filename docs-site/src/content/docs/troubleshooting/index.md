---
title: Troubleshooting
description: Which Hearthwork command to use when something is slow, broken or confusing.
---

Start with the command that matches your problem:

| Command | Use it when |
|---|---|
| `hearthwork status` | You want the current state: host, model, context, slots, queue, route, profile, budget. |
| `hearthwork logs [--server] [-f]` | You want the host's log (requests, queue, start and stop, repairs) or llama-server's own output. |
| `hearthwork doctor` | An agent misbehaves and you want to know which layer is at fault. |
| `hearthwork diagnose` | Sessions were slow or failed and you want the measured reason. |

## Pages in this section

- [Doctor](/troubleshooting/doctor/): a compatibility check with one verdict per agent.
- [Diagnose](/troubleshooting/diagnose/): what limited your sessions, from measured numbers.
- [Logs](/troubleshooting/logs/): the host log and the model server's output.
- [Common errors](/troubleshooting/common-errors/): the exact messages, with causes and fixes.

## A good first sequence

1. `hearthwork status` to see whether the host and model are running and what state they are in.
2. `hearthwork doctor --quick` to find the layer at fault.
3. `hearthwork logs` (or `logs --server`) to read what happened.
4. Still stuck? Open an issue on [GitHub](https://github.com/BillyMRX1/hearthwork/issues) with the output of `hearthwork doctor` and, if you like, `hearthwork diagnose --export report.json` (numbers only, no prompts or code).
