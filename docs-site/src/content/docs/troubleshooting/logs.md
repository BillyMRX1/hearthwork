---
title: Logs and status
description: Read the host log and the model server's output, and what hearthwork status shows.
---

```sh
hearthwork logs              # the host's log: requests, queue, start and stop, repairs
hearthwork logs --server     # llama-server's own output
hearthwork logs -f           # follow the log as it grows
hearthwork logs -n 200       # show more lines
```

Use the host log to see how requests were queued and when tool calls were repaired. Use `--server` when the model failed to load or crashed: it holds llama.cpp's own messages, such as out-of-memory errors.

To run the host in a terminal and watch its log directly, use `hearthwork serve` (Ctrl+C stops it).

## Status

`hearthwork status [--json]` shows the host, model, context, slots, requests per client, the budget, switching, idle and recovery state, the sharing route and the active [profile](/guides/setup-profiles/). After a crash it shows the last failure, the exit code and the last 20 lines of the server log.

## Where the files are

The host log, `server.log`, `logs/requests.jsonl` and Doctor reports are all in your [data folder](/reference/data-and-config/).

## Debug dumps

Set `HEARTHWORK_RELAY_DUMP=<folder>` to write every request before and after the relay's cleanup. Set `HEARTHWORK_CAPTURE=1` to also record full request and response bodies. Both contain your code and conversations. Use them only locally and delete the files afterwards.
