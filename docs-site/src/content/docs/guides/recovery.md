---
title: Recovery
description: What Hearthwork does when the model server crashes or runs out of memory.
---

The host watches llama-server. If it crashes or runs out of memory, the host restarts it, at most 3 times per 15 minutes (after 2, 10 and 30 seconds). A server that is only busy or slow to answer `/health` is never restarted.

## What clients see

- A request refused before the model accepted it is retried once after the restart.
- A stream that breaks midway ends with the API's error event (Anthropic `event: error`, OpenAI error chunk). A second generation is never spliced on, and agent tasks are never rerun.

## After three restarts

The state becomes `failed: <reason>`. Requests get a clear 503 that names the reason and points to `hearthwork logs --server`. `hearthwork start` clears the failure and tries again.

`hearthwork status` (and `status --json`) shows the last failure, the exit code and the last 20 lines of the server log.

See also [Common errors](/troubleshooting/common-errors/) and [Logs](/troubleshooting/logs/).
