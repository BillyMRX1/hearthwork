---
title: Model switching and idle unload
description: Switch models without stopping the host, and free memory when the model is idle.
---

## Switching

`hearthwork start --model X` while the host is running switches models without stopping the host:

1. New requests wait in the queue.
2. Running requests finish (up to 2 minutes).
3. The prompt cache is saved.
4. llama-server restarts with the new model (auto context).
5. The queue resumes.

It refuses while requests are running unless you add `--force`, which cancels them after 10 seconds (their clients get a clear error). Only one switch runs at a time. `hearthwork status` shows `switching to X (draining N)` and `loading X`.

## Idle unload

Idle unload is off by default.

```sh
hearthwork idle 30     # unload the model after 30 minutes without requests
hearthwork idle off    # turn it off
```

Unloading frees VRAM and RAM. The host keeps running, and the next request loads the model again. Clients just wait, and `status` says `loading`. The model is never unloaded while a request is running or waiting.

The `work` [budget preset](/guides/budgets/) turns idle unload on (15 minutes).
