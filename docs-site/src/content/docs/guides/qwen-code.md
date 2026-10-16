---
title: Qwen Code
description: Run Qwen Code on your local model with hearthwork qwen.
---

| | |
|---|---|
| Command | `hearthwork qwen` |
| Install | `npm i -g @qwen-code/qwen-code` |
| API used | OpenAI Chat Completions |

Run `hearthwork qwen` from your project folder. Extra arguments go to Qwen Code.

## Your settings are not changed

Hearthwork writes a settings file in the [data folder](/reference/data-and-config/) and points Qwen Code at it through `QWEN_CODE_SYSTEM_SETTINGS_PATH`. Your own Qwen Code settings are not edited.

## Good to know

Qwen Code runs a background memory extraction that makes extra requests to the model. With few slots those requests can queue behind your main conversation. [`hearthwork diagnose`](/troubleshooting/diagnose/) shows how much time went into waiting in the queue.

Preview exactly what would run with `hearthwork qwen --dry-run`.
