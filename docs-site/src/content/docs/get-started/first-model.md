---
title: Your first model
description: Download a GGUF model from Hugging Face and understand how Hearthwork sizes its context.
---

Download a model from Hugging Face with a repository name or a link:

```sh
hearthwork model unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF                    # a repo: lists files, you pick one
hearthwork model https://huggingface.co/<org>/<repo>                           # same, as a link
hearthwork model https://huggingface.co/<org>/<repo>/blob/main/<file>.gguf    # one file
```

- The repo list shows each file's size, smallest first, and whether it fits: in GPU memory (fastest), in GPU memory plus RAM (slower, fine for MoE models), or too big.
- Files go to `<models folder>/<org>/<repo>/`, which is LM Studio's layout. Multi-part models download all parts, and interrupted downloads resume when you run the command again.
- For gated or private repositories, set the `HF_TOKEN` environment variable.

## Context

With `context: auto`, Hearthwork reads the model file (trained context, layers, KV heads, attention layout) and, together with your GPU memory and RAM, works out a range for each model. For example: `128K (auto; range 64K-256K, model max 256K)`.

- The **minimum** is 64K, because agents' own prompts are large.
- The **recommended** size is what `auto` starts with, at most 128K.
- The **maximum** is what fits in GPU memory plus RAM.

`hearthwork context [model] [N|auto]` shows the range or saves a size for a model (`96K`, `98304`). The order of precedence is: `start --context` first, then the size saved for the model, then `server.context` in `config.json`, then `auto`.

## Custom chat templates

If a model's chat template is broken in some other way, put a fixed copy at `templates/<model file name without .gguf>.jinja` in the [data folder](/reference/data-and-config/).

Next: [run your first agent session](/get-started/first-session/).
