---
title: API mode
description: One local endpoint that speaks both the OpenAI and the Anthropic API, for tools without a Hearthwork launcher.
---

`hearthwork api` is one local endpoint that speaks both the OpenAI and the Anthropic API. Use it for tools Hearthwork has no launcher for: the official SDKs, Continue, your own scripts. It applies the same request cleanup as the launchers.

```sh
hearthwork api                      # leave it running; Ctrl+C stops it
hearthwork api --show               # copy-paste setup for each tool
```

## Endpoints

- **OpenAI:** base URL `http://127.0.0.1:8080/v1`. Supports `chat/completions`, `responses`, `completions`, `models`, and `embeddings` if the model supports it.
- **Anthropic:** base URL `http://127.0.0.1:8080`. Supports `v1/messages` and `v1/messages/count_tokens`.
- Streaming works.

## Options

- **This computer only** (127.0.0.1). For other devices use `hearthwork share`.
- `--port N` changes the port.
- `--api-key KEY` requires a key (`Authorization: Bearer` or `x-api-key`). Without it, any program on this computer can use the model.
- `--allow-switch`: a request whose `"model"` matches another local model file switches to it through the host (see [Model switching](/guides/model-switching/)). It is off by default because switching interrupts everything else using the model.
- `--show` prints setup for the OpenAI and Anthropic Python SDKs, curl, environment variables, Aider, OpenCode, Continue and Cursor. Cursor may not work with localhost.

## Models

If no model is running, the last used one is started. `/v1/models` lists the running model first, then every local model. Any model name in a request uses the running model.

Replies to `/v1/responses` without `max_output_tokens` are capped at 8192 tokens. Other endpoints use your values unchanged.
