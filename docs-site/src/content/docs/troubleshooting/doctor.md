---
title: Doctor
description: hearthwork doctor checks the server, the API protocol and each agent, and says which layer is at fault.
---

```sh
hearthwork doctor [--agent NAME | --all] [--quick] [--json] [--keep]
```

Doctor prints the environment (Hearthwork and llama.cpp versions, model, quantization, architecture, trained and running context, chat template fingerprint), then runs these stages and gives one verdict per agent.

1. **Server.** The model server answers and serves a model. On a remote host, it also checks that this device is still trusted.
2. **Protocol,** once per API the agents use, through the same relay: a plain reply, streaming, a tool call, a tool-result round trip, the next turn with tool history, and a late system message.
3. **Agent.** The agent is installed, its configuration builds, and it gives a headless reply in a disposable folder with its own temporary config. Your own config is never touched.
4. **Coding task** (skipped by `--quick`). The agent creates `calc.py` with `add(a, b)`, and Hearthwork imports it and checks the result itself.

## Verdicts

- `works`
- `configuration/connection problem at <stage>`
- `protocol problem at <stage> (<API>)`
- `protocol OK, but the model failed the coding task (model capability)`

Failed stages come with numbered suggestions. Aider never calls tools, so the tool stages do not count for it. The exit code is 0 when everything passes.

Each run is saved as JSON (no credentials) in `doctor/` in the [data folder](/reference/data-and-config/). [Setup profiles](/guides/setup-profiles/) use these reports as proof.

## Reading a verdict

| Verdict says | It means | Try |
|---|---|---|
| Problem at Server | The model is not running or not answering. | `hearthwork status`, then `hearthwork logs --server`. |
| Protocol problem | The relay and the model do not agree on the API. | Send the Doctor output in an issue; try another chat template or model. |
| Problem at Agent | The agent is missing or its launch failed. | `hearthwork agents`, then `hearthwork <agent> --dry-run`. |
| Model failed the coding task | Everything is connected, but the model is not capable enough. | Try a larger or different model. |
