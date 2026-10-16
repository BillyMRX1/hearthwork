---
title: Setup profiles
description: Save a setup that works together with the proof, then re-check, export and apply it.
---

A profile captures a setup that works, together with the proof:

- the model's identity (file, size, fingerprint of the first 16 MB, quantization, architecture, chat template)
- the runtime and llama.cpp build
- installed agent versions
- the effective server settings (context, KV type, slots, batch, fit target, any budget preset)
- a hardware summary
- the latest Doctor and benchmark results

Profiles live in `profiles/` in the [data folder](/reference/data-and-config/).

```sh
hearthwork profile save [NAME] [--doctor]  # default name "default"; runs doctor --quick if no report from the last 24 h
hearthwork profile list | show NAME
hearthwork profile check [NAME] [--run]    # --run re-runs Doctor; a pass makes the profile verified again
hearthwork profile apply NAME              # model, context/slots/batch/KV and runtime build; Doctor validates afterwards
hearthwork profile apply previous          # the settings from before the last apply
hearthwork profile export NAME FILE        # no credentials, no device keys, paths replaced by placeholders
hearthwork profile import FILE             # stored as unverified; lists exactly what differs on this computer
```

## Status

| Status | When |
|---|---|
| **verified** | Doctor said "works" for every recorded agent. |
| **needs recheck** | The llama.cpp build, an agent version, the chat template or the model file changed. Run `profile check --run`. |
| **unverified** | Imported, or never checked. |

`hearthwork status` shows `profile: NAME (verified|needs recheck|unverified)`.

## Safe to try

`apply` saves your current settings as the `previous` profile first, and restores them automatically if validation fails. Agent versions are only reported, never changed. See [Doctor](/troubleshooting/doctor/) for what validation runs.
