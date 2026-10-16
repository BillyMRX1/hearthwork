---
title: Diagnose
description: hearthwork diagnose explains why sessions were slow or failed, using numbers the host recorded.
---

`hearthwork diagnose` reads the numbers the host records for every generation request. It does this passively: streams are never buffered or delayed.

What is recorded: client, API, outcome, queue wait, time to first byte, duration, prompt, cached and generated tokens, speeds, and the request's size against the context. The numbers go to `logs/requests.jsonl` in the [data folder](/reference/data-and-config/) (rotated at about 5 MB). **No prompt or response text is recorded, and nothing leaves your computer.**

```sh
hearthwork diagnose                      # everything recorded
hearthwork diagnose --since 30m --client opencode
hearthwork diagnose --last 50 --json
hearthwork diagnose --export report.json # numbers only: no prompts, code, credentials, paths or device names
```

## How to read the report

The report separates two kinds of findings:

- **Observed** findings are backed by the numbers, for example time spent waiting in the queue because several clients shared the slots.
- **Possible causes** are explanations the numbers are consistent with. They are labelled as such.

Each finding comes with a suggestion. What the backend does not report is shown as "unavailable". Model load time, memory use and time in agent tools are not measured.

## Opt-in raw capture

For debugging, set `HEARTHWORK_CAPTURE=1` before starting the host to also write full request and response bodies (your code and conversations) to `logs/capture.jsonl`. It is a local file, never exported, and off by default. Delete it when you are done.
