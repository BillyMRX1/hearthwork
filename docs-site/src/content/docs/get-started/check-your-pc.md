---
title: Check your PC
description: Find out if your computer is suited for a local coding model, and which models fit, before installing anything.
---

Hearthwork runs a language model on your own GPU and RAM, so the first question is whether your computer is up to it. You can ask without installing anything:

```sh
uvx hearthwork check
```

If you already installed Hearthwork, run `hearthwork check` (add `--json` for machine-readable output).

The check shows your hardware (OS, CPU, RAM and GPU with its memory), a verdict, the coding agents it finds installed, and the best version of each suggested coding model that fits. Model sizes are read live from Hugging Face. Recommendations are computed from your computer's specs, not taken from a fixed list.

## The verdicts

| Verdict | Meaning |
|---|---|
| **Recommended** | A GPU with 12 GB or more and enough GPU memory plus RAM for a good coding model, or a Mac with 32 GB or more. |
| **Limited** | A smaller GPU or tight memory. Expect slower replies, or a smaller, weaker version of the model. Setup asks before continuing. |
| **Not recommended** | No GPU, integrated graphics only, or very little memory. Nothing is downloaded unless you agree. |

The thresholds are estimates. For real numbers on your own setup, run [the benchmark](/guides/benchmark/).

## What counts as a good model

Any GGUF model works if it supports tool calling (agents read, edit and run things through tools) and fits in memory. Mixture-of-experts (MoE) models stay fast even when part of them lives in RAM. Dense models slow down a lot once they spill out of GPU memory.

Next: [install Hearthwork](/get-started/install/).
