---
title: Context sizing and model choice
description: How Hearthwork sizes the context, places the model on your GPU, and recommends models from your specs.
---

## Context sizing

The context is computed per model and per computer. Hearthwork reads the model file (trained context, layers, KV heads, attention layout) and combines it with your GPU memory and RAM into a range: a minimum of 64K (agents' own prompts are large), a recommended size (what `auto` starts with, at most 128K), and a maximum that fits in GPU memory plus RAM. See [Your first model](/get-started/first-model/) for the commands.

A prompt that does not fit gets a clear error, and requests are never truncated.

## GPU first

llama.cpp's `--fit` puts as much of the model on the GPU as fits, keeping some free for the desktop, and runs the rest from RAM, on every start. The prompt batch is 4096 on GPUs with 12 GB or more, with 2 slots so background requests do not evict the conversation's cache.

Mixture-of-experts (MoE) models stay fast when partly in RAM. Dense models slow down a lot once they spill out of GPU memory.

## How models are recommended

`hearthwork check` computes recommendations from your computer's specs rather than from a fixed list. It reads the hardware (OS, CPU, RAM, GPU and its memory), gets the real file sizes of suggested coding models from Hugging Face, and shows the best version of each that fits: in GPU memory (fastest), in GPU memory plus RAM, or too big. The verdict thresholds are estimates (see [Check your PC](/get-started/check-your-pc/)). Your own numbers come from the [benchmark](/guides/benchmark/).

## Resource estimates

The estimated need of a running model is its weights plus the KV cache for its context plus about 1.5 GB of runtime. [Budgets](/guides/budgets/) compare this with what you allow and with what is free now.
