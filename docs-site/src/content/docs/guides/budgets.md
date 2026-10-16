---
title: Resource budgets and presets
description: Keep room on your computer for gaming or work while the model runs.
---

`hearthwork budget` shows three separate things for each resource:

| Column | Meaning |
|---|---|
| **Configured** | The budget you or a preset set. |
| **Estimated** | The need of the running model: weights + KV cache for its context + about 1.5 GB runtime. |
| **Measured** | What is used now: llama-server's RAM from the OS, GPU memory from `nvidia-smi` when available, else "unavailable". |

`hearthwork status` ends with the same table.

## Presets

| Preset | Command | Effect |
|---|---|---|
| dedicated | `budget use dedicated` | Keeps 0.5 GB of VRAM free for other programs; gives the AI the most. |
| balanced (default) | `budget use balanced` | The setup's own settings. |
| work | `budget use work` | For gaming or work: keeps about 5 GB of VRAM free, 1 slot, half the CPU threads, unloads after 15 idle minutes. |
| custom | `budget set vram=10GB ram=12GB slots=1 threads=8 idle=20m` | Your own numbers on top of the current preset. `budget reset` returns to balanced. |

## What is enforced

- Slots (`-np`), threads (`-t`) and idle unload are llama.cpp flags or host settings.
- The VRAM budget becomes llama.cpp's `--fit-target` (best effort).
- The OS cannot cap RAM, so Hearthwork estimates the need before every start or switch and compares it with the budget and the RAM that is free now (the model being replaced counts as free). The default budget only warns about RAM. Presets and an explicit `ram=` are binding.

## `--over-budget` and `--force`

An impossible start is refused with the numbers and options (a smaller quantization, an explicit smaller context, another preset). The context is never shrunk silently. `hearthwork start --over-budget` starts anyway.

`--force` is separate: it interrupts running requests.

## Changing the budget

Changing the budget while a model runs restarts it through the host's switch path. `status` shows `pending: ...` when the saved settings differ from the running server, and `hearthwork start` applies them when nothing is running. On unified memory (Apple silicon), memory is counted once.
