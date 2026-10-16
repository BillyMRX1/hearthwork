---
title: Install and update
description: Install Hearthwork with uv or pipx, what first-run setup does, and how to update.
---

Hearthwork works on Windows, macOS and Linux. It needs Python 3.9 or newer and the agent(s) you want to use. It has no extra Python packages.

```sh
uv tool install hearthwork
```

or `pipx install hearthwork`. To get the latest code from GitHub instead:

```sh
uv tool install git+https://github.com/BillyMRX1/hearthwork
```

This gives you one command, `hearthwork`.

## Update, version, setup again

- **Update:** `hearthwork update` (or `uv tool upgrade hearthwork`).
- **Version and data folder:** `hearthwork --version`.
- **Setup again:** `hearthwork setup`. Add `--update-llama` to get the newest llama.cpp, or `--reset` to start from scratch.

## First-run setup

Running `hearthwork` (or `hearthwork start`) starts setup automatically when nothing is configured. Setup:

1. **Checks the computer:** OS, CPU, RAM and GPU (NVIDIA, AMD, Intel, Apple) with its memory. For AMD and Intel it reads the GPU's own memory, so integrated graphics are not mistaken for a real GPU.
2. **Judges whether a local model is worth it here** and says why (see [Check your PC](/get-started/check-your-pc/)).
3. **Gets llama.cpp:** one already in `bin/` or on your PATH, otherwise the newest official build that matches the hardware: CUDA for NVIDIA (13 or 12, chosen from the driver), ROCm for AMD RDNA2 or newer (otherwise Vulkan), Vulkan or SYCL for Intel, Metal on macOS, CPU if there is no GPU. Builds live side by side in `bin/<runtime>-b<build>/` (see `hearthwork runtime`). A new build is smoke-tested before use, and the previous one is kept for `hearthwork runtime rollback`.
4. **Asks for the models folder.** It finds LM Studio's folder and offers it, so both tools share the same files.
5. **Picks settings for this hardware:** context `auto`, prompt batch size, VRAM headroom and an 8-bit KV cache.
6. **Saves everything** to `config.json` in the [data folder](/reference/data-and-config/).

## Upgrading from the clone version

Earlier versions ran from a cloned folder with `.bat` and `.sh` scripts. Those are gone. On first run Hearthwork offers to import an old setup it finds, or you can run `hearthwork setup --import <old folder>`. It copies your settings, llama.cpp, prompt caches and benchmark results, and leaves the old folder as it is.

Next: [download your first model](/get-started/first-model/).
