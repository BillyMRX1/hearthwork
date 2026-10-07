#!/usr/bin/env python3
"""Hearthwork check: will a local model for coding agents be worth it on this computer, and which one?

Run it without installing anything:
  uvx --from git+https://github.com/BillyMRX1/hearthwork hearthwork-check
or, once installed:  hearthwork check      (--json for scripts)

Read-only: it checks the hardware and installed agents, gives the same verdict Hearthwork's setup gives, and
looks up live file sizes on Hugging Face to pick the best version of each suggested model that fits.
"""
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .runtime import recommend
from .onboard import CYAN, GREEN, RED, RESET, WINDOWS, YELLOW, detect, judge, lmstudio_folders, memory_gb

BOLD, DIM = "\033[1m", "\033[2m"
REPO = "https://github.com/BillyMRX1/hearthwork"

# Coding models with tool-calling chat templates (checked on Hugging Face, 2026-10). "moe": only a few experts run
# per token, so the model stays usable when part of it sits in system RAM. Measured on the reference PC
# (RTX 5060 Ti 16 GB + 22.6 GB RAM, 64K context): MoE Qwen3-Coder-30B partly in RAM 18-31 tokens/s; dense
# Qwen3.8-27B partly in RAM ~6 tokens/s.
# "tested": Hearthwork benchmark (bench.py, 8 graded coding tasks) on the reference PC, Q4_K_M, 2026-10-07.
MODELS = [
    {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "name": "Qwen3.5-35B-A3B", "moe": True, "tested": True,
     "note": "best in the benchmark: 8/8 with Codex (2.6 min) and Claude Code (3.6 min)"},
    {"repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "name": "Qwen3-Coder-30B-A3B", "moe": True, "tested": True,
     "note": "8/8 with Claude Code (3.9 min) and Codex (4.2 min); a bit smaller"},
    {"repo": "unsloth/GLM-4.7-Flash-GGUF", "name": "GLM-4.7-Flash (30B MoE)", "moe": True, "tested": True,
     "note": "8/8 with Codex (4.5 min), 7.7/8 with Claude Code (missed one bug)"},
    {"repo": "ggml-org/gpt-oss-20b-GGUF", "name": "gpt-oss-20b", "moe": True, "note": "small and fast; good at tool use"},
    {"repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "name": "Devstral-Small-2-24B", "moe": False,
     "note": "coding agent (dense: needs to fit the GPU to be fast)"},
    {"repo": "unsloth/Qwen3.5-27B-GGUF", "name": "Qwen3.5-27B", "moe": False, "note": "dense"},
    {"repo": "unsloth/Qwen3-Coder-Next-GGUF", "name": "Qwen3-Coder-Next (80B MoE)", "moe": True,
     "note": "strongest here; needs a big machine"},
    {"repo": "lmstudio-community/Qwen3.5-9B-GGUF", "name": "Qwen3.5-9B", "moe": False,
     "note": "small machines only; much weaker with agents' tools"},
]
# Best quality first. Below Q3, answers get noticeably worse.
QUANTS = ["Q4_K_M", "UD-Q4_K_XL", "MXFP4", "Q4_K_S", "Q3_K_M", "UD-Q3_K_XL", "Q3_K_S", "UD-Q2_K_XL", "Q2_K"]
KV_GB_64K = 3.4  # KV cache for a 64K context at q8_0, measured with Qwen3-Coder-30B (varies by model)


def version(binary):
    path = shutil.which(binary)
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else "installed"
    except (OSError, subprocess.SubprocessError):
        return "installed"


def quant_sizes(repo):
    """{quant: GB} from Hugging Face (multi-part files summed); None when offline."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hearthwork-check"}), timeout=20) as r:
            files = json.load(r)
    except Exception:
        return None
    sizes = {}
    for f in files:
        path = f.get("path", "")
        if not path.endswith(".gguf") or "mmproj" in path.lower():
            continue
        name = re.sub(r"-\d{5}-of-\d{5}", "", Path(path).name)
        for quant in QUANTS:
            if name.upper().endswith(f"-{quant.upper()}.GGUF") or name.upper().endswith(f"_{quant.upper()}.GGUF"):
                sizes[quant] = sizes.get(quant, 0) + (f.get("lfs") or {}).get("size", f.get("size", 0)) / 2**30
    return sizes


def place(model, sizes, gpu_room, ram_room):
    """Best quantization for this machine -> (quant, GB, fit) with fit in fast / good / slow, or None if too big."""
    if model["moe"]:  # MoE stays fast partly in RAM: best quality that fits at all
        for quant in QUANTS:
            if quant in sizes and sizes[quant] <= gpu_room + ram_room:
                return quant, sizes[quant], "fast" if sizes[quant] <= gpu_room else "good"
        return None
    for quant in QUANTS:  # dense: fitting the GPU entirely matters most
        if quant in sizes and sizes[quant] <= gpu_room:
            return quant, sizes[quant], "fast"
    for quant in QUANTS:  # then GPU + RAM
        if quant in sizes and sizes[quant] <= gpu_room + ram_room:
            return quant, sizes[quant], "good" if model["moe"] else "slow"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hearthwork check", description="Will Hearthwork (local models for coding agents) be worth it here?")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    hw = detect()
    verdict, reasons, expect = judge(hw)
    runtime, runtime_why = recommend(hw)
    ram_total, ram_free = memory_gb()
    home_disk = shutil.disk_usage(Path.home()).free / 2**30
    agents = {"Claude Code": version("claude"), "Codex": version("codex")}
    vram = hw.get("vramGB") or 0
    if hw["backend"] == "metal":
        gpu_room, ram_room = max(0.0, hw["ramGB"] * 0.75 - 1.5 - KV_GB_64K), 0.0
    else:
        gpu_room = max(0.0, vram - 1.5 - KV_GB_64K) if hw["backend"] in ("cuda", "vulkan") else 0.0
        ram_room = max(0.0, hw["ramGB"] - 6)

    picks = []
    offline = False
    for model in MODELS:
        sizes = quant_sizes(model["repo"])
        if sizes is None:
            offline = True
            continue
        fit = place(model, sizes, gpu_room, ram_room)
        picks.append({**model, "quant": fit[0] if fit else None, "sizeGB": round(fit[1], 1) if fit else None,
                      "fit": fit[2] if fit else "too big"})
    rank = {"fast": 0, "good": 1, "slow": 2, "too big": 3}
    picks.sort(key=lambda p: (p["fit"] == "too big", not p.get("tested"), rank[p["fit"]], MODELS.index(next(m for m in MODELS if m["repo"] == p["repo"]))))

    if args.json:
        print(json.dumps({"hardware": hw, "ramFreeGB": round(ram_free, 1), "diskFreeGB": round(home_disk, 1),
                          "runtime": runtime, "runtimeReason": runtime_why, "verdict": verdict, "reasons": reasons, "expect": expect, "agents": agents,
                          "models": picks, "offline": offline}, indent=2))
        return

    print(f"\n{BOLD}Hearthwork check{RESET}  {DIM}(local models for Claude Code / Codex){RESET}\n")
    print(f"{BOLD}Your computer{RESET}")
    print(f"  OS       {platform.system()} {platform.release()} ({hw['arch']})   Python {platform.python_version()}")
    print(f"  CPU      {hw['cpuThreads']} threads")
    print(f"  RAM      {ram_total:.1f} GB ({ram_free:.1f} GB free now)")
    gpus = ", ".join(hw["gpus"]) or "no usable GPU"
    print(f"  GPU      {gpus}" + (f"  ({vram:.1f} GB)" if hw["backend"] == "vulkan" and vram else "")
          + f"   ->  llama.cpp {hw['backend']}" + (f", driver CUDA {hw['cudaDriver']}" if hw.get("cudaDriver") else ""))
    print(f"  Runtime  {runtime}: {runtime_why}")
    print(f"  Disk     {home_disk:.0f} GB free in your home drive"
          + (f"; LM Studio models: {lmstudio_folders()[0]}" if lmstudio_folders() else ""))

    color = {"recommended": GREEN, "limited": YELLOW}.get(verdict, RED)
    print(f"\n{BOLD}Verdict{RESET}  {color}{BOLD}{verdict.upper()}{RESET}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"  {expect}")

    print(f"\n{BOLD}Coding agents{RESET}")
    for name, found in agents.items():
        print(f"  {name:12} " + (f"{GREEN}{found}{RESET}" if found else f"{YELLOW}not installed{RESET}"))
    if not any(agents.values()):
        print(f"  {YELLOW}Install at least one: Claude Code (https://code.claude.com) or Codex "
              f"(https://developers.openai.com/codex).{RESET}")

    print(f"\n{BOLD}Models for this computer{RESET}  {DIM}(best version that fits; sizes live from Hugging Face){RESET}")
    labels = {"fast": f"{GREEN}fits the GPU: fast{RESET}", "good": f"{GREEN}GPU + RAM: good (MoE){RESET}",
              "slow": f"{YELLOW}GPU + RAM: slow (dense){RESET}", "too big": f"{RED}too big{RESET}"}
    for p in picks:
        size = f"{p['quant']:<10} {p['sizeGB']:5.1f} GB" if p["quant"] else " " * 19
        tested = f" {CYAN}[tested]{RESET}" if p.get("tested") else ""
        print(f"  {p['name']:<28} {size}  {labels[p['fit']]}{tested}")
        print(f"  {DIM}{'':<28} {p['note']}{RESET}")
    if offline:
        print(f"  {YELLOW}Some models could not be looked up (offline?).{RESET}")
    print(f"  {DIM}Measured on an RTX 5060 Ti 16 GB + 22.6 GB RAM: MoE partly in RAM 18-31 tokens/s, "
          f"dense partly in RAM ~6 tokens/s.{RESET}")

    best = next((p for p in picks if p["fit"] in ("fast", "good")), None)
    print(f"\n{BOLD}Next step{RESET}")
    if verdict == "not recommended" or not best:
        print("  A local model would be too slow or too weak for coding on this computer. Cloud Claude Code / Codex is the"
              " better choice here.")
    else:
        print(f"  1. Install Hearthwork:  {CYAN}uv tool install hearthwork{RESET}   (or: pipx install hearthwork)")
        print(f"  2. Download a model:    {CYAN}hearthwork model {best['repo']}{RESET}   "
              f"(pick {best['quant']}, {best['sizeGB']} GB)")
        print(f"  3. From your project folder, run {CYAN}hearthwork{RESET}: setup takes a minute, then pick your agent.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
