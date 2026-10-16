#!/usr/bin/env python3
"""Hearthwork check: will a local model for coding agents be worth it on this computer, and which one?

Run it without installing anything:
  uvx --from git+https://github.com/BillyMRX1/hearthwork hearthwork-check
or, once installed:  hearthwork check      (--json for scripts)

Read-only: it checks the hardware and installed agents, gives the same verdict Hearthwork's setup gives, and
ranks suggested models for this computer's memory, using live file sizes and model headers from Hugging Face.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .runtime import recommend
from .models import best, describe, gather, rank
from .onboard import CYAN, GREEN, RED, RESET, WINDOWS, YELLOW, detect, judge, lmstudio_folders, memory_gb, settings_for

BOLD, DIM = "\033[1m", "\033[2m"
REPO = "https://github.com/BillyMRX1/hearthwork"

def version(binary):
    path = shutil.which(binary)
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else "installed"
    except (OSError, subprocess.SubprocessError):
        return "installed"


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
    candidates = gather()
    offline = any(c["files"] is None for c in candidates)
    picks = rank(candidates, hw, ram_free, settings_for(hw))
    top = best(picks)

    if args.json:
        print(json.dumps({"hardware": hw, "ramFreeGB": round(ram_free, 1), "diskFreeGB": round(home_disk, 1),
                          "runtime": runtime, "runtimeReason": runtime_why, "verdict": verdict, "reasons": reasons, "expect": expect, "agents": agents,
                          "models": picks, "recommended": top["repo"] if top else None, "offline": offline}, indent=2))
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

    print(f"\n{BOLD}Models for this computer{RESET}  {DIM}(ranked for this computer; sizes live from Hugging Face){RESET}")
    colors = {"fast": GREEN, "good": GREEN, "slow": YELLOW, "too big": RED}
    for p in picks:
        label, lines = describe(p)
        size = f"{p['quant']:<10} {p['sizeGB']:5.1f} GB" if p["quant"] else " " * 19
        star = f" {CYAN}<- best for this computer{RESET}" if top and p["repo"] == top["repo"] else ""
        print(f"  {p['name']:<28} {size}  {YELLOW if p.get('lowQuality') else colors[p['fit']]}{label}{RESET}{star}")
        for line in lines:
            print(f"  {DIM}{'':<28} {line}{RESET}")
    if offline:
        print(f"  {YELLOW}Some models could not be looked up (offline?). Context memory is estimated conservatively.{RESET}")

    print(f"\n{BOLD}Next step{RESET}")
    if verdict == "not recommended" or not top:
        print("  A local model would be too slow or too weak for coding on this computer. Cloud Claude Code / Codex is the"
              " better choice here.")
    else:
        print(f"  1. Install Hearthwork:  {CYAN}uv tool install hearthwork{RESET}   (or: pipx install hearthwork)")
        print(f"  2. Download a model:    {CYAN}hearthwork model {top['repo']}{RESET}   "
              f"(pick {top['quant']}, {top['sizeGB']} GB)")
        print(f"  3. From your project folder, run {CYAN}hearthwork{RESET}: setup takes a minute, then pick your agent.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
