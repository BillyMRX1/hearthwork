#!/usr/bin/env python3
"""Download a GGUF model from Hugging Face into your models folder (the one set during setup).

  hearthwork model <link>

<link> can be:
  https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF                 a repository: pick a file
  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF                                        the same, short form
  https://huggingface.co/<org>/<repo>/blob/main/<file>.gguf   (or /resolve/...)    one file
Files go to <models folder>/<org>/<repo>/, LM Studio's layout, so LM Studio sees them too. Multi-part models
download all parts. Interrupted downloads resume when run again. For gated/private repos set HF_TOKEN.
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .onboard import CYAN, GREEN, RED, RESET, YELLOW, WINDOWS, ask, detect, download, load_config, memory_gb

SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def parse(link):
    """(repo, revision, file path or None)."""
    link = link.strip().strip('"')
    match = re.match(r"(?:https?://)?(?:www\.)?(?:hf\.co|huggingface\.co)/(.+)", link)
    path = match.group(1) if match else link
    parts = path.strip("/").split("/")
    if len(parts) < 2:
        sys.exit(f"{RED}Not a Hugging Face repo or file link: {link}{RESET}")
    repo = "/".join(parts[:2])
    if len(parts) >= 5 and parts[2] in ("blob", "resolve", "tree"):
        file = "/".join(parts[4:]) or None
        return repo, parts[3], file if file and file.endswith(".gguf") else None
    return repo, "main", None


def api(url):
    headers = {"User-Agent": "hearthwork"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        hint = " (gated or private: accept its license on the website and set HF_TOKEN)" if error.code in (401, 403) else ""
        sys.exit(f"{RED}Hugging Face said {error.code} for {url}{hint}{RESET}")


def gguf_choices(repo, revision):
    """[(label, [paths], total bytes)]: single files and multi-part models; vision projectors excluded."""
    files = api(f"https://huggingface.co/api/models/{repo}/tree/{urllib.parse.quote(revision)}?recursive=true")
    sizes = {f["path"]: (f.get("lfs") or {}).get("size", f.get("size", 0)) for f in files if f.get("type") == "file"}
    groups = {}
    for path, size in sizes.items():
        if not path.endswith(".gguf") or "mmproj" in path.lower():
            continue
        key = SHARD.sub(".gguf", path)
        groups.setdefault(key, []).append(path)
    choices = [(key, sorted(paths), sum(sizes[p] for p in paths)) for key, paths in groups.items()]
    return sorted(choices, key=lambda c: c[2])


def fit_note(size_gb, vram, ram_free, server):
    # GPU room for weights: VRAM minus what stays free and the KV cache (~3 GB per 64K context at q8_0;
    # it varies by model, so this is an estimate).
    gpu_room = vram - server.get("fitTargetMiB", 1536) / 1024 - 3 * server.get("context", 65536) / 65536
    if vram and size_gb <= gpu_room:
        return f"{GREEN}fits in GPU memory (fastest){RESET}"
    if size_gb <= max(gpu_room, 0) + ram_free * 0.8:
        return f"{YELLOW}GPU + RAM (slower; fine for MoE models){RESET}"
    return f"{RED}too big for this computer{RESET}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hearthwork model", description="Download a GGUF model from Hugging Face into your models folder.")
    parser.add_argument("link", nargs="?", help="Hugging Face repo or .gguf file link")
    args = parser.parse_args(argv)
    config = load_config()
    folder = config.get("modelsDir")
    if not folder or not Path(folder).is_dir():
        sys.exit(f"No models folder set yet. Run `hearthwork setup` first.")
    link = args.link or ask("Hugging Face link (repo or .gguf file): ")
    repo, revision, file = parse(link)

    choices = gguf_choices(repo, revision)
    if not choices:
        sys.exit(f"{RED}No .gguf files in {repo}. Look for a repo whose name ends in -GGUF.{RESET}")
    if file:
        picked = [c for c in choices if file in c[1] or SHARD.sub(".gguf", file) == c[0]]
        if not picked:
            sys.exit(f"{RED}{file} is not in {repo}.{RESET}")
        label, paths, total = picked[0]
    else:
        hw = config.get("hardware") or detect()
        vram, ram_free = hw.get("vramGB") or 0, memory_gb()[1]
        print(f"\n{repo}  (this computer: {vram:.0f} GB GPU memory, {ram_free:.0f} GB RAM free)")
        for i, (label, paths, total) in enumerate(choices, 1):
            parts = f" ({len(paths)} parts)" if len(paths) > 1 else ""
            note = fit_note(total / 2**30, vram, ram_free, config.get("server", {}))
            print(f"  {i:2}) {total / 2**30:6.1f} GB  {Path(label).name}{parts}   {note}")
        while True:
            answer = ask("\nWhich file? (number, Enter to cancel) ")
            if not answer:
                return
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                label, paths, total = choices[int(answer) - 1]
                break

    target_dir = Path(folder) / repo.split("/")[0] / repo.split("/")[1]
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {Path(label).name} ({total / 2**30:.1f} GB) to {target_dir}")
    for path in paths:
        target = target_dir / Path(path).name
        if target.exists():
            print(f"  {target.name}: already downloaded")
            continue
        url = f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision)}/{urllib.parse.quote(path)}"
        download(url, target, Path(path).name)
    print(f"{GREEN}Done.{RESET} Start it with {CYAN}hearthwork{RESET} (or `hearthwork start`) and pick it from the list.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
