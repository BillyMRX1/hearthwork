#!/usr/bin/env python3
"""Run the llama.cpp server in this terminal (Ctrl+C stops it). hearthwork does the same in the background.

  start.bat / ./start.sh / python start.py   [--model <part of a file name>] [--context N] [--setup]
First run: setup (onboard.py). Then: pick a model (last used highlighted, Enter keeps it).
"""
import argparse
import sys

from onboard import CYAN, HERE, RESET, WINDOWS, load_config, onboard, setup_complete
from server import choose_and_remember, run_foreground


def main():
    parser = argparse.ArgumentParser(description="Start llama.cpp in this terminal with a model you pick.")
    parser.add_argument("--model", help="part of a model file name; skips the menu")
    parser.add_argument("--context", type=int, help="context size (default: chosen by setup)")
    parser.add_argument("--setup", action="store_true", help="re-run setup (hardware, llama.cpp, models folder)")
    args = parser.parse_args()
    config = load_config()
    if args.setup or not setup_complete(config):
        config = onboard(config)
    model = choose_and_remember(config, args.model)
    if not model:
        sys.exit(1)
    context = args.context or config["server"]["context"]
    ext = "bat" if WINDOWS else "sh"
    print(f"\nStarting {model.name}   context: {context}   port: {config['server']['port']}")
    print(f"{CYAN}When it says 'listening on', run from your project:  {HERE / ('claude.' + ext)}  or  "
          f"{HERE / ('codex.' + ext)}{RESET}\n", flush=True)
    run_foreground(config, model, context)


if __name__ == "__main__":
    main()
