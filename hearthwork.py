#!/usr/bin/env python3
"""Hearthwork: one menu for everything. Pick a coding agent, and it starts the local model for you.

Run it from the project you want the agent to work on:
  C:\\path\\to\\hearthwork\\hearthwork.bat  (Windows)
  /path/to/hearthwork/hearthwork.sh       (macOS/Linux)
  ... hearthwork.bat claude / codex      (skip the menu and go straight to that agent)

First run: setup (hardware check, verdict, llama.cpp, models folder). The model server runs in the background
(its own window on Windows) and stays up between agent sessions until you stop it from the menu or on quit.
"""
import os
import sys

from harnesses import HARNESSES, installed, launch
from onboard import CYAN, GREEN, HERE, RED, RESET, YELLOW, ask, find_models, load_config, onboard, setup_complete, yes
from server import STATE, choose_and_remember, served_model, start_background, stop

BOLD, DIM = "\033[1m", "\033[2m"


def status_lines(config):
    hw, port = config.get("hardware", {}), config["server"]["port"]
    running = served_model(port)
    gpu = ", ".join(hw.get("gpus") or []) or "CPU only"
    verdict = hw.get("verdict", "")
    color = {"recommended": GREEN, "limited": YELLOW}.get(verdict, RED)
    lines = [f"  Project:  {os.getcwd()}",
             f"  Model:    " + (f"{GREEN}running{RESET}  {running}  (port {port})" if running
                               else f"{DIM}not running{RESET}  (last used: {os.path.basename(config.get('lastModel') or '-')})"),
             f"  Computer: {gpu}, {hw.get('ramGB', '?')} GB RAM" + (f"  {color}{verdict}{RESET}" if verdict else "")]
    return running, lines


def ensure_server(config):
    """The running model's name, starting one (after picking a model) if none is running."""
    port = config["server"]["port"]
    name = served_model(port)
    if name:
        return name
    if not find_models(config["modelsDir"]):
        print(f"{YELLOW}No models yet. Choose 'Download a model' first.{RESET}")
        return None
    model = choose_and_remember(config)
    if not model or not start_background(config, model):
        return None
    return served_model(port)


def run_agent(config, key):
    name = ensure_server(config)
    if not name:
        return
    title = HARNESSES[key]["title"]
    print(f"\n{CYAN}Starting {title} in {os.getcwd()} with {name}. Exit it to come back to this menu.{RESET}\n")
    launch(key, config["server"]["port"], name, config["server"]["context"])


def switch_model(config):
    model = choose_and_remember(config)
    if model:
        start_background(config, model)


def download_model():
    import subprocess
    link = ask("Hugging Face link or org/repo (Enter to cancel): ")
    if link:
        subprocess.call([sys.executable, str(HERE / "model.py"), link])


def benchmark(config):
    """Run bench.py (8 graded coding tasks) on the running model, starting one if needed."""
    import subprocess
    if not ensure_server(config):
        return
    agents = [key for key in HARNESSES if installed(key)]
    if not agents:
        print(f"{YELLOW}No coding agent is installed.{RESET}")
        return
    key = agents[0]
    if len(agents) > 1:
        names = " / ".join(f"{i}) {HARNESSES[k]['title']}" for i, k in enumerate(agents, 1))
        answer = ask(f"Benchmark through which agent? {names} [Enter = 1]: ")
        if answer.isdigit() and 1 <= int(answer) <= len(agents):
            key = agents[int(answer) - 1]
    print(f"{DIM}About 5-15 minutes; the agent works in its own folder under bench/runs.{RESET}")
    subprocess.call([sys.executable, str(HERE / "bench.py"), "--agent", key])


def main():
    config = load_config()
    if not setup_complete(config):
        config = onboard(config)
        input("\nPress Enter to continue...")
    if len(sys.argv) > 1 and sys.argv[1] in HARNESSES:
        run_agent(config, sys.argv[1])
    while True:
        running, lines = status_lines(config)
        print(f"\n{BOLD}=== Hearthwork: local models for your coding agents ==={RESET}")
        print("\n".join(lines))
        options = []
        for key, harness in HARNESSES.items():
            note = "" if installed(key) else f"  {DIM}(not installed: {harness['install']}){RESET}"
            options.append((harness["title"] + note, lambda k=key: run_agent(config, k)))
        options += [("Start / switch model", lambda: switch_model(config)),
                    ("Download a model", download_model),
                    ("Stop the model server" + ("" if running else f"  {DIM}(not running){RESET}"), lambda: stop(config)),
                    ("Benchmark the model: 8 graded coding tasks + scoreboard", lambda: benchmark(config)),
                    ("Setup: hardware check, llama.cpp update, models folder", lambda: config.update(onboard(config)))]
        print()
        for i, (label, _) in enumerate(options, 1):
            print(f"  {i}) {label}")
        print("  q) Quit")
        choice = ask("\nChoose: ").lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            try:
                options[int(choice) - 1][1]()
            except SystemExit:  # setup skipped / model download cancelled: back to the menu
                pass
        else:
            print(f"{RED}Type a number from the list, or q.{RESET}")
    if STATE.exists() and served_model(config["server"]["port"]) and yes("Stop the model server?", default=True):
        stop(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
