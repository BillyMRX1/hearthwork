"""The `hearthwork` menu: pick a coding agent, and it starts the local model for you.

Run `hearthwork` from the project you want the agent to work on (`hearthwork claude` / `hearthwork codex` skip the
menu). The model server runs in the background (its own window on Windows) and stays up between agent sessions
until you stop it from the menu or on quit.
"""
import os

from .harnesses import HARNESSES, installed, launch
from .onboard import CYAN, GREEN, RED, RESET, YELLOW, ask, find_models, onboard, yes
from .paths import BENCH, STATE
from .server import choose_and_remember, served_model, start_background, stop

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
        print(f"{YELLOW}No models yet. Download one first: menu option 'Download a model', or "
              f"`hearthwork model <Hugging Face link>`.{RESET}")
        return None
    model = choose_and_remember(config)
    if not model or not start_background(config, model):
        return None
    return served_model(port)


def run_agent(config, key, args=(), back_to_menu=True):
    """Start agent `key` in this terminal and folder, starting the model first if needed. Returns its exit code."""
    name = ensure_server(config)
    if not name:
        return 1
    title = HARNESSES[key]["title"]
    after = " Exit it to come back to this menu." if back_to_menu else ""
    print(f"\n{CYAN}Starting {title} in {os.getcwd()} with {name}.{after}{RESET}\n")
    return launch(key, config["server"]["port"], name, config["server"]["context"], args=args)


def switch_model(config):
    model = choose_and_remember(config)
    if model:
        start_background(config, model)


def download_model():
    from . import model
    link = ask("Hugging Face link or org/repo (Enter to cancel): ")
    if link:
        model.main([link])


def benchmark(config):
    """8 graded coding tasks on the running model (starting one if needed), through an agent you pick."""
    from . import bench
    if not ensure_server(config):
        return
    agents = [key for key in HARNESSES if installed(key)]
    if not agents:
        print(f"{YELLOW}No coding agent is installed.{RESET}")
        return
    key = agents[0]
    if len(agents) > 1:
        names = " / ".join(f"{i}) {HARNESSES[k]['title']}" for i, k in enumerate(agents, 1))
        answer = ask(f"Benchmark through which agent? {names} / a) all [Enter = 1]: ")
        if answer.lower() == "a":
            print(f"{DIM}Release check: each agent in turn after a warmup; 10-30 minutes.{RESET}")
            return bench.main(["--all"])
        if answer.isdigit() and 1 <= int(answer) <= len(agents):
            key = agents[int(answer) - 1]
    print(f"{DIM}About 5-15 minutes; the agent works in its own folder under {BENCH / 'runs'}.{RESET}")
    bench.main(["--agent", key])


def main(config):
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
