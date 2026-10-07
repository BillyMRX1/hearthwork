"""The `hearthwork` command.

  hearthwork                    menu (run it from the project you want the agent to work on)
  hearthwork claude [args...]   Claude Code with the local model (starts one if needed); args go to `claude`
  hearthwork codex [args...]    Codex with the local model; args go to `codex`
  hearthwork start [--model X]  start the model in the background     hearthwork stop
  hearthwork serve [--model X]  run the model server in this terminal (Ctrl+C stops it)
  hearthwork status             what is running
  hearthwork model <link>       download a GGUF model from Hugging Face
  hearthwork bench              8 graded coding tasks through an agent, with a scoreboard
  hearthwork check              is this computer suited, and which models fit
  hearthwork setup              hardware check, llama.cpp download/update, models folder
  hearthwork update             update Hearthwork itself
"""
import shutil
import subprocess
import sys

from . import __version__
from .harnesses import HARNESSES
from .onboard import CYAN, GREEN, RESET, load_config, offer_import, onboard, setup_complete
from .paths import HOME

REPO = "git+https://github.com/BillyMRX1/hearthwork"
USAGE = __doc__.split("\n", 2)[2]


def configured(interactive=True):
    """config, running first-time setup (or importing an older setup) when needed."""
    config = load_config()
    if setup_complete(config):
        return config
    config = offer_import(config)
    if setup_complete(config):
        return config
    config = onboard(config)
    if interactive:
        input("\nPress Enter to continue...")
    return config


def model_args(argv, name):
    import argparse
    parser = argparse.ArgumentParser(prog=f"hearthwork {name}")
    parser.add_argument("--model", help="part of a model file name; skips the model menu")
    parser.add_argument("--context", type=int, help="context size (default: chosen by setup)")
    return parser.parse_args(argv)


def update():
    """Upgrade with whichever tool installed Hearthwork."""
    if shutil.which("uv") and "uv" in sys.executable.replace("\\", "/").split("/"):
        return subprocess.call(["uv", "tool", "upgrade", "hearthwork"])
    if shutil.which("pipx") and "pipx" in sys.executable.replace("\\", "/"):
        return subprocess.call(["pipx", "upgrade", "hearthwork"])
    print(f"Upgrade with the tool you installed Hearthwork with, e.g.\n  uv tool upgrade hearthwork\n"
          f"  pipx upgrade hearthwork\n  pip install --upgrade {REPO}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command, rest = (argv[0], argv[1:]) if argv else ("menu", [])
    try:
        if command in ("-h", "--help", "help"):
            print(USAGE)
        elif command in ("-V", "--version", "version"):
            print(f"hearthwork {__version__}  (data: {HOME})")
        elif command == "menu":
            from . import menu
            menu.main(configured())
        elif command in HARNESSES:  # every following argument belongs to the agent (e.g. -p "...")
            from .menu import run_agent
            sys.exit(run_agent(configured(), command, rest, back_to_menu=False))
        elif command in ("start", "serve"):
            from .server import choose_and_remember, run_foreground, start_background
            args = model_args(rest, command)
            config = configured()
            model = choose_and_remember(config, args.model)
            if not model:
                sys.exit(1)
            if command == "start":
                ok = start_background(config, model, args.context)
                if ok:
                    print(f"{CYAN}Now run `hearthwork claude` or `hearthwork codex` from your project.{RESET}")
                sys.exit(0 if ok else 1)
            print(f"\nStarting {model.name}   context: {args.context or config['server']['context']}   "
                  f"port: {config['server']['port']}")
            print(f"{CYAN}When it says 'listening on', run `hearthwork claude` or `hearthwork codex` "
                  f"from your project.{RESET}\n", flush=True)
            run_foreground(config, model, args.context)
        elif command == "stop":
            from .server import stop
            stop(load_config())
        elif command == "status":
            from .server import served_model
            config = load_config()
            port = config.get("server", {}).get("port", 8001)
            running = served_model(port)
            print(f"hearthwork {__version__}   data: {HOME}")
            print(f"model: {GREEN + running + RESET if running else 'not running'}" + (f"  (port {port})" if running else ""))
            print(f"models folder: {config.get('modelsDir', '-')}")
        elif command == "model":
            from . import model
            configured()
            model.main(rest)
        elif command == "bench":
            from . import bench
            if "--show" not in rest:
                from .menu import ensure_server
                if not ensure_server(configured()):
                    sys.exit(1)
            bench.main(rest)
        elif command == "check":
            from . import check
            check.main(rest)
        elif command == "setup":
            from . import onboard as setup
            setup.main(rest)
        elif command == "update":
            sys.exit(update())
        else:
            print(f"Unknown command: {command}\n\n{USAGE}")
            sys.exit(2)
    except KeyboardInterrupt:
        print()
        sys.exit(130)


def check_main():
    """`hearthwork-check`: the checker on its own, e.g. `uvx --from git+https://github.com/BillyMRX1/hearthwork hearthwork-check`."""
    from . import check
    check.main()
