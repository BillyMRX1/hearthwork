"""The `hearthwork` command.

  hearthwork                    menu (run it from the project you want the agent to work on)
  hearthwork claude [args...]   Claude Code with the local model (starts one if needed); args go to `claude`
  hearthwork codex [args...]    Codex with the local model; args go to `codex`
  hearthwork opencode|aider|qwen [args...]   OpenCode, Aider or Qwen Code with the local model
  hearthwork agents             supported agents: installed?, version, API, how Hearthwork connects each
  hearthwork <agent> --dry-run  show the command, environment and generated files, start nothing (also: task --dry-run)
  hearthwork task "TASK"        one coding task for another agent to delegate (no terminal needed); see --help
  hearthwork mcp [install ...]  MCP server so any agent can use the local model as a subagent
  hearthwork start [--model X] [--force] [--over-budget]   start the model (under the Hearthwork host); --force interrupts running requests, --over-budget starts even if the memory estimate exceeds the budget
  hearthwork idle [30|off]      unload the model after N idle minutes (off by default); the next request loads it
  hearthwork budget [show|use dedicated|balanced|work|set vram=10GB ram=12GB slots=1 threads=8 idle=20m|reset]   resource budget
  hearthwork stop [--force]     stop the model and the host (saves the prompt cache); --force even with requests running
  hearthwork serve [--model X]  run the host in this terminal with its log (Ctrl+C stops it)
  hearthwork logs [--server] [-f]   the host's log (requests, queue); --server: llama-server's output
  hearthwork alias add <name> <agent>   short command for it, e.g. ccl  (also: alias list, alias remove <name>)
  hearthwork context [model] [N|auto]   context per model: range for this computer; save N for a model
  hearthwork diagnose [--last N | --since 30m | --all] [--client X] [--json] [--export FILE]   what limited your sessions (measured; default last 24 h)
  hearthwork status [--json]    host, model, context, slots, requests per client; switching, idle and recovery state
  hearthwork share [--port N]   share the running model on your home network (Ctrl+C stops); a PIN pairs each computer
  hearthwork api [--port N]     OpenAI + Anthropic compatible endpoint on this computer for any tool (--show: setup snippets)
  hearthwork devices            trusted devices (devices remove <name> revokes one)
  hearthwork connect [host]     use the model of a computer that is sharing (finds it, or give host[:port]); pair with the PIN
  hearthwork disconnect         back to local models
  hearthwork model <link>       download a GGUF model from Hugging Face
  hearthwork bench [--agent X]  8 graded coding tasks through an agent, with a scoreboard
  hearthwork bench --all        release check: every installed agent in turn, after a warmup, with a summary table
  hearthwork tune [--time 30m] [--min-context N] [--agent X] [--quick-tasks N] [--dry-run] [--json]   try a few installed configurations, save the best tested one as a profile (nothing downloaded; the original model is restored)
  hearthwork doctor [--agent X] [--quick] [--json]   server, API protocol and agent checks; says which layer is at fault
  hearthwork check              is this computer suited, and which models fit
  hearthwork profile save|list|show|check|apply|export|import   verified setup profiles (model, runtime, settings + Doctor proof)
  hearthwork runtime            llama.cpp engines: list, install <name>, use <name>, update, rollback
  hearthwork setup              hardware check, llama.cpp download/update, models folder
  hearthwork update             update Hearthwork itself (stops a running model first, restarts it after)
"""
import shutil
import subprocess
import sys

from . import __version__, context as ctx, runtime
from .harnesses import HARNESSES
from .onboard import ConfigError, CYAN, GREEN, RESET, load_config, offer_import, onboard, save_config, setup_complete
from .paths import HOME

REPO = "git+https://github.com/BillyMRX1/hearthwork"
USAGE = __doc__.split("\n", 2)[2]
# commands whose `--help` must not load the config or touch the model, the network or running processes
HELP_FIRST = ("start", "serve", "bench", "tune", "model", "setup", "context", "menu", "update", "status", "stop", "disconnect", "agents")


def configured(interactive=True, remote_ok=False):
    """config, running first-time setup (or importing an older setup) when needed. With `remote_ok`, a computer
    that only uses another computer's model (hearthwork connect) needs no setup."""
    config = load_config()
    if remote_ok and config.get("remote"):
        return config
    if setup_complete(config):
        migrate(config)
        return config
    config = offer_import(config)
    if setup_complete(config):
        migrate(config)
        return config
    config = onboard(config)
    if interactive:
        input("\nPress Enter to continue...")
    return config


def migrate(config):
    """Adopt settings of older versions (runtime.migrate, context.migrate); saved if anything changed."""
    if any([runtime.migrate(config), ctx.migrate(config)]):
        save_config(config)


def model_args(argv, name):
    import argparse
    parser = argparse.ArgumentParser(prog=f"hearthwork {name}")
    parser.add_argument("--model", help="part of a model file name; skips the model menu")
    parser.add_argument("--context", type=ctx.parse_k, help="context for this start, e.g. 98304 or 96K (default: auto, see `hearthwork context`)")
    parser.add_argument("--force", action="store_true",
                        help="interrupt running requests")
    parser.add_argument("--over-budget", action="store_true",
                        help="start even if the memory estimate exceeds the budget")
    return parser.parse_args(argv)


def has_terminal():
    """Is stdin a real terminal? (On Windows NUL also claims to be one, so ask the console API.)"""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        import ctypes
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-10), ctypes.byref(mode)))
    return True


def _upgrade_cmd():
    """The tool command that upgrades this install, or None when no known installer tool owns it."""
    if shutil.which("uv") and "uv" in sys.executable.replace("\\", "/").split("/"):
        return ["uv", "tool", "upgrade", "hearthwork"]
    if shutil.which("pipx") and "pipx" in sys.executable.replace("\\", "/"):
        return ["pipx", "upgrade", "hearthwork"]
    return None


def _importable():
    """Can a fresh Python still import hearthwork? This process runs on the code loaded before the upgrade."""
    try:
        return subprocess.run([sys.executable, "-c", "import hearthwork"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def update():
    """Upgrade with whichever tool installed Hearthwork. The running host process locks the program files on
    Windows (and a half-replaced install breaks everything), so it is stopped first and the model restarted after."""
    cmd = _upgrade_cmd()
    if cmd is None:
        print(f"Upgrade with the tool you installed Hearthwork with, e.g.\n  uv tool upgrade hearthwork\n"
              f"  pipx upgrade hearthwork\n  pip install --upgrade hearthwork")
        return 0
    from pathlib import Path
    from . import host
    config = load_config()
    running = host.status() is not None
    last = Path(config.get("lastModel") or "")
    stopped = False
    if running:
        print("Stopping the model to update (restarts after)")
        if not host.stop(config):  # plain stop: it refuses while requests are active and says why
            print("`hearthwork update` stopped: the model could not be stopped while requests are running. "
                  "Finish them (`hearthwork status` shows them), then update again. Nothing was changed and "
                  "the model is still running.")
            return 1
        stopped = True
    try:
        code = subprocess.call(cmd)
        why = None if code == 0 else f"the updater exited {code}"
    except Exception as error:  # the tool vanished or crashed: leave the install alone, just report
        code, why = 1, f"the updater did not run ({type(error).__name__}: {error})"
    if code and not _importable():
        print(f"Update stopped halfway ({why}) and Hearthwork can no longer be imported. Recovery:")
        print(f"  1. close other Hearthwork windows; if Python still runs from the {cmd[0]} tools folder, stop it "
              "(Task Manager, or taskkill /F /IM python.exe)")
        print(f"  2. reinstall:  {cmd[0]} tool install --force hearthwork" if cmd[0] == "uv"
              else f"  2. reinstall:  pipx install --force hearthwork")
        print("  3. start the model again:  hearthwork start")
        return 1
    if stopped:
        print("Update done." if not code else f"Update did not finish ({why}), but Hearthwork still works.")
        ok = host.ensure(config, last) if last.is_file() else False
        print(f"Model restarted ({last.name})." if ok else "Start the model again with `hearthwork start`.")
    else:
        print("Update done." if not code else f"Update did not finish ({why}), but Hearthwork still works.")
    return code


def help_first(command):
    """`hearthwork <command> --help`, without loading the config or starting anything: argparse prints its own help
    and exits 0; the others print their line of USAGE."""
    if command in ("start", "serve"):
        model_args(["--help"], command)
    elif command == "bench":
        from . import bench
        bench.main(["--help"])
    elif command == "tune":
        from . import tune
        tune.main(["--help"])
    elif command == "model":
        from . import model
        model.main(["--help"])
    elif command == "setup":
        from . import onboard as setup
        setup.main(["--help"])
    elif command == "menu":
        print(USAGE)
    else:
        print("\n".join(line for line in USAGE.splitlines() if line.split()[:2] == ["hearthwork", command]))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command, rest = (argv[0], argv[1:]) if argv else ("menu", [])
    try:
        if command in ("-h", "--help", "help"):
            print(USAGE)
        elif rest[:1] in (["-h"], ["--help"]) and command in HELP_FIRST:
            sys.exit(help_first(command))
        elif command in ("-V", "--version", "version"):
            print(f"hearthwork {__version__}  (data: {HOME})")
        elif command == "menu":
            from . import menu
            menu.main(configured(remote_ok=True))
        elif command == "task":
            from . import task
            sys.exit(task.main(rest))
        elif command == "mcp":
            from . import mcp
            sys.exit(mcp.main(rest))
        elif command == "agents":
            from .harnesses import agents_table
            print(agents_table())
        elif command in HARNESSES and rest[:1] == ["--dry-run"]:  # only first: the agent's own flags follow it
            from .harnesses import preview
            from .task import preview_model
            name, context, remote = preview_model(load_config())
            preview(command, name, context, args=rest[1:], remote=remote)
        elif command in HARNESSES:  # every following argument belongs to the agent (e.g. -p "...")
            if not rest and not has_terminal():
                print(f"{command} needs a terminal to talk to you. Running inside another agent? Use:\n"
                      f'  hearthwork task --agent {command} "what to do"      (or "-" to read the task from stdin)\n'
                      f"or add the MCP server once: hearthwork mcp install claude|codex|print\n"
                      f'To pass your own flags: hearthwork {command} ' + ('-p "prompt"' if command == "claude" else 'exec "prompt"'), file=sys.stderr)
                sys.exit(2)
            from .menu import run_agent
            sys.exit(run_agent(configured(remote_ok=True), command, rest, back_to_menu=False))
        elif command in ("start", "serve"):
            from . import host
            from .server import choose_and_remember
            args = model_args(rest, command)
            force, over_budget = args.force, args.over_budget
            config = configured()
            if command == "serve" and host.running():
                print("The Hearthwork host is already running. `hearthwork status` shows it; `hearthwork stop` stops it first.")
                sys.exit(1)
            model = choose_and_remember(config, args.model)
            if not model:
                sys.exit(1)
            if command == "start":
                ok = host.ensure(config, model, args.context, force=force, over_budget=over_budget)
                if ok:
                    print(f"{CYAN}Now run `hearthwork claude` or `hearthwork codex` from your project.{RESET}")
                sys.exit(0 if ok else 1)
            print(f"\nStarting {model.name}   host: 127.0.0.1:{host.port(config)}   model server: {config['server']['port']}")
            print(f"{CYAN}When the log says 'Model ready', run `hearthwork claude` or `hearthwork codex` from your "
                  f"project. Ctrl+C stops the host and the model.{RESET}\n", flush=True)
            sys.exit(host.serve(config, model, args.context, console=True))
        elif command == "host":  # hidden: the host process itself, started by host.ensure()
            from . import host
            sys.exit(host.host_main(rest))
        elif command == "logs":
            from . import host
            sys.exit(host.logs_main(rest))
        elif command == "diagnose":
            from . import diagnostics
            sys.exit(diagnostics.main(rest))
        elif command == "budget":
            from . import budgets
            sys.exit(budgets.main(rest))
        elif command == "idle":
            from . import host
            sys.exit(host.idle_main(rest))
        elif command == "stop":
            from . import host
            sys.exit(0 if host.stop(load_config(), force="--force" in rest) else 1)
        elif command == "status":
            from . import host
            config = load_config()
            port = config.get("server", {}).get("port", 8001)
            if "--json" not in rest:
                print(f"hearthwork {__version__}   data: {HOME}")
            if config.get("remote") and "--json" not in rest:
                from .remote import RemoteError, session, status_lines
                try:  # the block below says what was tried and that it is not reachable; no second red error
                    got = session(config)
                except RemoteError:
                    got = None
                for line in status_lines(config, got):
                    print(line)
                sys.exit(0 if got else 1)
            got = host.status()
            if "--json" in rest:
                import json
                if got and config.get("server"):
                    from . import budgets
                    got["budget"] = budgets.summary(config, got)
                print(json.dumps(got, indent=2) if got else json.dumps({"host": None}))
                sys.exit(0 if got else 1)
            if got:
                for line in host.status_lines(got):
                    print(line)
                from . import diagnostics
                info = host.read_info()
                print(diagnostics.status_line(host.call(info, "diagnostics") if info else None))
                from . import budgets
                for line in budgets.table(config, got):
                    print(line)
            else:
                from .server import running_context, served_model
                running = served_model(port)
                n_ctx = running_context(port) if running else None
                print("host: not running" + (f"  (a model server from an older Hearthwork answers on port {port}: any command that needs "
                                             "the model adopts it)" if running else ""))
                print(f"model: {GREEN + running + RESET if running else 'not running'}"
                      + (f"  (port {port}, context {n_ctx:,} = {ctx.format_k(n_ctx)})" if n_ctx else f"  (port {port})" if running else ""))
            print(f"models folder: {config.get('modelsDir', '-')}")
            chosen = runtime.selected(config)
            if chosen:
                print(f"runtime: {chosen[0]} b{chosen[1]}")
            from . import setups
            profile_line = setups.status_line(config)
            if profile_line:
                print(profile_line)
            runtime.print_notice(config)
        elif command == "context":
            sys.exit(ctx.main(rest, configured()))
        elif command == "model":
            from . import model
            configured()
            model.main(rest)
        elif command == "bench":
            from . import bench
            if "--show" not in rest and not load_config().get("remote"):
                from .menu import ensure_server
                if not ensure_server(configured()):
                    sys.exit(1)
            bench.main(rest)
        elif command == "tune":
            from . import tune
            sys.exit(tune.main(rest))
        elif command == "share":
            from . import share
            sys.exit(share.share_main(rest))
        elif command == "api":
            from . import api
            sys.exit(api.api_main(rest))
        elif command == "devices":
            from . import share
            sys.exit(share.devices_main(rest))
        elif command == "connect":
            from . import remote
            sys.exit(remote.connect_main(rest))
        elif command == "disconnect":
            from . import remote
            sys.exit(remote.disconnect_main())
        elif command == "doctor":
            from . import doctor
            sys.exit(doctor.main(rest))
        elif command == "check":
            from . import check
            check.main(rest)
        elif command == "setup":
            from . import onboard as setup
            try:
                load_config()
            except ConfigError:
                if not setup.back_up_corrupt_config():
                    sys.exit(1)
            setup.main(rest)
        elif command == "runtime":
            sys.exit(runtime.main(rest))
        elif command == "profile":
            from . import setups
            sys.exit(setups.main(rest))
        elif command == "alias":
            from . import aliases
            sys.exit(aliases.main(rest))
        elif command == "statusline":  # hidden: the status line Claude Code shows, see harnesses.claude_settings
            from .harnesses import statusline_text
            line = statusline_text(rest)  # utf-8 bytes: Claude Code reads utf-8 whatever the console code page
            sys.stdout.buffer.write((line + "\n").encode("utf-8"))
        elif command == "update":
            sys.exit(update())
        else:
            print(f"Unknown command: {command}\n\n{USAGE}")
            sys.exit(2)
    except KeyboardInterrupt:
        print()
        sys.exit(130)
    except ConfigError as error:
        print(f"hearthwork: {error}")
        sys.exit(1)


def check_main():
    """`hearthwork-check`: the checker on its own, e.g. `uvx --from git+https://github.com/BillyMRX1/hearthwork hearthwork-check`."""
    from . import check
    check.main()
