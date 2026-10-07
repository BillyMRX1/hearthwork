"""The llama.cpp server: build its command, start it in the foreground or background, check and stop it."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from onboard import GREEN, HERE, RED, RESET, WINDOWS, YELLOW, ask, find_models, memory_gb

STATE = HERE / "server.json"  # the background server: pid, model, port
LOG = HERE / "server.log"     # its output on macOS/Linux (Windows shows it in its own window)


def pick_model(models, folder, last):
    names = [str(m) for m in models]
    default = names.index(last) if last in names else 0
    print(f"\nModels in {folder}:")
    for i, model in enumerate(models):
        line = f"  {i + 1:2}) {model_size_gb(model):6.1f} GB  {model.name}"
        print(f"{GREEN}{line}   <- last used{RESET}" if str(model) == last else line)
    while True:
        answer = ask(f"\nWhich model? [Enter = {default + 1}, c = cancel] ")
        if not answer:
            return models[default]
        if answer.lower() == "c":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]
        print(f"{RED}Type a number from 1 to {len(models)}.{RESET}")


def model_size_gb(model):
    """All parts of a multi-part model count."""
    if "-00001-of-" in model.name:
        return sum(p.stat().st_size for p in model.parent.glob(model.name.replace("-00001-of-", "-*-of-"))) / 2**30
    return model.stat().st_size / 2**30


def warn_if_too_big(config, model):
    vram, ram_free = config["hardware"].get("vramGB") or 0, memory_gb()[1]
    size = model_size_gb(model)
    if size > vram + ram_free * 0.8:
        print(f"{YELLOW}Warning: this model is {size:.1f} GB, but only ~{vram:.0f} GB GPU memory + {ram_free:.0f} GB free RAM "
              f"are available. It may fail to load or be very slow. Close other programs or pick a smaller model.{RESET}")


def command(config, model, context=None):
    s = config["server"]
    cmd = [config["llamaServer"], "-m", str(model), "-c", str(context or s["context"]), "-fa", "on", "--jinja",
           "--fit", "on", "--fit-target", str(s["fitTargetMiB"]), "--load-mode", "none",
           "-ctk", s["kvCacheType"], "-ctv", s["kvCacheType"], "-np", str(s["slots"]), "-kvu",
           "-b", str(s["batch"]), "-ub", str(s["batch"]), "--host", "127.0.0.1", "--port", str(s["port"])]
    # Escape hatch for a chat template broken in a way the relay (harnesses.py) doesn't cover.
    template = HERE / "templates" / f"{model.stem}.jinja"
    if template.is_file():
        cmd += ["--chat-template-file", str(template)]
    return cmd


def served_model(port, timeout=2):
    """File name (without .gguf) of the model a server on `port` is serving, or None if none answers."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as response:
            served = json.load(response)["data"][0]["id"]
        return Path(served.replace("\\", "/")).stem
    except Exception:
        return None


def ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def _alive(pid):
    if WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_background(config, model, context=None):
    """Start the server in its own window (Windows) or in the background with a log file; wait until ready."""
    stop(config, quiet=True)
    cmd = command(config, model, context)
    if WINDOWS:  # its own console window shows the server log; closing that window stops the server
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        where = "its own window"
    else:
        log = open(LOG, "w")
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        where = str(LOG)
    STATE.write_text(json.dumps({"pid": process.pid, "model": str(model), "port": config["server"]["port"]}))
    print(f"Starting {model.name} (log: {where})", end="", flush=True)
    started = time.time()
    while not ready(config["server"]["port"]):
        if process.poll() is not None:
            print(f"\n{RED}The server stopped while loading (exit code {process.returncode}).{RESET}")
            if not WINDOWS and LOG.exists():
                print("\n".join(LOG.read_text(errors="replace").splitlines()[-15:]))
            STATE.unlink(missing_ok=True)
            return False
        time.sleep(1)
        print(".", end="", flush=True)
    print(f" {GREEN}ready in {time.time() - started:.0f} s{RESET}")
    return True


def stop(config, quiet=False):
    """Stop the server this tool started in the background (not one started by hand with start.py)."""
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        if not quiet:
            print("No background server started from here is running.")
        return
    pid = state["pid"]
    if _alive(pid):
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        for _ in range(30):
            if not served_model(state["port"], timeout=1):
                break
            time.sleep(0.5)
        if not quiet:
            print(f"Stopped {Path(state['model']).name}.")
    STATE.unlink(missing_ok=True)


def run_foreground(config, model, context=None):
    """start.py: run the server in this terminal until Ctrl+C."""
    try:
        sys.exit(subprocess.call(command(config, model, context)))
    except KeyboardInterrupt:
        pass


def choose_and_remember(config, model_hint=None):
    """Pick a model (menu, or `model_hint` = part of a file name) and remember it as last used."""
    from onboard import save_config
    models = find_models(config["modelsDir"])
    if not models:
        print(f"\nNo models in {config['modelsDir']} yet. Download one first.")
        return None
    if model_hint:
        matches = [m for m in models if model_hint.lower() in m.name.lower()]
        if len(matches) != 1:
            print(f"'{model_hint}' matches {len(matches)} models; use a more specific part of the file name.")
            return None
        model = matches[0]
    else:
        model = pick_model(models, config["modelsDir"], config.get("lastModel"))
    if model:
        config["lastModel"] = str(model)
        save_config(config)
        warn_if_too_big(config, model)
    return model
