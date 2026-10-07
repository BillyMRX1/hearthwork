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

DIM = "\033[2m"

STATE = HERE / "server.json"  # the background server: pid, model, port
LOG = HERE / "server.log"     # its output on macOS/Linux (Windows shows it in its own window)
SLOTS = HERE / "cache" / "slots"  # saved prompt caches (see save_slots)
MIN_SAVE_TOKENS = 2048  # smaller prompts (e.g. an agent's title request) must not overwrite a saved big one


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


def slot_dir(config, model, context=None):
    """Where this model's processed prompts are saved. Per model, context and KV type: a saved cache only
    restores into a server set up the same way."""
    s = config["server"]
    path = SLOTS / f"{model.stem}-c{context or s['context']}-{s['kvCacheType']}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def command(config, model, context=None):
    s = config["server"]
    cmd = [config["llamaServer"], "-m", str(model), "-c", str(context or s["context"]), "-fa", "on", "--jinja",
           "--fit", "on", "--fit-target", str(s["fitTargetMiB"]), "--load-mode", "none",
           "-ctk", s["kvCacheType"], "-ctv", s["kvCacheType"], "-np", str(s["slots"]), "-kvu",
           "-b", str(s["batch"]), "-ub", str(s["batch"]), "--host", "127.0.0.1", "--port", str(s["port"]),
           "--slot-save-path", str(slot_dir(config, model, context))]
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


def _slot_request(port, slot, action, timeout=60):
    body = json.dumps({"filename": f"slot{slot}.bin"}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/slots/{slot}?action={action}", data=body,
                                     headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def save_slots(port, quiet=False):
    """Save the server's processed prompts to disk, so the next start skips re-reading them.

    An agent's first message makes the model read its whole system prompt (Claude Code: ~17-20K tokens,
    20-40 s); restored from disk that takes under a second. Only idle slots holding a substantial prompt
    are saved. Returns the number of tokens saved."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=5) as response:
            slots = json.load(response)
    except Exception:
        return 0
    saved = 0
    for slot in slots:
        if slot.get("is_processing") or slot.get("n_prompt_tokens", 0) < MIN_SAVE_TOKENS:
            continue
        try:
            saved += _slot_request(port, slot["id"], "save").get("n_saved", 0)
        except Exception:
            pass
    if saved and not quiet:
        print(f"{DIM}Saved the model's prompt cache ({saved:,} tokens) for a faster next start.{RESET}")
    return saved


def restore_slots(port, slots):
    """Load prompt caches saved by save_slots (if any) into the freshly started server."""
    restored = 0
    for slot in range(slots):
        try:
            restored += _slot_request(port, slot, "restore").get("n_restored", 0)
        except Exception:
            pass  # nothing saved for this slot yet
    return restored


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
    restored = restore_slots(config["server"]["port"], config["server"]["slots"])
    note = f" (restored {restored:,} cached prompt tokens: the first message will be quick)" if restored else ""
    print(f" {GREEN}ready in {time.time() - started:.0f} s{RESET}{note}")
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
        save_slots(state["port"], quiet=quiet)
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
    """start.py: run the server in this terminal until Ctrl+C. Saved prompt caches are restored once it is up."""
    import threading

    def restore_when_ready():
        for _ in range(600):
            if ready(config["server"]["port"]):
                restored = restore_slots(config["server"]["port"], config["server"]["slots"])
                if restored:
                    print(f"\n{GREEN}Restored {restored:,} cached prompt tokens: the first message will be quick.{RESET}\n",
                          flush=True)
                return
            time.sleep(1)

    threading.Thread(target=restore_when_ready, daemon=True).start()
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
