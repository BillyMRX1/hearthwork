"""The Hearthwork host: one long-running process per computer that owns llama-server and the single relay every client uses.

  agents / task / mcp -> 127.0.0.1:8470 -> queue + accounting -> relay (normalize, tool-call repair) -> llama-server
  `hearthwork api` (8080) and `hearthwork share` (8484) are foreground commands that forward to the same listener.

Any command that needs the model calls ensure(), which starts the host detached when it is not running. The host runs
as `hearthwork host` (hidden; `hearthwork serve` runs it in the foreground). Its control API, /hearthwork/control/status
and /stop on the listener, needs the random token in host.json. Logs: HOME/logs/host.log and server.log.
"""
import collections
import hashlib
import hmac
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

from . import context as ctx
from .harnesses import ClientGone, Overloaded, client_gone, make_relay
from .onboard import GREEN, RED, RESET, WINDOWS, YELLOW
from .paths import HOME, HOST_FILE, HOST_LOCK, HOST_LOG, LOG, STATE

DEFAULT_PORT = 8470
MAX_QUEUE = 16          # requests allowed to wait
MAX_WAIT = 600          # seconds one may wait
LOG_LIMIT = 5_000_000   # host.log is rotated (host.log.1) beyond this
DIM = "\033[2m"
LOAD_TIMEOUT = 1800     # seconds `ensure` waits for a model to load
DRAIN_SECONDS = 30      # a stop waits this long for running requests
GENERATION = ("/messages", "/chat/completions", "/responses", "/completions", "/completion")


def port(config=None):
    """The host's local listener port (8470 unless config.json says host.port)."""
    try:
        if config is None:
            from .onboard import load_config
            config = load_config()
        return int((config.get("host") or {}).get("port") or DEFAULT_PORT)
    except (TypeError, ValueError, OSError):
        return DEFAULT_PORT


# ---------- logging ----------

class Logger:
    """File-like (the host's stdout and stderr): timestamped lines in host.log, rotated at ~5 MB; also on the console
    when the host runs in the foreground."""

    def __init__(self, path=HOST_LOG, console=None, limit=LOG_LIMIT):
        self.path, self.console, self.limit = Path(path), console, limit
        self.lock, self.partial = threading.Lock(), ""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, text):
        if self.console:
            try:
                self.console.write(text)
                self.console.flush()
            except (OSError, ValueError):
                pass
        with self.lock:
            self.partial += text
            *lines, self.partial = self.partial.split("\n")
            lines = [line.rstrip("\r") for line in lines if line.strip()]
            if lines:
                self._append(lines)
        return len(text)

    def _append(self, lines):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            if self.path.exists() and self.path.stat().st_size > self.limit:
                backup = self.path.with_name(self.path.name + ".1")
                backup.unlink(missing_ok=True)
                self.path.rename(backup)
            with open(self.path, "a", encoding="utf-8", errors="replace") as f:
                f.write("".join(f"{stamp} {line}\n" for line in lines))
        except OSError:
            pass

    def flush(self):
        pass

    def isatty(self):
        return False


# ---------- request accounting and the queue ----------

def client_id(headers):
    """Who is asking: the X-Hearthwork-Client header (api, share), else the token the request carries. Hearthwork
    gives each agent session `local-<agent>-<id>`; any other token is shown as a short hash, never as itself."""
    named = (headers.get("x-hearthwork-client") or "").strip()
    if named:
        return named[:60]
    auth = headers.get("authorization") or ""
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else (headers.get("x-api-key") or "").strip()
    if not token:
        return "unknown"
    if token.startswith("local-") and len(token) <= 60:
        return token
    return "key-" + hashlib.sha256(token.encode()).hexdigest()[:6]


def display_name(cid):
    """local-claude-ab12 -> claude (ab12)."""
    if cid.startswith("local-"):
        agent, _, short = cid[6:].rpartition("-")
        return f"{agent} ({short})" if agent else cid
    return cid


class Ticket:
    def __init__(self, gate, entry):
        self.gate, self.entry, self.done = gate, entry, False

    def release(self, outcome="ok"):
        with self.gate.cond:  # the stream thread and the disconnect watcher may both release (cond is an RLock)
            if self.done:
                return
            self.done = True
        self.gate._release(self.entry, outcome)


class Gate:
    """At most `limit()` generation requests run at once; the others wait in one FIFO queue (at most `max_queue` of them,
    each for at most `max_wait` seconds). A stream counts as running until the caller releases the ticket."""

    def __init__(self, limit, log=None, max_queue=MAX_QUEUE, max_wait=MAX_WAIT):
        self.limit, self.max_queue, self.max_wait = limit, max_queue, max_wait
        self.log = log or (lambda message: None)
        self.cond = threading.Condition()
        self.queue = collections.deque()
        self.active = []
        self.clients = {}
        self.rejected = 0
        self.closing = False

    def wants(self, command, path):
        return command == "POST" and path.rstrip("/").endswith(GENERATION)

    def client_of(self, headers):
        return client_id(headers)

    def _client(self, cid):
        return self.clients.setdefault(cid, {"active": 0, "queued": 0, "total": 0, "last": 0})

    def admit(self, sock, cid, path=""):
        entry = {"client": cid, "path": path, "queued": time.monotonic()}
        who = display_name(cid)
        with self.cond:
            if self.closing:
                raise Overloaded("The Hearthwork host is shutting down.")
            if self.queue or len(self.active) >= self.limit():
                if len(self.queue) >= self.max_queue:
                    self.rejected += 1
                    self.log(f"{who} {path}: refused, the queue is full ({len(self.queue)} waiting)")
                    raise Overloaded(f"The model is busy and {len(self.queue)} requests are already waiting. Try again in a minute.")
                self.queue.append(entry)
                self._client(cid)["queued"] += 1
                self.log(f"{who} {path}: queued (position {len(self.queue)}, {len(self.active)} running)")
                try:
                    while not (self.queue[0] is entry and len(self.active) < self.limit()):
                        if self.closing:
                            raise Overloaded("The Hearthwork host is shutting down.")
                        if time.monotonic() - entry["queued"] >= self.max_wait:
                            self.rejected += 1
                            self.log(f"{who} {path}: gave up after {self.max_wait:.0f} s in the queue")
                            raise Overloaded(f"The request waited {self.max_wait:.0f} s for a free slot of the model. Try again later.")
                        self.cond.wait(0.25)
                        if client_gone(sock):
                            self.log(f"{who} {path}: caller left the queue")
                            raise ClientGone()
                finally:
                    # leave the queue (admitted, or not): the next in line may go
                    if entry in self.queue:
                        self.queue.remove(entry)
                    self._client(cid)["queued"] -= 1
                    self.cond.notify_all()
            entry["started"] = time.monotonic()
            self.active.append(entry)
            client = self._client(cid)
            client["active"] += 1
            client["total"] += 1
            client["last"] = time.time()
            self.log(f"{who} {path}: running ({len(self.active)}/{self.limit()}), waited {entry['started'] - entry['queued']:.1f} s")
        return Ticket(self, entry)

    def _release(self, entry, outcome):
        with self.cond:
            if entry in self.active:
                self.active.remove(entry)
            client = self._client(entry["client"])
            client["active"] -= 1
            self.cond.notify_all()
            self.log(f"{display_name(entry['client'])} {entry['path']}: {outcome} after {time.monotonic() - entry['started']:.1f} s")
            if len(self.clients) > 50:  # keep the table small: forget the idle ones that were seen longest ago
                idle = sorted((c["last"], k) for k, c in self.clients.items() if not c["active"] and not c["queued"])
                for _, key in idle[:len(self.clients) - 50]:
                    del self.clients[key]

    def close(self):
        with self.cond:
            self.closing = True
            self.cond.notify_all()

    def wait_idle(self, seconds):
        """Wait until no request is running (at most `seconds`); True when idle."""
        end = time.monotonic() + seconds
        with self.cond:
            while self.active and time.monotonic() < end:
                self.cond.wait(0.25)
            return not self.active

    def snapshot(self):
        with self.cond:
            return {"limit": self.limit(), "active": len(self.active), "queued": len(self.queue), "rejected": self.rejected,
                    "clients": {k: dict(v) for k, v in self.clients.items() if v["active"] or v["queued"] or v["total"]}}


# ---------- the host process ----------

_LOCK = None


def acquire_lock(path=None):
    """Take the host lock for the life of this process (OS-level: it vanishes with the process, even if killed).
    False when another host holds it. Taken before anything else, so a second host never touches llama-server."""
    global _LOCK
    handle = open(path or HOST_LOCK, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _LOCK = handle
    return True


def write_private(path, text):
    """Write a file only the user can read where the OS allows (POSIX mode 600; Windows: the user's own profile folder)."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def props_slots(llama_port, fallback):
    """Parallel requests llama-server runs (/props total_slots), else the configured slots."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{llama_port}/props", timeout=3) as response:
            total = json.load(response).get("total_slots")
        if isinstance(total, int) and total > 0:
            return total
    except (OSError, ValueError, AttributeError):
        pass
    return fallback


class Host:
    def __init__(self, config, log):
        self.config, self.log = config, log
        self.llama_port = config["server"]["port"]
        self.slots = int(config["server"].get("slots") or 2)
        self.token = secrets.token_urlsafe(24)
        self.started = time.time()
        self.model = self.context = self.error = None
        self.ready, self.adopted, self.restored = False, False, 0
        self.process = None
        self.stop_event = threading.Event()
        self.gate = Gate(lambda: self.slots, log)
        self.listener = None

    def record(self):
        write_private(HOST_FILE, json.dumps({"pid": os.getpid(), "port": port(self.config), "token": self.token,
                                             "model": self.model, "started": self.started, "adopted": self.adopted}))

    def status(self):
        return {"pid": os.getpid(), "port": port(self.config), "model": self.model, "context": self.context,
                "ready": self.ready, "error": self.error, "adopted": self.adopted, "restored": self.restored,
                "serverPort": self.llama_port, "started": self.started, "uptime": time.time() - self.started,
                "slots": self.slots, **self.gate.snapshot()}

    def control(self, handler):
        """Answer /hearthwork/control/* (token required); True when the request was handled here."""
        path = handler.path.split("?")[0].rstrip("/")
        if not path.startswith("/hearthwork/"):
            return False
        auth = handler.headers.get("authorization") or ""
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else (handler.headers.get("x-hearthwork-token") or "")
        if not path.startswith("/hearthwork/control/") or not hmac.compare_digest(given.encode(), self.token.encode()):
            handler._send_json(401 if path.startswith("/hearthwork/control/") else 404,
                               {"error": {"message": "control token required" if path.startswith("/hearthwork/control/") else "not found"}})
            return True
        if path == "/hearthwork/control/status" and handler.command == "GET":
            handler._send_json(200, self.status())
        elif path == "/hearthwork/control/stop" and handler.command == "POST":
            handler._send_json(200, {"stopping": True})
            self.stop_event.set()
        else:
            handler._send_json(404, {"error": {"message": "not found"}})
        return True

    # -- llama-server --

    def start_model(self, model, context, adopt):
        from . import server
        try:
            if adopt:
                self.adopted = True
                name = server.served_model(self.llama_port, timeout=60)
                if not name:
                    raise RuntimeError(f"nothing answers on port {self.llama_port}")
                self.log(f"Using the model server already running on port {self.llama_port}: {name}")
            else:
                server.stop(self.config, quiet=True)  # a stale record or leftover process of an older start
                context, source, rng = ctx.choose(self.config, model, context)
                self.log(ctx.describe(context, source, rng))
                cmd = server.command(self.config, model, context)
                self.log("Starting " + model.name)
                self.process = server.spawn_server(cmd)
                STATE.write_text(json.dumps({"pid": self.process.pid, "model": str(model), "port": self.llama_port}))
                started = time.time()
                while not server.ready(self.llama_port):
                    if self.process.poll() is not None:
                        tail = ""
                        try:
                            tail = "\n".join(LOG.read_text(errors="replace").splitlines()[-10:])
                        except OSError:
                            pass
                        STATE.unlink(missing_ok=True)
                        raise RuntimeError(f"llama-server stopped while loading (exit code {self.process.returncode})\n{tail}")
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)
                self.restored = server.restore_slots(self.llama_port, self.config["server"]["slots"])
                self.log(f"Model ready in {time.time() - started:.0f} s" + (f", restored {self.restored:,} cached prompt tokens" if self.restored else ""))
            self.model = server.served_model(self.llama_port, timeout=30) or (model.stem if model else None)
            self.context = server.running_context(self.llama_port)
            self.slots = props_slots(self.llama_port, self.slots)
            self.ready = True
            self.record()
        except Exception as error:
            self.error = str(error)
            self.log(f"The model did not start: {error}")
            self.stop_event.set()

    def shutdown(self):
        from . import server
        self.log("Stopping: saving the prompt cache, then the model server")
        self.gate.close()  # no new requests; the queued ones get an overload error
        if not self.gate.wait_idle(DRAIN_SECONDS):
            self.log(f"{len(self.gate.active)} request(s) still running after {DRAIN_SECONDS} s; stopping anyway")
        self.ready = False
        try:
            server.stop(self.config, quiet=False)
        except Exception:
            self.log(traceback.format_exc())
        if self.listener:
            threading.Thread(target=self.listener.shutdown, daemon=True).start()
        try:
            if json.loads(HOST_FILE.read_text()).get("pid") == os.getpid():
                HOST_FILE.unlink()
        except (OSError, ValueError):
            pass


def serve(config, model=None, context=None, adopt=False, console=False):
    """Run the host in this process until stopped; returns the exit code."""
    if not acquire_lock():
        print("Another Hearthwork host is already running; this one exits without touching the model.")
        return 1
    log = Logger(console=sys.stdout if console else None)
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = log
    host = Host(config, lambda message: print(message, flush=True))
    try:
        try:
            host.listener = make_relay(host.llama_port, ("127.0.0.1", port(config)), host.control, gate=host.gate, exclusive=True)
        except OSError as error:
            print(f"Cannot listen on port {port(config)} (another host running?): {error}")
            return 1
        host.listener.daemon_threads = True
        host.record()
        print(f"Hearthwork host {os.getpid()} listening on 127.0.0.1:{port(config)}")
        threading.Thread(target=host.listener.serve_forever, daemon=True).start()
        threading.Thread(target=host.start_model, args=(model, context, adopt), daemon=True).start()
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGTERM, lambda *_: host.stop_event.set())
            except (ValueError, OSError):
                pass
        refreshed = time.time()
        try:
            while not host.stop_event.wait(1):
                if host.process and host.ready and host.process.poll() is not None:
                    host.error = f"llama-server exited (code {host.process.returncode})"
                    print(host.error + "; stopping the host")
                    break
                if host.ready and time.time() - refreshed > 30:
                    refreshed = time.time()
                    host.slots = props_slots(host.llama_port, host.slots)
        except KeyboardInterrupt:
            pass
        if host.error:
            time.sleep(2)  # let a waiting `hearthwork start` read the error
        host.shutdown()
        return 1 if host.error else 0
    except Exception:
        print(traceback.format_exc())
        host.shutdown()
        return 1
    finally:
        sys.stdout, sys.stderr = saved


def host_main(argv):
    """`hearthwork host` (hidden): run the host; `--console` also prints its log (what `hearthwork serve` does)."""
    import argparse
    from .cli import configured
    parser = argparse.ArgumentParser(prog="hearthwork host")
    parser.add_argument("--model")
    parser.add_argument("--context", type=ctx.parse_k)
    parser.add_argument("--adopt", action="store_true", help="use the llama-server that is already running")
    parser.add_argument("--console", action="store_true")
    args = parser.parse_args(argv)
    config = configured(interactive=False)
    return serve(config, Path(args.model) if args.model else None, args.context, args.adopt, args.console)


# ---------- clients of the host ----------

def _alive(pid):
    from .server import _alive as alive
    return alive(pid)


def read_info():
    """host.json of a running host, or None. A record whose process is gone is removed."""
    try:
        info = json.loads(HOST_FILE.read_text())
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not _alive(pid):
        HOST_FILE.unlink(missing_ok=True)
        return None
    return info


def call(info, path, method="GET", timeout=5):
    """JSON answer of the host's control API, or None."""
    request = urllib.request.Request(f"http://127.0.0.1:{info['port']}/hearthwork/control/{path}", method=method,
                                     data=b"" if method == "POST" else None, headers={"Authorization": f"Bearer {info['token']}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def status():
    """The host's status (model, context, slots, requests per client), or None when no host answers."""
    info = read_info()
    return call(info, "status") if info else None


def running():
    return status() is not None


def spawn(args):
    """Start `hearthwork host` detached from this terminal: no console window on Windows, its own session elsewhere."""
    cmd = [sys.executable, "-m", "hearthwork", "host", *args]
    quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(HOME))
    if WINDOWS:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(cmd, creationflags=flags | 0x01000000, close_fds=True, **quiet)  # CREATE_BREAKAWAY_FROM_JOB
        except OSError:  # the caller's job does not allow breakaway
            return subprocess.Popen(cmd, creationflags=flags, close_fds=True, **quiet)
    return subprocess.Popen(cmd, start_new_session=True, **quiet)


def _wait_ready(label, process=None, timeout=LOAD_TIMEOUT):
    """Wait for the host's model to be ready. Gives up with a message when the host never answers, disappears (e.g.
    `hearthwork stop` in another terminal), or the model does not load within `timeout` seconds."""
    print(f"{label}", end="", flush=True)
    started = time.time()
    seen, silent, gone_since = False, None, None
    while True:
        info = read_info()
        got = call(info, "status") if info else None
        now = time.time()
        if got and got.get("error"):
            print(f"\n{RED}The model did not start: {got['error']}{RESET}")
            return False
        if got and got.get("ready"):
            note = f" (restored {got['restored']:,} cached prompt tokens: the first message will be quick)" if got.get("restored") else ""
            print(f" {GREEN}ready in {now - started:.0f} s{RESET}{note}")
            return True
        seen = seen or bool(info)
        if got:
            silent = None
        elif info:  # a process with that pid, but not answering as a host with this token: dead, reused, or hung
            silent = silent or now
            if now - silent > 60:
                print(f"\n{RED}Process {info['pid']} does not answer as a Hearthwork host (see `hearthwork logs`).{RESET}")
                return False
        elif seen:
            print(f"\n{RED}The host stopped before the model was ready (`hearthwork stop` elsewhere, or a crash: see `hearthwork logs`).{RESET}")
            return False
        elif process is not None and process.poll() is not None:
            gone_since = gone_since or now
            if now - gone_since > 10:  # our host died (a host started by someone else would have written host.json)
                print(f"\n{RED}The host stopped while starting (see `hearthwork logs`).{RESET}")
                _tail(HOST_LOG)
                return False
        if now - started > timeout:
            print(f"\n{RED}The model was not ready after {timeout:.0f} s (see `hearthwork logs --server`).{RESET}")
            return False
        time.sleep(1)
        print(".", end="", flush=True)


def _tail(path, lines=10):
    try:
        print("\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:]))
    except OSError:
        pass


def ensure(config, model=None, context=None, force=False):
    """Make sure the host runs the model. Starts the host (and llama-server) when needed; adopts a llama-server an older
    Hearthwork started. A different `model` or `context` than the running one restarts the host's server, but refuses
    while requests are active unless `force`. Returns True when the model is ready."""
    from . import server
    llama = config["server"]["port"]
    info = read_info()
    got = call(info, "status") if info else None
    if info and not got:  # alive but silent: still starting
        return _wait_ready("Waiting for the Hearthwork host")
    if got:
        same = model is None or (got.get("model") == model.stem and (context is None or got.get("context") == context))
        if same:
            return True if got.get("ready") else _wait_ready("Waiting for the model")
        busy = got["active"] + got["queued"]
        if busy and not force:
            print(f"{YELLOW}Not switching: {busy} request(s) are running or waiting ({got['active']} active, {got['queued']} queued). "
                  f"Wait for them, or use --force to interrupt them.{RESET}")
            return False
        stop(config, quiet=True, force=True)
    elif server.port_open(llama):  # a llama-server of an older Hearthwork (or someone else's) is already there
        name = server.served_model(llama) or server.served_model(llama, timeout=60)
        if not name:
            print(f"{YELLOW}Port {llama} is in use but no model answers. Stop it with `hearthwork stop` first.{RESET}")
            return False
        if model is None or (name == model.stem and context in (None, server.running_context(llama))):
            print(f"{DIM}Adopting the model server already running ({name}).{RESET}")
            return _wait_ready("Starting the host", spawn(["--adopt"]))
        server.stop(config, quiet=True)
        if server.port_open(llama):
            print(f"{YELLOW}Port {llama} is still in use. Stop that program first.{RESET}")
            return False
    if model is None:
        print(f"{YELLOW}No model to start.{RESET}")
        return False
    args = ["--model", str(model)] + (["--context", str(context)] if context else [])
    return _wait_ready(f"Starting {model.name} (logs: `hearthwork logs`)", spawn(args))


def stop(config, quiet=False, force=False):
    """Stop the model (saving its prompt cache) and the host. Also stops a llama-server an older Hearthwork started.
    With requests active or queued it refuses unless `force`; then it stops admitting requests and waits up to
    30 s for the running ones. Returns False when it refused."""
    from . import server
    info = read_info()
    if info:
        got = call(info, "status")
        if got and got["active"] + got["queued"] and not force:
            print(f"{YELLOW}Not stopping: {got['active']} request(s) running and {got['queued']} waiting:{RESET}")
            for line in status_lines(got)[3:]:
                print(line)
            print("Wait for them, or use `hearthwork stop --force`.")
            return False
        call(info, "stop", "POST")
        for _ in range(240):
            if not _alive(info["pid"]):
                break
            time.sleep(0.5)
        else:
            if WINDOWS:
                subprocess.run(["taskkill", "/PID", str(info["pid"]), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.kill(info["pid"], signal.SIGKILL)
                except OSError:
                    pass
        HOST_FILE.unlink(missing_ok=True)
        if STATE.exists():
            server.stop(config, quiet=True)  # leftovers of a host that had to be killed
        if not quiet:
            print("Stopped the model and the Hearthwork host.")
        return True
    if STATE.exists():
        server.stop(config, quiet=quiet)
    elif not quiet:
        print("No model server started by Hearthwork is running.")
    return True


# ---------- status and logs ----------

def duration(seconds):
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return (f"{days} d " if days else "") + (f"{hours} h " if hours or days else "") + (f"{minutes} min" if minutes or hours or days else f"{secs} s")


def status_lines(got):
    """Text lines for `hearthwork status` from a host status."""
    lines = [f"host: {GREEN}running{RESET}  pid {got['pid']}, port {got['port']}, up {duration(got['uptime'])}"
             + (f"  {DIM}(model server started by an older Hearthwork){RESET}" if got.get("adopted") else "")]
    if got.get("error"):
        lines.append(f"model: {RED}{got['error']}{RESET}")
    elif got.get("ready"):
        n_ctx = got.get("context")
        lines.append(f"model: {GREEN}{got['model']}{RESET}  (port {got['serverPort']}"
                     + (f", context {n_ctx:,} = {ctx.format_k(n_ctx)}" if n_ctx else "") + f", {got['slots']} slots)")
    else:
        lines.append(f"model: {YELLOW}loading{RESET}")
    lines.append(f"requests: {got['active']} active / {got['limit']} slots, {got['queued']} queued"
                 + (f", {got['rejected']} refused" if got.get("rejected") else ""))
    for cid, c in sorted(got["clients"].items()):
        lines.append(f"  {display_name(cid):<28} active {c['active']}  queued {c['queued']}  total {c['total']}")
    return lines


def logs_main(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="hearthwork logs", description="Show the host's log (requests, queue, start and stop).")
    parser.add_argument("--server", action="store_true", help="show llama-server's own output instead")
    parser.add_argument("-f", "--follow", action="store_true", help="keep printing new lines (Ctrl+C stops)")
    parser.add_argument("-n", type=int, default=100, help="how many lines to start with (default 100)")
    args = parser.parse_args(argv)
    path = LOG if args.server else HOST_LOG
    if not path.exists():
        print(f"No log yet: {path}")
        return 0 if not args.follow else 1
    text = path.read_text(encoding="utf-8", errors="replace")
    print("\n".join(text.splitlines()[-args.n:]), flush=True)
    if not args.follow:
        return 0
    position = len(text.encode("utf-8", errors="replace"))
    try:
        while True:
            time.sleep(0.5)
            try:
                size = path.stat().st_size
                if size < position:  # rotated or restarted
                    position = 0
                if size > position:
                    with open(path, "rb") as f:
                        f.seek(position)
                        data = f.read()
                    position += len(data)
                    sys.stdout.write(data.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            except OSError:
                pass
    except KeyboardInterrupt:
        return 0
