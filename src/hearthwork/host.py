"""The Hearthwork host: one long-running process per computer that owns llama-server and the single relay every client uses.

  agents / task / mcp -> 127.0.0.1:8470 -> queue + accounting -> relay (normalize, tool-call repair) -> llama-server
  `hearthwork api` (8080) and `hearthwork share` (8484) are foreground commands that forward to the same listener.

Any command that needs the model calls ensure(), which starts the host detached when it is not running. The host runs
as `hearthwork host` (hidden; `hearthwork serve` runs it in the foreground). Its control API, /hearthwork/control/status
and /stop on the listener, needs the random token in host.json. Logs: HOME/logs/host.log and server.log.
"""
import atexit
import collections
import hashlib
import hmac
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import budgets, context as ctx, diagnostics
from .harnesses import ClientGone, Overloaded, Unavailable, client_gone, make_relay
from .onboard import GREEN, RED, RESET, WINDOWS, YELLOW
from .paths import HOME, HOST_FILE, HOST_LOCK, HOST_LOG, LOG, STATE
from .procs import pid_alive

DEFAULT_PORT = 8470
MAX_QUEUE = 16          # requests allowed to wait
MAX_WAIT = 600          # seconds one may wait
LOG_LIMIT = 5_000_000   # host.log is rotated (host.log.1) beyond this
DIM = "\033[2m"
LOAD_TIMEOUT = 1800     # seconds `ensure` waits for a model to load
DRAIN_SECONDS = 30      # a stop waits this long for running requests
GENERATION = ("/messages", "/chat/completions", "/responses", "/completions", "/completion")
SESSION_LIMIT = 200                 # the registry keeps at most this many sessions
SESSION_MAX_AGE = 7 * 24 * 3600     # a session this old is dropped even if its pid is still alive (pid reuse)


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
    """local-claude-ab12 -> claude (ab12). The unlabeled fallback ("unknown") shows as "other"."""
    if cid == "unknown":
        return "other"
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
        self.hold = None            # why new requests wait (switching, loading, recovering...); None: admit as usual
        self.failed = None          # the model server failed for good: requests get this as a 503
        self.wake = None            # called while requests wait on a held gate (starts an unloaded model)
        self.models_fallback = None  # host: the /v1/models list while llama-server is not up
        self.on_refused = None      # host: wait for a restart; True when the model server is back
        self.last_activity = time.monotonic()
        self.hint = None            # latest model server failure, told to the requests it interrupts

    def wants(self, command, path):
        return command == "POST" and path.rstrip("/").endswith(GENERATION)

    def client_of(self, headers):
        return client_id(headers)

    def _client(self, cid):
        return self.clients.setdefault(cid, {"active": 0, "queued": 0, "total": 0, "last": 0})

    def admit(self, sock, cid, path=""):
        entry = {"client": cid, "path": path, "queued": time.monotonic()}
        who = display_name(cid)
        if self.hold and self.wake:
            self.wake()
        with self.cond:
            if self.closing:
                raise Overloaded("The Hearthwork host is shutting down.")
            if self.failed:
                raise Unavailable(self.failed)
            self.last_activity = time.monotonic()
            if self.queue or self.hold or len(self.active) >= self.limit():
                if len(self.queue) >= self.max_queue:
                    self.rejected += 1
                    self.log(f"{who} {path}: refused, the queue is full ({len(self.queue)} waiting)")
                    raise Overloaded(f"The model is busy and {len(self.queue)} requests are already waiting. Try again in a minute.")
                self.queue.append(entry)
                self._client(cid)["queued"] += 1
                self.log(f"{who} {path}: queued (position {len(self.queue)}, {len(self.active)} running)")
                try:
                    while not (self.queue[0] is entry and len(self.active) < self.limit() and not self.hold):
                        if self.closing:
                            raise Overloaded("The Hearthwork host is shutting down.")
                        if self.failed:
                            raise Unavailable(self.failed)
                        if self.hold and self.wake:
                            self.wake()
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
            entry["started"] = self.last_activity = time.monotonic()
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
            self.last_activity = time.monotonic()
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

    def set_hold(self, reason=None, failed=None):
        """Hold (or release) new requests: they wait in the queue, within the normal wait limit. `failed` makes them fail."""
        with self.cond:
            self.hold, self.failed = reason, failed
            self.cond.notify_all()

    def cancel_all(self, message):
        """Cancel the running requests (a forced switch): each client gets `message` as its error. Returns how many."""
        with self.cond:
            entries = list(self.active)
        for entry in entries:
            entry["cancelled"] = message
            abort = entry.get("abort")
            if abort:
                abort()
        return len(entries)

    def wait_available(self):
        """A request that is not a generation (count_tokens...) while the model is unloaded or loading: wake it and wait,
        within the normal wait limit, instead of failing."""
        if not self.hold and not self.failed:
            return
        end = time.monotonic() + self.max_wait
        with self.cond:
            while self.hold or self.failed:
                if self.failed:
                    raise Unavailable(self.failed)
                if self.closing:
                    raise Overloaded("The Hearthwork host is shutting down.")
                if time.monotonic() > end:
                    raise Overloaded(f"The model was not ready after {self.max_wait:.0f} s. Try again later.")
                if self.wake:
                    self.wake()
                self.cond.wait(0.25)

    def recovered_for_retry(self):
        """A request was refused or reset before the model server accepted it: True once the host got it running again."""
        return bool(self.on_refused and self.on_refused())

    def failure_text(self):
        return self.failed or self.hint

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


class Sessions:
    """Local agent sessions that are using this host: client id -> {client, kind, pid, since}. A client registers when
    it starts (control API) and drops out when it deregisters, when its process is gone, or when it is too old for its
    pid to mean anything. Only local clients register; a client of a remote host is on another computer and is gated
    there through the share."""

    def __init__(self, alive=None):
        self.lock = threading.Lock()
        self.items = {}
        self.alive = alive

    def _gone(self, pid):
        # pid_alive (OpenProcess) is a fast call: status() must stay quick even while the model is loading, or a client
        # reads the host as silent. tasklist (host._alive) is far too slow to run on every status request.
        try:
            return not (self.alive or pid_alive)(int(pid))
        except (TypeError, ValueError):
            return True

    def _drop_stale(self):
        now = time.time()
        for client in [c for c, s in self.items.items() if self._gone(s["pid"]) or now - s["since"] > SESSION_MAX_AGE]:
            del self.items[client]

    def register(self, client, kind=None, pid=None):
        """Record `client` as an open session. Returns the record, or None when the client id is empty."""
        client = (client or "").strip()[:60]
        if not client:
            return None
        with self.lock:
            self._drop_stale()
            if client not in self.items and len(self.items) >= SESSION_LIMIT:  # forget the oldest to stay small
                del self.items[min(self.items, key=lambda k: self.items[k]["since"])]
            kind = (kind or "agent").strip()[:40]
            self.items[client] = {"client": client, "kind": kind or "agent",
                                  "pid": int(pid or os.getpid()), "since": time.time()}
            return dict(self.items[client])

    def unregister(self, client):
        with self.lock:
            self._drop_stale()
            return self.items.pop((client or "").strip()[:60], None) is not None

    def clear(self):
        with self.lock:
            self.items.clear()

    def open(self):
        """The open sessions, oldest first; stale ones (dead pid) are dropped here too."""
        with self.lock:
            self._drop_stale()
            return [dict(s) for s in sorted(self.items.values(), key=lambda s: s["since"])]


def slots_text(count):
    """'1 slot' / 'N slots'."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 0
    return f"{count} slot" + ("" if count == 1 else "s")


def session_line(session):
    """One open session, as "claude session since 14:32:10 (pid 1234)"."""
    return (f"{session['kind']} session since "
            f"{time.strftime('%H:%M:%S', time.localtime(session['since']))} (pid {session['pid']})")


def sessions_message(sessions):
    """Why a switch is refused while agent sessions are open, listing each one."""
    listed = "; ".join(session_line(s) for s in sessions)
    count = len(sessions)
    # no trailing period: the callers (ensure, apply_pending) add their own sentence after it
    return (f"{count} agent session{'' if count == 1 else 's'} {'is' if count == 1 else 'are'} still using the model: "
            f"{listed}. Wait for {'it' if count == 1 else 'them'} to finish, or use --force to interrupt")


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


OOM_PATTERNS = ("out of memory", "failed to allocate", "cudamalloc failed", "unable to allocate", "outofdevicememory",
                "bad_alloc", "insufficient memory")
SWITCH_DRAIN = 120      # a switch waits this long for running requests (with --force: FORCE_GRACE, then cancels them)
FORCE_GRACE = 10
HEALTH_EVERY = 10       # seconds between /health checks of llama-server
HEALTH_TIMEOUT = 10     # a slower answer than this is a failed check, never a reason to restart by itself
HEALTH_FAILS = 6        # consecutive failed checks (and no request running) before a silent process is restarted
RESTART_LIMIT = 3       # restarts allowed per RESTART_WINDOW seconds; then the state is "failed"
RESTART_WINDOW = 900
BACKOFF = (2, 10, 30)   # seconds before the first, second, third restart
LOAD_DEADLINE = LOAD_TIMEOUT  # seconds before a model that never gets ready is killed and counted as a failed start
LOAD_POLL = 1           # seconds between checks while llama-server loads
MEMORY_WAIT = 60        # a restart waits at most this long for the stopped model's RAM to come back before loading
MEMORY_POLL = 2         # seconds between free-RAM checks while waiting
MEMORY_RECOVER = 0.9    # share of the stopped server's RAM that should return before the next one loads into free RAM


def tail_lines(n=20, path=None):
    try:
        return (path or LOG).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def classify_exit(code, tail):
    """"out of memory" when llama-server's last lines say so, else "crash"."""
    low = "\n".join(tail).lower()
    return "out of memory" if any(p in low for p in OOM_PATTERNS) else "crash"


def classify_error(error):
    """Kind of a failed /health request: "connection refused", "connection reset", "timeout" or "connection error"."""
    reason = getattr(error, "reason", error)
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused"
    if isinstance(reason, (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected)):
        return "connection reset"
    return "timeout" if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason) else "connection error"


def health(llama_port, timeout=None):
    """(True, None) when llama-server answers /health at all (a "loading" 503 counts: it is alive), else (False, kind)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{llama_port}/health", timeout=timeout or HEALTH_TIMEOUT):
            return True, None
    except urllib.error.HTTPError:
        return True, None
    except (OSError, http.client.HTTPException) as error:
        return False, classify_error(error)


def failure_record(kind, code=None, detail=None):
    reason = {"crash": "llama-server crashed", "out of memory": "llama-server ran out of memory"}.get(kind, f"llama-server stopped answering ({kind})")
    if code is not None:
        reason += f" (exit code {code})"
    if detail:
        reason += f": {detail}"
    return {"kind": kind, "reason": reason, "exitCode": code, "time": time.time(), "logTail": tail_lines()}


def idle_setting(config):
    """Minutes of no requests before the model is unloaded; 0 = never (the default)."""
    try:
        return max(0, int((config.get("host") or {}).get("idleUnloadMinutes") or 0))
    except (TypeError, ValueError):
        return 0


class Host:
    def __init__(self, config, log):
        self.config, self.log = config, log
        self.llama_port = config["server"]["port"]
        self.slots = int(budgets.effective(config)["slots"] or 2)
        self.token = secrets.token_urlsafe(24)
        self.started = time.time()
        self.model = self.context = self.error = None
        self.ready, self.adopted, self.restored = False, False, 0
        self.process = None
        self.stop_event = threading.Event()
        self.gate = Gate(lambda: self.slots, log)
        self.sessions = Sessions()
        diagnostics.RECORDER.context = lambda: (self.context, self.slots, self.model)
        self.listener = None
        self.state = "starting"       # starting, loading, ready, switching, unloaded, recovering, failed
        self.model_path = self.load_context = self.started_settings = None
        self.tlock = threading.Lock()  # one transition (switch, unload, load, recovery) at a time
        self.transition = None
        self.switch_error = None
        self.idle_minutes = idle_setting(config)
        self.restarts, self.recovered = [], 0
        self.failure, self.health_fails, self.last_health, self.health_at = None, 0, None, 0.0
        self.fail_marker, self.logged_health = None, False
        self._wlock, self._waking = threading.Lock(), False
        self.gate.wake = self.wake
        self.gate.on_refused = self.wait_recovered
        self.gate.models_fallback = self.models_list

    def models_list(self):
        """/v1/models while llama-server is down (unloaded, loading): the model the host would load."""
        if not self.model_path:
            return None
        return [{"id": self.model_path.name, "object": "model"}]

    def refresh_config(self):
        """Settings changed since the host started (`hearthwork context X N`) count at the next load."""
        try:
            from .onboard import load_config
            fresh = load_config()
            if fresh.get("server"):
                self.config = fresh
        except Exception:
            pass

    def record(self):
        write_private(HOST_FILE, json.dumps({"pid": os.getpid(), "port": port(self.config), "token": self.token,
                                             "model": self.model, "started": self.started, "adopted": self.adopted}))

    def status(self):
        transition = dict(self.transition) if self.transition else None
        if transition:
            transition["draining"] = len(self.gate.active) if transition.get("phase") == "draining" else 0
        idle_for = None
        if self.ready:
            idle_for = 0 if self.gate.active or self.gate.queue else time.monotonic() - self.gate.last_activity
        now = time.monotonic()
        return {"pid": os.getpid(), "port": port(self.config), "model": self.model, "context": self.context, "requestedContext": self.load_context, "settings": self.started_settings,
                "ready": self.ready, "error": self.error, "adopted": self.adopted, "restored": self.restored,
                "serverPort": self.llama_port, "started": self.started, "uptime": time.time() - self.started,
                "slots": self.slots, "state": self.state, "transition": transition, "switchError": self.switch_error,
                "idleMinutes": self.idle_minutes, "idleFor": idle_for, "sessions": self.sessions.open(),
                "recovery": {"restarts": len([t for t in self.restarts if now - t < RESTART_WINDOW]), "limit": RESTART_LIMIT,
                             "windowMinutes": RESTART_WINDOW // 60, "recovered": self.recovered, "lastFailure": self.failure,
                             "healthFails": self.health_fails, "lastHealth": self.last_health},
                **self.gate.snapshot()}

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
        name = path.rsplit("/", 1)[-1]
        if path == "/hearthwork/control/status" and handler.command == "GET":
            handler._send_json(200, self.status())
        elif path == "/hearthwork/control/diagnostics" and handler.command == "GET":  # ?seconds=N (default 300)
            seconds = int((urllib.parse.parse_qs(handler.path.partition("?")[2]).get("seconds") or ["300"])[0])
            payload = diagnostics.summarize(diagnostics.RECORDER.recent(seconds))
            payload["running"] = diagnostics.RECORDER.running()
            handler._send_json(200, payload)
        elif path == "/hearthwork/control/stop" and handler.command == "POST":
            handler._send_json(200, {"stopping": True})
            self.stop_event.set()
        elif path == "/hearthwork/control/session" and handler.command == "POST":
            try:
                body = json.loads(handler.read_body() or b"{}")
            except ValueError:
                body = {}
            action = body.get("action")
            if action == "register":
                record = self.sessions.register(body.get("client"), body.get("kind"), body.get("pid"))
            elif action == "unregister":
                self.sessions.unregister(body.get("client"))
                record = None
            else:
                handler._send_json(400, {"error": {"message": "action must be register or unregister"}})
                return True
            handler._send_json(200, {"session": record, "sessions": self.sessions.open()})
        elif handler.command == "POST" and name in ("switch", "idle", "load"):
            try:
                body = json.loads(handler.read_body() or b"{}")
            except ValueError:
                body = {}
            if name == "switch":
                code, payload = self.switch(body.get("model"), body.get("context"), bool(body.get("force")))
            elif name == "idle":
                code, payload = self.set_idle(body.get("minutes"))
            else:
                code, payload = self.request_load()
            handler._send_json(code, payload)
        else:
            handler._send_json(404, {"error": {"message": "not found"}})
        return True

    # -- llama-server --

    def set_ready(self):
        self.ready, self.state, self.transition = True, "ready", None
        self.health_fails, self.last_health = 0, None
        self.gate.hint = None
        self.gate.last_activity = time.monotonic()
        self.gate.set_hold(None)
        self.record()

    def kill_server(self):
        """Terminate the llama-server this host spawned (and its children); forget its record."""
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            if WINDOWS:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            try:
                proc.wait(10)
            except Exception:
                pass
        STATE.unlink(missing_ok=True)

    def release_server(self):
        """Stop the llama-server this host owns and wait (bounded) for its RAM to come back before the next one loads.
        Killing a 12+ GB server and starting the replacement at once makes Windows page the whole load (minutes
        instead of a minute), so measure what it held and how much RAM was free, stop it, then wait for the recovery."""
        from . import server
        from .onboard import memory_gb
        # only the server this host owns: a stale server.json can point at an unrelated (reused) pid
        pid = self.process.pid if self.process is not None else None
        previous = budgets.process_rss_gb(pid)
        free_before = memory_gb()[1]
        server.stop(self.config, quiet=True)  # saves the prompt cache first
        self.wait_memory_released(previous, free_before)

    def wait_memory_released(self, previous_gb, free_before, timeout=MEMORY_WAIT):
        """Wait for a stopped model's RAM to return, so the next server loads into free memory instead of paging.
        Bounded, and it stops early once the numbers settle (nothing left to reclaim) or the target is reached."""
        if not previous_gb or previous_gb < 2.0 or free_before is None:
            return  # no previous model of ours, or too small to matter
        from .onboard import memory_gb
        target = free_before + previous_gb * MEMORY_RECOVER
        self.log(f"waiting for the previous model's memory to be released ({previous_gb:.0f} GB)")
        end, best, steady = time.monotonic() + timeout, free_before, 0
        while time.monotonic() < end:
            time.sleep(MEMORY_POLL)
            free = memory_gb()[1]
            if free >= target:
                self.log(f"the previous model's memory is back ({free:.1f} GB free)")
                return
            if free > best + 0.5:  # still coming back
                best, steady = free, 0
            else:                  # the numbers stopped moving: waiting longer will not bring more back
                steady += 1
                if steady >= 2:
                    self.log(f"memory looks settled at {free:.1f} GB free")
                    return
        self.log(f"continuing with {memory_gb()[1]:.1f} GB free (waited {timeout:.0f} s for {previous_gb:.0f} GB)")

    def load(self, model, context):
        """Start llama-server on `model` and wait until it answers. Raises RuntimeError when it cannot."""
        from . import server
        self.release_server()  # a stale record or leftover process of an older start; wait for its memory first
        requested = context  # None = auto: recomputed at the next load, for the model and settings of that time
        context, source, rng = ctx.choose(self.config, model, context)
        self.log(ctx.describe(context, source, rng))
        cmd = server.command(self.config, model, context)
        self.log("Starting " + model.name)
        self.process = server.spawn_server(cmd)
        started = time.time()
        try:
            STATE.write_text(json.dumps({"pid": self.process.pid, "model": str(model), "port": self.llama_port}))
            while not server.ready(self.llama_port):
                if self.process.poll() is not None:
                    tail = tail_lines(10)
                    STATE.unlink(missing_ok=True)
                    kind = classify_exit(self.process.returncode, tail)
                    raise RuntimeError(f"llama-server stopped while loading (exit code {self.process.returncode}"
                                       f"{', out of memory' if kind == 'out of memory' else ''})\n" + "\n".join(tail))
                if self.stop_event.is_set():
                    raise RuntimeError("the host is stopping")
                if time.time() - started > LOAD_DEADLINE:
                    raise RuntimeError(f"the model did not load within {LOAD_DEADLINE:.0f} s")
                time.sleep(LOAD_POLL)
        except BaseException:
            self.kill_server()  # never leave a llama-server behind holding the GPU
            raise
        self.model_path, self.load_context = model, requested
        eff = budgets.effective(self.config)
        self.started_settings = {"fitTargetMiB": eff["fitTargetMiB"], "slots": eff["slots"], "threads": eff["threads"],
                                 "kvCacheType": self.config["server"].get("kvCacheType"), "batch": self.config["server"].get("batch"), "context": context,
                                 "server": cmd[0], "modelPath": str(model), "requested": requested}
        self.restored = server.restore_slots(self.llama_port, self.config["server"]["slots"])
        self.log(f"Model ready in {time.time() - started:.0f} s" + (f", restored {self.restored:,} cached prompt tokens" if self.restored else ""))
        self.model = server.served_model(self.llama_port, timeout=30) or model.stem
        self.context = server.running_context(self.llama_port)
        self.slots = props_slots(self.llama_port, self.slots)

    def start_model(self, model, context, adopt):
        from . import server
        try:
            if adopt:
                self.adopted = True
                name = server.served_model(self.llama_port, timeout=60)
                if not name:
                    raise RuntimeError(f"nothing answers on port {self.llama_port}")
                self.log(f"Using the model server already running on port {self.llama_port}: {name}")
                self.model = name
                self.context = server.running_context(self.llama_port)
                self.slots = props_slots(self.llama_port, self.slots)
            else:
                self.state = "loading"
                self.gate.set_hold("loading model")
                self.load(model, context)
            self.set_ready()
        except Exception as error:
            if self.stop_event.is_set():
                return
            self.error = str(error)
            self.log(f"The model did not start: {error}")
            self.stop_event.set()

    # -- transitions: switch, idle unload, recovery --

    def begin(self, kind, target=None):
        """Take the transition lock (False when another transition runs: only one at a time)."""
        if not self.tlock.acquire(False):
            return False
        self.transition = {"kind": kind, "target": target, "phase": kind, "since": time.time()}
        return True

    def end(self):
        self.transition = None
        if self.tlock.locked():
            self.tlock.release()

    def busy_message(self):
        kind = (self.transition or {}).get("kind")
        return "a switch is already in progress" if kind == "switch" else f"the host is busy ({kind or 'starting'}); try again in a moment"

    def switch(self, model, context, force):
        """Control API: switch to another model file. Returns (HTTP status, payload); the work runs in the background."""
        if self.adopted:
            return 409, {"error": {"message": "this model server was started by an older Hearthwork; run `hearthwork stop` first"}}
        path = Path(model) if model else None
        if not path or not path.is_file():
            return 400, {"error": {"message": f"model file not found: {model}"}}
        if not self.begin("switch", path.stem):
            return 409, {"error": {"message": self.busy_message()}}
        if self.state not in ("ready", "failed", "unloaded"):
            self.end()
            return 409, {"error": {"message": "the model is still starting; try again in a moment"}}
        sessions = self.sessions.open()
        if sessions and not force:  # a session that has not sent its first request yet still counts: it is using the model
            self.end()
            self.log(f"Switch refused: {sessions_message(sessions)}")
            return 409, {"error": {"message": sessions_message(sessions)}}
        self.switch_error = None
        self.transition["phase"] = "draining"
        threading.Thread(target=self._switch, args=(path, context, force), daemon=True).start()
        return 202, {"switching": path.stem}

    def _switch(self, path, context, force):
        from . import server
        previous, previous_context = self.model_path, self.load_context
        name = path.stem
        self.refresh_config()
        try:
            self.gate.set_hold(f"switching to {name}")
            self.transition["phase"] = "draining"
            self.log(f"Switching to {name}: no new requests start; {len(self.gate.active)} running")
            if not self.gate.wait_idle(FORCE_GRACE if force else SWITCH_DRAIN):
                if not force:
                    self.switch_error = (f"Not switching: {len(self.gate.active)} request(s) were still running after "
                                         f"{SWITCH_DRAIN} s. Try again later, or use --force.")
                    self.log(self.switch_error)
                    self.gate.set_hold(None if self.state == "ready" else self.gate.hold, self.gate.failed)
                    return
                n = self.gate.cancel_all(f"The model is being switched to {name}; this request was cancelled.")
                self.log(f"Switching to {name}: cancelled {n} running request(s)")
                self.gate.wait_idle(10)
            if force:
                closed = self.sessions.open()
                if closed:  # a forced switch closes the registrations it interrupts, so the next one is not refused
                    self.log("Switching to " + name + ": closing " + "; ".join(session_line(s) for s in closed))
                    self.sessions.clear()
            self.transition["phase"] = "loading"
            self.ready, self.state = False, "switching"
            self.gate.set_hold(f"loading {name}")
            self.release_server()  # saves the prompt cache, then waits for the old server's RAM to come back
            self.process = None
            try:
                self.load(path, context)
            except Exception as error:
                if self.stop_event.is_set():
                    return
                self.switch_error = f"Could not switch to {name}: {str(error).splitlines()[0]}"
                self.log(self.switch_error)
                if not previous:
                    raise
                self.log(f"Going back to {previous.stem}")
                self.load(previous, previous_context)  # failing again: the handler below marks the host failed
            self.set_ready()
        except Exception as error:
            if not self.stop_event.is_set():
                self.give_up(failure_record("crash", detail=(str(error).splitlines() or [""])[0]))
        finally:
            self.end()

    def set_idle(self, minutes):
        try:
            minutes = max(0, int(minutes or 0))
        except (TypeError, ValueError):
            return 400, {"error": {"message": "minutes must be a number, or 0 for off"}}
        self.idle_minutes = minutes
        self.gate.last_activity = time.monotonic()
        self.log("Idle unload: " + (f"after {minutes} min without requests" if minutes else "off"))
        return 200, {"idleMinutes": minutes}

    def unload(self):
        """Idle unload: save the prompt cache, stop llama-server, keep the host. Never with a request active or queued."""
        from . import server
        if not self.begin("unload"):
            return
        try:
            with self.gate.cond:
                if self.gate.active or self.gate.queue:
                    return
                self.gate.hold = "loading model"  # from here on new requests wait; the first one loads the model again
            self.log(f"No requests for {self.idle_minutes} min: unloading the model")
            self.ready, self.state = False, "unloading"
            try:
                self.release_server()  # stop it and wait for its memory to come back before the next load
            except Exception:
                self.log(traceback.format_exc())
            self.kill_server()  # whatever stop left running
        finally:
            if self.state == "unloading":
                self.state = "unloaded"
            self.end()

    def wake(self):
        """A request waits and the model is unloaded: load it in the background."""
        if self.state != "unloaded" or self.stop_event.is_set():
            return
        with self._wlock:
            if self._waking:
                return
            self._waking = True
        threading.Thread(target=self.reload, daemon=True).start()

    def reload(self):
        try:
            if not self.begin("loading", self.model_path.stem if self.model_path else None):
                return
            try:
                if self.state != "unloaded" or not self.model_path:
                    return
                self.state = "loading"
                self.refresh_config()
                self.log("A request is waiting: loading the model")
                try:
                    self.load(self.model_path, self.load_context)
                    self.set_ready()
                except Exception as error:
                    if not self.stop_event.is_set():
                        self.give_up(failure_record("crash", detail=(str(error).splitlines() or [""])[0]))
            finally:
                self.end()
        finally:
            self._waking = False

    def request_load(self):
        """Control API: load an unloaded model now, or clear a failed state and try again (`hearthwork start`)."""
        if self.state == "failed" and self.model_path:
            self.restarts.clear()
            self.log("Failed state cleared: starting the model again")
            self.gate.set_hold("loading model")
            self.state = "unloaded"
        if self.state == "unloaded":
            self.wake()
        return 200, {"state": self.state}

    def give_up(self, failure):
        self.failure = failure
        self.ready, self.state = False, "failed"
        message = (f"The model server failed: {failure['reason']}. See `hearthwork logs --server` for its output; "
                   f"`hearthwork start` tries again.")
        self.log("Giving up: " + failure["reason"])
        self.gate.set_hold(None, message)

    def tick(self):
        """Once a second from the main loop: process exit, /health, idle unload. Only the host restarts anything."""
        if self.state != "ready" or self.adopted or self.process is None or self.stop_event.is_set() or self.transition:
            return
        code = self.process.poll()
        if code is not None:
            self.recover(failure_record(classify_exit(code, tail_lines()), code))
            return
        now = time.monotonic()
        if now - self.health_at >= HEALTH_EVERY:
            self.health_at = now
            ok, kind = health(self.llama_port)
            if ok:
                self.health_fails, self.last_health, self.logged_health = 0, None, False
            else:
                busy = bool(self.gate.active)
                if self.health_fails == 0 or busy or self.fail_marker != self.gate.last_activity:
                    self.health_fails = 0 if busy else 1  # the window restarts: only silence with no activity counts
                    self.fail_marker = self.gate.last_activity
                else:
                    self.health_fails += 1
                self.last_health = kind
                if not self.logged_health:
                    self.logged_health = True
                    self.log(f"Health check of llama-server failed ({kind}); it may only be busy, so it is not restarted")
                # an alive process is only restarted when it keeps failing while nothing at all is running on it
                if self.health_fails >= HEALTH_FAILS and not self.gate.active and self.process.poll() is None:
                    self.recover(failure_record(kind, detail=f"no answer from /health for {self.health_fails} checks"))
                    return
        if self.idle_minutes:
            with self.gate.cond:
                idle = not self.gate.active and not self.gate.queue and now - self.gate.last_activity >= self.idle_minutes * 60
            if idle:
                self.unload()

    def recover(self, failure):
        """Restart llama-server after a failure: at most RESTART_LIMIT times per RESTART_WINDOW, with backoff."""
        if not self.begin("recovery"):
            return
        try:
            self.failure = failure
            self.ready, self.state = False, "recovering"
            self.gate.hint = failure["reason"]
            self.gate.set_hold("recovering")
            self.log("Model server failure: " + failure["reason"])
            for line in failure["logTail"][-5:]:
                self.log("  server.log: " + line)
            if self.process is not None and self.process.poll() is not None:
                STATE.unlink(missing_ok=True)  # its pid may be reused: never signal it
            while not self.stop_event.is_set():
                now = time.monotonic()
                self.restarts = [t for t in self.restarts if now - t < RESTART_WINDOW]
                if len(self.restarts) >= RESTART_LIMIT:
                    self.give_up(failure)
                    return
                delay = BACKOFF[min(len(self.restarts), len(BACKOFF) - 1)]
                self.transition["phase"] = f"restart {len(self.restarts) + 1} of {RESTART_LIMIT} in {delay} s"
                self.log(f"Restarting llama-server in {delay} s (restart {len(self.restarts) + 1} of {RESTART_LIMIT})")
                if self.stop_event.wait(delay):
                    return
                self.restarts.append(time.monotonic())
                self.transition["phase"] = "loading"
                try:
                    self.load(self.model_path, self.load_context)
                except Exception as error:
                    if self.stop_event.is_set():
                        return
                    text = str(error)
                    tail = tail_lines()
                    failure = failure_record(classify_exit(None, tail), detail=text.splitlines()[0])
                    self.failure = failure
                    self.gate.hint = failure["reason"]
                    self.log("Restart failed: " + failure["reason"])
                    continue
                self.recovered += 1
                self.log("llama-server is back")
                self.set_ready()
                return
        finally:
            self.end()

    def wait_recovered(self, timeout=MAX_WAIT):
        """A request was refused or reset before llama-server took it. Wait for the host's own restart (only the
        host restarts); True when the model server is up again, False when it failed or the wait ran out."""
        from . import server
        time.sleep(1.5)  # the monitor notices a dead process within a second
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.stop_event.is_set():
            if self.state == "failed":
                return False
            if self.state == "ready" and (self.process is None or self.process.poll() is None) and server.ready(self.llama_port):
                return True
            time.sleep(0.25)
        return False

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
                try:
                    host.tick()
                except Exception:
                    print(traceback.format_exc())
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


def call(info, path, method="GET", timeout=5, payload=None, errors=False):
    """JSON answer of the host's control API, or None. With `errors`, an HTTP error answer is returned too (it has "error")."""
    data = json.dumps(payload).encode() if payload is not None else b"" if method == "POST" else None
    request = urllib.request.Request(f"http://127.0.0.1:{info['port']}/hearthwork/control/{path}", method=method,
                                     data=data, headers={"Authorization": f"Bearer {info['token']}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return json.load(error) if errors else None
        except ValueError:
            return None
    except (OSError, ValueError):
        return None


def register_session(client, kind=None, pid=None):
    """Tell the local host that `client` (a local agent session id) has started using it, so a switch waits for it.
    No-op when no host answers. True when the host registered it."""
    if not client:
        return False
    info = read_info()
    if not info:
        return False
    answer = call(info, "session", "POST", payload={"action": "register", "client": client, "kind": kind,
                                                    "pid": pid or os.getpid()})
    return bool(answer) and answer.get("session") is not None


def unregister_session(client):
    """Tell the local host that `client` is done (best effort: a dead pid is dropped by the host anyway)."""
    if not client:
        return False
    info = read_info()
    if not info:
        return False
    return bool(call(info, "session", "POST", payload={"action": "unregister", "client": client}))


_session_lock = threading.Lock()
_open_sessions = {}          # client -> pid, for the process-exit safety net
_session_previous = {}       # signal -> handler that was installed before ours
_session_net_installed = False


def _track_session(client):
    """Remember a registered session so the safety net can drop it (process exit, Ctrl+C, kill via a signal)."""
    global _session_net_installed
    with _session_lock:
        _open_sessions[client] = os.getpid()
        install = not _session_net_installed
        _session_net_installed = True
    if not install:
        return
    atexit.register(_close_open_sessions)
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGINT", "SIGBREAK", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            _session_previous[number] = signal.getsignal(number)
            signal.signal(number, _session_signal)
        except (ValueError, OSError):
            pass


def _forget_session(client):
    with _session_lock:
        _open_sessions.pop(client, None)


def _close_open_sessions():
    """Deregister every session this process registered (idempotent: used by atexit and the signal safety net)."""
    with _session_lock:
        clients = list(_open_sessions)
        _open_sessions.clear()
    for client in clients:
        try:
            unregister_session(client)
        except Exception:
            pass


def _session_signal(number, frame):
    """A signal arrived: drop this process's sessions, then hand the signal on to whoever had it before us."""
    _close_open_sessions()
    previous = _session_previous.get(number, signal.SIG_DFL)
    if previous == signal.SIG_IGN:
        return
    if callable(previous):
        previous(number, frame)
        return
    try:
        signal.signal(number, signal.SIG_DFL)
        os.kill(os.getpid(), number)
    except (ValueError, OSError):
        pass


class Session:
    """A local agent session registered with the host. `close()` deregisters it; process exit, Ctrl+C and a signal
    do too (the safety net above), and a session whose process dies hard is dropped by the host as stale."""

    def __init__(self, client, kind=None):
        self.client, self.kind, self.closed = client, kind, False
        self.registered = bool(client) and register_session(client, kind, os.getpid())
        if self.registered:
            _track_session(client)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.client:
            _forget_session(self.client)
            unregister_session(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def session(client, kind=None):
    """Register `client` as a local agent session with the host; returns a Session (inert when no host answers)."""
    return Session(client, kind)


def status():
    """The host's status (model, context, slots, requests per client), or None when no host answers."""
    info = read_info()
    return call(info, "status") if info else None


def hosted_model():
    """Name of the model of a running host even while it is unloaded, loading or recovering (nothing answers on the
    model port then, but any command that needs the model just waits for it), else None."""
    got = status()
    return (got.get("model") or "model") if got else None


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
        if got and got.get("state") == "failed":
            why = ((got.get("recovery") or {}).get("lastFailure") or {}).get("reason", "unknown")
            print(f"\n{RED}The model server failed: {why} (see `hearthwork logs --server`){RESET}")
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


def wait_switch(timeout=LOAD_TIMEOUT):
    """Wait for a switch to end; prints what the host is doing. True when the host runs the new model."""
    started, shown = time.time(), None
    while time.time() - started < timeout:
        info = read_info()
        got = call(info, "status") if info else None
        if got is None:
            print(f"\n{RED}The host stopped answering during the switch (see `hearthwork logs`).{RESET}")
            return False
        if got.get("transition") is None and got.get("state") != "switching":
            if got.get("switchError"):
                print(f"\n{RED}{got['switchError']}{RESET}")
                return False
            if got.get("state") == "failed":
                print(f"\n{RED}The model server failed (see `hearthwork logs --server`).{RESET}")
                return False
            print(f" {GREEN}ready in {time.time() - started:.0f} s{RESET}")
            return True
        line = transition_text(got)
        if line != shown:
            print(f"\n{line}", end="", flush=True)
            shown = line
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\n{RED}The switch did not finish after {timeout:.0f} s (see `hearthwork logs`).{RESET}")
    return False


def apply_pending(config, info, got, force=False, drain=False, over_budget=False):
    """The model is the right one; settings changed since it started (budget, profile, KV type...) apply now through the
    switch path when nothing is running, else they stay pending (shown by `status`). The idle setting applies live."""
    wanted = budgets.effective(config)["idleMinutes"]
    if wanted != got.get("idleMinutes"):
        call(info, "idle", "POST", payload={"minutes": wanted})
    diffs = budgets.pending(config, got)
    if not diffs:
        return True
    busy = got["active"] + got["queued"]
    if busy and not force and not drain:
        print(f"{YELLOW}Settings waiting for the model to be idle ({'; '.join(diffs)}): {busy} request(s) running or waiting. "
              f"Run `hearthwork start` again later, or `hearthwork start --force` to interrupt them.{RESET}")
        return True
    model = Path((got.get("settings") or {})["modelPath"])
    if not budgets.guard(config, model, (got.get("settings") or {}).get("requested"), {"ramGB": budgets.measure().get("ramGB") or 0.0}, over_budget):
        return False
    answer = call(info, "switch", "POST", timeout=10, errors=True,
                  payload={"model": str(model), "context": (got.get("settings") or {}).get("requested"), "force": force}) or {}
    if answer.get("error"):
        print(f"{YELLOW}Settings not applied now: {answer['error'].get('message')}. They stay pending.{RESET}")
        return True
    print(f"Applying new settings ({'; '.join(diffs)}) to {model.name} (logs: `hearthwork logs`)", end="", flush=True)
    return wait_switch()


def ensure(config, model=None, context=None, force=False, drain=False, over_budget=False):
    """Make sure the host runs the model. Starts the host (and llama-server) when needed; adopts a llama-server an older
    Hearthwork started. A different `model` or `context` than the running one restarts the host's server, but refuses
    while requests are active unless `force` or `drain` (then the switch waits for them, up to 2 minutes, while new
    requests queue). Returns True when the model is ready."""
    from . import server
    llama = config["server"]["port"]
    info = read_info()
    got = call(info, "status") if info else None
    if info and not got:  # alive but silent: still starting
        return _wait_ready("Waiting for the Hearthwork host")
    if got and got.get("state") in ("failed", "unloaded"):  # `start` clears a failed state / loads an unloaded model
        if model is None or (got.get("model") == model.stem and context in (None, got.get("context"))):
            call(info, "load", "POST", payload={})
            return _wait_ready("Starting the model" if got["state"] == "unloaded" else "Restarting the model")
    if got and (got.get("transition") or {}).get("kind") == "switch" and model is not None \
            and got["transition"].get("target") == model.stem:
        return wait_switch()  # someone else is already switching to it
    if got:
        same = model is None or (got.get("model") == model.stem and (context is None or got.get("context") == context))
        if same:
            if not got.get("ready"):
                return _wait_ready("Waiting for the model")
            return apply_pending(config, info, got, force, drain, over_budget)
        busy = got["active"] + got["queued"]
        if busy and not force and not drain:
            print(f"{YELLOW}Not switching: {busy} request(s) are running or waiting ({got['active']} active, {got['queued']} queued). "
                  f"Wait for them, or use --force to interrupt them.{RESET}")
            return False
        if not budgets.guard(config, model, context, {"ramGB": budgets.measure().get("ramGB") or 0.0}, over_budget):
            return False
        if not got.get("adopted"):
            answer = call(info, "switch", "POST", timeout=10, errors=True,
                          payload={"model": str(model), "context": context, "force": force}) or {}
            if answer.get("error"):
                print(f"{YELLOW}Not switching: {answer['error'].get('message')}.{RESET}")
                return False
            print(f"Switching to {model.name} (logs: `hearthwork logs`)", end="", flush=True)
            return wait_switch()
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
    if not budgets.guard(config, model, context, over_budget=over_budget):
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


def transition_text(got):
    """What the host is doing instead of serving: switching to X (draining N), loading X, recovering, unloaded."""
    t, state = got.get("transition") or {}, got.get("state")
    if t.get("kind") == "switch":
        if t.get("phase") == "draining":
            return f"switching to {t.get('target')} (draining {t.get('draining', 0)})"
        return f"loading {t.get('target')}"
    if state == "recovering" or t.get("kind") == "recovery":
        why = ((got.get("recovery") or {}).get("lastFailure") or {}).get("reason", "failure")
        return f"recovering: {why} ({t.get('phase') or 'restarting'})"
    if state == "unloaded":
        return "unloaded (idle); loads on the next request"
    if t.get("kind") == "loading" or state == "loading":
        return f"loading {t.get('target') or 'the model'}"
    return state or "starting"


def idle_main(argv):
    """`hearthwork idle [minutes|off]`: show or set the idle unload (off by default; 30 is a good value)."""
    from .onboard import load_config, save_config
    if argv[:1] in (["-h"], ["--help"]):
        print("Usage: hearthwork idle [minutes|off]")
        return 0
    config = load_config()
    current = idle_setting(config)
    if not argv:
        now = f"after {current} min without requests" if current else "off"
        print(f"Idle unload: {now}. Set it with `hearthwork idle 30` or `hearthwork idle off`.")
        return 0
    word = argv[0].lower()
    try:
        minutes = 0 if word in ("off", "no", "never") else int(word)
        if minutes < 0:
            raise ValueError
    except ValueError:
        print("Usage: hearthwork idle [minutes|off]")
        return 2
    config.setdefault("host", {})["idleUnloadMinutes"] = minutes
    if (config.get("budget") or {}).get("base"):  # a preset/custom budget owns the idle setting: keep this as its override
        config["budget"].setdefault("custom", {})["idle"] = minutes
    save_config(config)
    info = read_info()
    if info:
        call(info, "idle", "POST", payload={"minutes": minutes})
    if minutes:
        print(f"Idle unload: after {minutes} min without requests (the model unloads, the host stays; the next request loads it).")
    else:
        print("Idle unload: off.")
    return 0


def status_lines(got):
    """Text lines for `hearthwork status` from a host status."""
    lines = [f"host: {GREEN}running{RESET}  pid {got['pid']}, port {got['port']}, up {duration(got['uptime'])}"
             + (f"  {DIM}(model server started by an older Hearthwork){RESET}" if got.get("adopted") else "")]
    state, recovery = got.get("state"), got.get("recovery") or {}
    last = recovery.get("lastFailure")
    if got.get("error"):
        lines.append(f"model: {RED}{got['error']}{RESET}")
    elif state == "failed":
        lines.append(f"model: {RED}failed: {(last or {}).get('reason', 'unknown')}{RESET}  (`hearthwork logs --server`; `hearthwork start` tries again)")
    elif state in ("switching", "recovering", "unloaded") or got.get("transition"):
        lines.append(f"model: {YELLOW}{transition_text(got)}{RESET}")
    elif got.get("ready"):
        n_ctx = got.get("context")
        lines.append(f"model: {GREEN}{got['model']}{RESET}  (port {got['serverPort']}"
                     + (f", context {n_ctx:,} = {ctx.format_k(n_ctx)}" if n_ctx else "") + f", {slots_text(got['slots'])})")
    else:
        lines.append(f"model: {YELLOW}loading{RESET}")
    if got.get("switchError"):
        lines.append(f"switch: {YELLOW}{got['switchError']}{RESET}")
    minutes = got.get("idleMinutes")
    idle_text = f"after {minutes} min without requests" if minutes else "off"
    lines.append(f"idle unload: {idle_text}" + (f"  (idle for {duration(got['idleFor'])})" if minutes and got.get("idleFor") else ""))
    if last and (state in ("failed", "recovering") or recovery.get("recovered")):
        lines.append(f"recovery: {recovery.get('restarts', 0)} of {recovery.get('limit')} restarts in the last {recovery.get('windowMinutes')} min, "
                     f"{recovery.get('recovered', 0)} recovered; last failure: {last['reason']}")
    elif recovery.get("lastHealth"):
        lines.append(f"health: last check failed ({recovery['lastHealth']}); a busy server is not restarted")
    sessions = got.get("sessions") or []
    if sessions:
        lines.append(f"sessions: {len(sessions)} open  (a switch waits for them; `hearthwork start --force` interrupts)")
        for session in sessions:
            lines.append(f"  {session_line(session)}")
    lines.append(f"requests: {got['active']} active / {slots_text(got['limit'])}, {got['queued']} queued"
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
