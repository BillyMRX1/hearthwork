"""`hearthwork mcp`: a stdio MCP server that lets any agent hand coding tasks to the local model (Claude Code, Codex, OpenCode,
Aider or Qwen Code running on it; see `hearthwork agents`). Newline-delimited JSON-RPC 2.0 on stdin/stdout; nothing else is ever printed to stdout.

  hearthwork mcp                       run the server (what the agent's MCP config starts)
  hearthwork mcp install claude|codex  register it with that agent
  hearthwork mcp install print         JSON snippet for other clients (Cursor, OpenCode, Claude Desktop, ...)

Tasks run as `hearthwork task` child processes, so a task survives a short tool timeout of the caller:
local_task_start returns an id at once and local_task_result waits up to 55 s per call.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

from . import __version__
from .harnesses import HARNESSES
from .procs import KillOnClose, kill_tree, new_group_flags, parallel_slots, setup_stdio

VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
MAX_WAIT = 55

TASK_PROPERTIES = {
    "task": {"type": "string", "description": "Complete task description: files to touch, interfaces, and the command that verifies it (e.g. the test command)."},
    "agent": {"type": "string", "enum": list(HARNESSES), "default": "claude", "description": "Which agent runs it on the local model."},
    "cwd": {"type": "string", "description": "Absolute folder to work in (default: the server's folder)."},
    "allow_commands": {"type": "array", "items": {"type": "string"},
                       "description": 'Command patterns the agent may run, e.g. ["pytest *", "python *"] (claude, opencode, qwen). Without them it can only read and edit files. Codex uses its own sandbox instead; aider never runs commands.'},
    "timeout_seconds": {"type": "integer", "default": 900, "description": "The agent is killed after this long."},
}


def guidance(slots):
    return (f"Use for small, well-specified coding tasks: name the files, the interfaces and the command that verifies the result. "
            f"It is a local model, so keep each task focused, and review what it changed afterwards. "
            f"You can run up to {slots} tasks in parallel (the server's slots; more are queued and start when one finishes); give parallel tasks different files.")


def tools(slots):
    note = guidance(slots)
    return [
        {"name": "local_task_start", "description": f"Start a coding task on the local model in the background; returns an id. Then poll local_task_result. {note}",
         "inputSchema": {"type": "object", "properties": TASK_PROPERTIES, "required": ["task"]}},
        {"name": "local_task_result", "description": f"Wait up to wait_seconds (max {MAX_WAIT}) for a task started with local_task_start. Returns status running (call again), done or failed, and the report (files created/modified/deleted and the agent's final message) when finished.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "wait_seconds": {"type": "integer", "default": 50, "maximum": MAX_WAIT}}, "required": ["id"]}},
        {"name": "local_task", "description": f"Run a coding task on the local model and wait for the report (blocking; can take minutes, so the caller's tool timeout must be long: for Codex raise tool_timeout_sec). Prefer local_task_start + local_task_result when unsure. {note}",
         "inputSchema": {"type": "object", "properties": TASK_PROPERTIES, "required": ["task"]}},
        {"name": "local_task_cancel", "description": "Cancel a task started with local_task_start (queued or running); a running one is killed with everything it started.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
        {"name": "local_model_status", "description": "Which local model is available (host, model, context) and whether it is reachable.",
         "inputSchema": {"type": "object", "properties": {}}},
    ]


class Tasks:
    """Background `hearthwork task` processes, by id. At most `slots` run at once (the model server's parallel
    sessions); the rest wait in a queue and start as slots free up. Everything running is killed on shutdown()."""

    def __init__(self, default_cwd=None, slots=2):
        self.items = {}
        self.queue = []  # ids waiting, oldest first
        self.slots = max(1, slots)
        self.cwd = default_cwd or os.getcwd()
        self.lock = threading.RLock()
        self.job = KillOnClose()  # Windows: children die with this process, even if it is killed hard
        self.closed = False

    def command(self, args):
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        agent = args.get("agent") or "claude"
        if agent not in HARNESSES:
            raise ValueError("agent must be one of: " + ", ".join(HARNESSES))
        allow = args.get("allow_commands") or []
        if not isinstance(allow, list) or not all(isinstance(a, str) for a in allow):
            raise ValueError("allow_commands must be a list of strings")
        timeout = args.get("timeout_seconds", 900)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        command = [sys.executable, "-m", "hearthwork", "task", "--json", "--agent", agent, "--timeout", str(timeout),
                   "--cwd", args.get("cwd") or self.cwd]
        for pattern in allow:
            command += ["--allow", pattern]
        return command + ["-"], task

    def start(self, args):
        """Queue a task (it starts at once when a slot is free). Returns its id."""
        command, task = self.command(args)
        with self.lock:
            if self.closed:
                raise ValueError("the server is shutting down")
            task_id = uuid.uuid4().hex[:8]
            self.items[task_id] = {"done": threading.Event(), "report": None, "error": None, "created": time.time(),
                                   "started": None, "state": "queued", "process": None, "command": command, "task": task}
            self.queue.append(task_id)
            self.pump()
        return task_id

    def running(self):
        return [i for i, item in self.items.items() if item["state"] == "running"]

    def pump(self):
        """Start queued tasks while slots are free."""
        with self.lock:
            while self.queue and len(self.running()) < self.slots and not self.closed:
                task_id = self.queue.pop(0)
                item = self.items[task_id]
                item["state"], item["started"] = "running", time.time()
                threading.Thread(target=self.work, args=(task_id,), daemon=True).start()

    def spawn(self, command):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HEARTHWORK_MCP="1")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", env=env, **new_group_flags())
        self.job.add(process)
        return process

    def work(self, task_id):
        item = self.items[task_id]
        try:
            with self.lock:
                if item["state"] != "running":  # cancelled before it started
                    return
                process = item["process"] = self.spawn(item["command"])
            stdout, stderr = process.communicate(item["task"])
            if item["state"] == "cancelled":
                return
            try:
                item["report"] = json.loads(stdout)
            except ValueError:
                item["error"] = (stderr or stdout or f"exit {process.returncode}").strip()[-1500:]
        except Exception as error:
            if item["state"] != "cancelled":
                item["error"] = str(error)
        finally:
            with self.lock:
                if item["state"] == "running":
                    item["state"] = "finished"
            item["done"].set()
            self.pump()

    def cancel(self, task_id):
        with self.lock:
            item = self.items.get(task_id)
            if not item:
                raise ValueError(f"unknown task id '{task_id}'")
            if item["state"] not in ("queued", "running"):
                return {"id": task_id, "status": "cancelled" if item["state"] == "cancelled" else "finished",
                        "note": "the task had already ended"}
            was = item["state"]
            item["state"] = "cancelled"
            if task_id in self.queue:
                self.queue.remove(task_id)
            if item["process"] is not None:
                kill_tree(item["process"])
            item["done"].set()
            self.pump()
            return {"id": task_id, "status": "cancelled", "was": was}

    def shutdown(self):
        """Refuse new tasks, drop the queue, kill every running task and its process tree."""
        with self.lock:
            self.closed = True
            self.queue.clear()
            for item in self.items.values():
                if item["state"] in ("queued", "running"):
                    item["state"] = "cancelled"
                    if item["process"] is not None:
                        kill_tree(item["process"])
                    item["done"].set()

    def result(self, task_id, wait):
        item = self.items.get(task_id)
        if not item:
            raise ValueError(f"unknown task id '{task_id}'")
        item["done"].wait(max(0, min(wait, MAX_WAIT)))
        if not item["done"].is_set():
            if item["state"] == "queued":
                position = self.queue.index(task_id) + 1 if task_id in self.queue else 1
                return {"id": task_id, "status": "queued", "position": position,
                        "detail": f"waiting for a free session of the model ({self.slots} at a time)"}
            return {"id": task_id, "status": "running", "elapsed_seconds": round(time.time() - (item["started"] or time.time()))}
        if item["state"] == "cancelled":
            return {"id": task_id, "status": "cancelled"}
        report = item["report"]
        ok = bool(report) and report.get("status") == "ok"
        out = {"id": task_id, "status": "done" if ok else "failed"}
        if report:
            out["report"] = report
        if item["error"]:
            out["error"] = item["error"]
        return out


def model_status():
    from .onboard import load_config
    config = load_config()
    try:
        if config.get("remote"):
            from .remote import RemoteError, session
            remote = config["remote"]
            host = remote.get("hostName") or remote["host"]
            try:
                name, context = session(config)
            except RemoteError as error:
                return {"host": host, "model": None, "context": None, "reachable": False, "detail": str(error).splitlines()[0]}
            return {"host": host, "model": name, "context": context, "reachable": True}
        from .server import running_context, served_model
        port = config.get("server", {}).get("port", 8001)
        name = served_model(port)
        return {"host": "local", "model": name, "context": running_context(port) if name else None, "reachable": bool(name),
                **({} if name else {"detail": "no model running; a task starts the last used one"})}
    except Exception as error:
        return {"host": "local", "model": None, "context": None, "reachable": False, "detail": str(error)}


class Server:
    def __init__(self, slots=2, cwd=None):
        self.slots = slots
        self.tasks = Tasks(cwd, slots)

    def handle(self, message):
        """Response dict for a JSON-RPC message, or None (notifications and responses get no answer)."""
        if not isinstance(message, dict):
            return error_reply(None, -32600, "Invalid Request")
        method, msg_id = message.get("method"), message.get("id")
        if method is None or "id" not in message:
            return None  # a notification (or a client's response): never answered
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return error_reply(msg_id, -32602, "params must be an object")
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                return reply(msg_id, {"protocolVersion": asked if asked in VERSIONS else VERSIONS[0],
                                      "capabilities": {"tools": {}},
                                      "serverInfo": {"name": "hearthwork", "version": __version__},
                                      "instructions": guidance(self.slots)})
            if method == "ping":
                return reply(msg_id, {})
            if method == "tools/list":
                return reply(msg_id, {"tools": tools(self.slots)})
            if method == "tools/call":
                name, args = params.get("name"), params.get("arguments") or {}
                if name not in {t["name"] for t in tools(self.slots)}:
                    return error_reply(msg_id, -32602, f"Unknown tool: {name}")
                if not isinstance(args, dict):
                    return error_reply(msg_id, -32602, "arguments must be an object")
                return reply(msg_id, self.call(name, args))
            return error_reply(msg_id, -32601, f"Method not found: {method}")
        except Exception as error:  # a tool failure is a result the model can read, not a protocol error
            return error_reply(msg_id, -32603, str(error))

    def call(self, name, args):
        try:
            if name == "local_task_start":
                data = {"id": self.tasks.start(args)}
            elif name == "local_task_result":
                if not isinstance(args.get("id"), str):
                    raise ValueError("id is required")
                wait = args.get("wait_seconds", 50)
                data = self.tasks.result(args["id"], wait if isinstance(wait, (int, float)) else 50)
            elif name == "local_task_cancel":
                if not isinstance(args.get("id"), str):
                    raise ValueError("id is required")
                data = self.tasks.cancel(args["id"])
            elif name == "local_task":
                task_id = self.tasks.start(args)
                data = self.tasks.result(task_id, 10 ** 9)
            else:
                data = model_status()
        except ValueError as error:
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}
        failed = name in ("local_task", "local_task_result") and data.get("status") in ("failed", "cancelled")
        return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}], "isError": failed}


def reply(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error_reply(msg_id, code, text):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}


def serve(slots=None):
    import atexit
    import signal
    from .onboard import load_config
    setup_stdio()
    if slots is None:
        slots = parallel_slots(load_config())
    server = Server(slots)
    out_lock = threading.Lock()

    def send(message):
        with out_lock:
            sys.stdout.buffer.write((json.dumps(message) + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()

    def dispatch(message):
        response = server.handle(message)
        if response is not None:
            send(response)

    def stop(*_):
        server.tasks.shutdown()

    def on_signal(number, frame):
        stop()
        os._exit(143)

    atexit.register(stop)
    for name in ("SIGTERM", "SIGINT", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), on_signal)
            except (ValueError, OSError):
                pass
    try:
        for line in sys.stdin.buffer:  # a thread per request: local_task blocks, and pings must still be answered
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                send(error_reply(None, -32700, "Parse error"))
                continue
            threading.Thread(target=dispatch, args=(message,), daemon=True).start()
    finally:
        stop()  # the client went away (stdin closed): leave no orphaned tasks
    return 0


# ---------- install ----------

def executable():
    """Absolute path of the hearthwork launcher (GUI clients may have another PATH)."""
    found = shutil.which("hearthwork")
    if found:
        return os.path.abspath(found)
    argv0 = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return argv0 if os.path.isfile(argv0) and not argv0.endswith(("__main__.py", ".py")) else sys.executable


def server_command():
    """[command, *args] that starts the MCP server."""
    exe = executable()
    return [exe, "mcp"] if os.path.basename(exe).lower().startswith("hearthwork") else [exe, "-m", "hearthwork", "mcp"]


def snippet():
    command = server_command()
    return {"mcpServers": {"hearthwork": {"command": command[0], "args": command[1:]}}}


def install(target):
    command = server_command()
    if target == "print":
        print(json.dumps(snippet(), indent=2))
        print("\nPaste this into the client's MCP settings (Cursor: ~/.cursor/mcp.json, Claude Desktop: claude_desktop_config.json, "
              "OpenCode: the \"mcp\" section of opencode.json uses command as a list).", file=sys.stderr)
        return 0
    binary = shutil.which(target)
    if not binary:
        print(f"{target} is not installed.", file=sys.stderr)
        return 1
    add = ([binary, "mcp", "add", "--scope", "user", "hearthwork", "--", *command] if target == "claude"
           else [binary, "mcp", "add", "hearthwork", "--", *command])
    done = subprocess.run(add, capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    text = (done.stdout + done.stderr).strip()
    if done.returncode != 0 and "already exists" in text:
        print(f"Hearthwork is already registered with {target}. (`{target} mcp remove hearthwork` to redo it.)")
        return 0
    print(text)
    if done.returncode == 0:
        print(f"Registered with {target}: {' '.join(command)}")
        if target == "codex":
            print("Codex gives MCP tools about 60 s by default. local_task_start + local_task_result fit that; for the blocking "
                  "local_task add to ~/.codex/config.toml under [mcp_servers.hearthwork]:\n  tool_timeout_sec = 900")
    return done.returncode


def main(argv):
    if argv[:1] == ["install"]:
        if len(argv) != 2 or argv[1] not in ("claude", "codex", "print"):
            print("usage: hearthwork mcp install claude|codex|print", file=sys.stderr)
            return 2
        return install(argv[1])
    if argv:
        print("usage: hearthwork mcp [install claude|codex|print]", file=sys.stderr)
        return 2
    return serve()
