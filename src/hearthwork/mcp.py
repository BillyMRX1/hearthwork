"""`hearthwork mcp`: a stdio MCP server that lets any agent hand coding tasks to the local model (Claude Code or Codex
running on it). Newline-delimited JSON-RPC 2.0 on stdin/stdout; nothing else is ever printed to stdout.

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

VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
MAX_WAIT = 55

TASK_PROPERTIES = {
    "task": {"type": "string", "description": "Complete task description: files to touch, interfaces, and the command that verifies it (e.g. the test command)."},
    "agent": {"type": "string", "enum": ["claude", "codex"], "default": "claude", "description": "Which agent runs it on the local model."},
    "cwd": {"type": "string", "description": "Absolute folder to work in (default: the server's folder)."},
    "allow_commands": {"type": "array", "items": {"type": "string"},
                       "description": 'Command patterns the claude agent may run, e.g. ["pytest *", "python *"]. Without them it can only read and edit files. Codex uses its own sandbox instead.'},
    "timeout_seconds": {"type": "integer", "default": 900, "description": "The agent is killed after this long."},
}


def guidance(slots):
    return (f"Use for small, well-specified coding tasks: name the files, the interfaces and the command that verifies the result. "
            f"It is a local model, so keep each task focused, and review what it changed afterwards. "
            f"You can run up to {slots} tasks in parallel (the server's slots); give parallel tasks different files.")


def tools(slots):
    note = guidance(slots)
    return [
        {"name": "local_task_start", "description": f"Start a coding task on the local model in the background; returns an id. Then poll local_task_result. {note}",
         "inputSchema": {"type": "object", "properties": TASK_PROPERTIES, "required": ["task"]}},
        {"name": "local_task_result", "description": f"Wait up to wait_seconds (max {MAX_WAIT}) for a task started with local_task_start. Returns status running (call again), done or failed, and the report (files created/modified/deleted and the agent's final message) when finished.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "wait_seconds": {"type": "integer", "default": 50, "maximum": MAX_WAIT}}, "required": ["id"]}},
        {"name": "local_task", "description": f"Run a coding task on the local model and wait for the report (blocking; can take minutes, so the caller's tool timeout must be long: for Codex raise tool_timeout_sec). Prefer local_task_start + local_task_result when unsure. {note}",
         "inputSchema": {"type": "object", "properties": TASK_PROPERTIES, "required": ["task"]}},
        {"name": "local_model_status", "description": "Which local model is available (host, model, context) and whether it is reachable.",
         "inputSchema": {"type": "object", "properties": {}}},
    ]


class Tasks:
    """Background `hearthwork task` processes, by id."""

    def __init__(self, default_cwd=None):
        self.items = {}
        self.cwd = default_cwd or os.getcwd()

    def command(self, args):
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        agent = args.get("agent") or "claude"
        if agent not in ("claude", "codex"):
            raise ValueError("agent must be 'claude' or 'codex'")
        allow = args.get("allow_commands") or []
        if not isinstance(allow, list) or not all(isinstance(a, str) for a in allow):
            raise ValueError("allow_commands must be a list of strings")
        timeout = args.get("timeout_seconds", 900)
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        command = [sys.executable, "-m", "hearthwork", "task", "--json", "--agent", agent, "--timeout", str(timeout),
                   "--cwd", args.get("cwd") or self.cwd]
        for pattern in allow:
            command += ["--allow", pattern]
        return command + ["-"], task

    def start(self, args):
        command, task = self.command(args)
        item = {"done": threading.Event(), "report": None, "error": None, "started": time.time()}
        task_id = uuid.uuid4().hex[:8]
        self.items[task_id] = item

        def work():
            try:
                done = subprocess.run(command, input=task, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace")
                try:
                    item["report"] = json.loads(done.stdout)
                except ValueError:
                    item["error"] = (done.stderr or done.stdout or f"exit {done.returncode}").strip()[-1500:]
            except Exception as error:
                item["error"] = str(error)
            finally:
                item["done"].set()

        threading.Thread(target=work, daemon=True).start()
        return task_id

    def result(self, task_id, wait):
        item = self.items.get(task_id)
        if not item:
            raise ValueError(f"unknown task id '{task_id}'")
        item["done"].wait(max(0, min(wait, MAX_WAIT)))
        if not item["done"].is_set():
            return {"id": task_id, "status": "running", "elapsed_seconds": round(time.time() - item["started"])}
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
        self.tasks = Tasks(cwd)

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
            elif name == "local_task":
                task_id = self.tasks.start(args)
                data = self.tasks.result(task_id, 10 ** 9)
            else:
                data = model_status()
        except ValueError as error:
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}
        failed = name in ("local_task", "local_task_result") and data.get("status") == "failed"
        return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}], "isError": failed}


def reply(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error_reply(msg_id, code, text):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}


def serve(slots=None):
    from .onboard import load_config
    if slots is None:
        slots = load_config().get("server", {}).get("slots", 2)
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
