"""`hearthwork task`: hand one coding task to the local (or connected) model through a coding agent (Claude Code,
Codex, OpenCode, Aider, Qwen Code; see `hearthwork agents`), without a terminal, and get a report back. For other agents
that want "our Claude" / "our Codex" as a subagent.

  hearthwork task [--agent claude|codex|...] [--allow "pytest *"]... [--cwd DIR] [--timeout SECONDS] [--json] [--dry-run] "TASK"

TASK may be "-" to read it from stdin (no shell quoting). The report goes to stdout (plain text, or JSON with
--json); progress and logs go to stderr. Exit code 0 only when the agent finished without error.

Permissions: Claude Code can read, search and edit files, and run only the commands matching --allow (each becomes
`Bash(<pattern>)`); anything else is denied. Codex runs with its workspace-write sandbox: it may edit inside the
folder and run commands there, and the sandbox (not --allow, which Codex ignores) decides the rest. OpenCode and Qwen Code get the same rule as Claude Code
(edits plus the --allow commands). Aider only edits files: it cannot limit the commands it runs, so they are off.
--dry-run prints the command, environment and generated files that would be used, and starts nothing.
"""
import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .harnesses import HARNESSES, installed, prepare, preview

SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__"}
MAX_FILES = 5000
SUFFIX = ("\n\nWork only inside the current folder. When you are done, finish with a short summary of what you "
          "changed and how you verified it.")


def log(text):
    print(text, file=sys.stderr, flush=True)


# ---------- file snapshot ----------

def snapshot(folder, limit=MAX_FILES):
    """{relative path: (size, mtime_ns)} of the files under `folder`, without .git, node_modules, .venv and
    __pycache__, at most `limit` files."""
    found = {}

    def walk(path, prefix):
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if len(found) >= limit:
                return
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIP_DIRS:
                        walk(entry.path, prefix + entry.name + "/")
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    found[prefix + entry.name] = (info.st_size, info.st_mtime_ns)
            except OSError:
                pass

    walk(str(folder), "")
    return found


def diff_snapshots(before, after):
    """(created, modified, deleted) path lists, each with sizes for the ones that exist now."""
    created = [{"path": p, "size": after[p][0]} for p in after if p not in before]
    modified = [{"path": p, "size": after[p][0]} for p in after if p in before and after[p] != before[p]]
    deleted = [{"path": p} for p in before if p not in after]
    return created, modified, deleted


# ---------- the agent command ----------

def task_args(agent, task, allow=(), last_message=None, prompt_file=None):
    """Arguments after the agent's name. The task itself goes in through stdin or a file (never on the command line)."""
    return HARNESSES[agent]["task"](allow, last_message, prompt_file)[0]


def parse_result(agent, stdout, stderr, returncode, last_message=None):
    """(final message, error text or None) from what the agent printed."""
    return HARNESSES[agent]["result"](stdout, stderr, returncode, last_message)


def kill_tree(process):
    """Kill the agent and everything it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.kill()
    except OSError:
        pass


def run_process(command, env, cwd, stdin_text, timeout):
    """(returncode, stdout, stderr, timed_out). The process tree is killed at the timeout."""
    extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", **extra)
    out, err = [], []
    readers = [threading.Thread(target=lambda s=s, b=b: b.append(s.read()), daemon=True)
               for s, b in ((process.stdout, out), (process.stderr, err))]
    for reader in readers:
        reader.start()
    try:
        process.stdin.write(stdin_text)
        process.stdin.close()
    except OSError:
        pass
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(process)
        process.wait()
    for reader in readers:
        reader.join(5)
    return process.returncode, "".join(out), "".join(err), timed_out


# ---------- model ----------

def model_for_task(config):
    """(port or None, model name, context, remote or None), starting the local model when none is running.
    Never asks anything. Raises SystemExit with the reason when no model can be used."""
    from .server import agent_context, served_model
    if config.get("remote"):
        from .remote import RemoteError, session
        try:
            name, context = session(config)
        except RemoteError as error:
            sys.exit(str(error))
        return None, name, context, config["remote"]
    port = config["server"]["port"]
    from .server import port_open
    name = served_model(port) or (port_open(port) and served_model(port, timeout=60))  # busy: wait, never restart it
    if name:
        return port, name, agent_context(config), None
    if port_open(port):
        raise SystemExit(f"Port {port} is in use but no model answers; run `hearthwork stop` first.")
    from .onboard import find_models
    from .server import start_background
    models = find_models(config["modelsDir"])
    last = config.get("lastModel")
    chosen = next((m for m in models if str(m) == last), None) or (models[0] if len(models) == 1 else None)
    if not chosen:
        sys.exit("No model is running and none can be chosen without asking you. Run `hearthwork start` once "
                 "(it asks which model), then try again." if models else
                 "No model is running and there are no models yet. Download one first: `hearthwork model <link>`.")
    with contextlib.redirect_stdout(sys.stderr):  # the start-up chatter must not reach stdout
        ok = start_background(config, chosen)
    name = served_model(port) if ok else None
    if not name:
        sys.exit("The model server did not start; see the messages above.")
    return port, name, agent_context(config), None


# ---------- the task ----------

def preview_model(config):
    """(model name, context, remote or None) for a dry run: the running model if there is one; nothing is started."""
    if config.get("remote"):
        from .remote import RemoteError, session
        try:
            return (*session(config), config["remote"])
        except RemoteError:
            return "<shared-model>", 32768, config["remote"]
    from .server import agent_context, served_model
    name = served_model(config.get("server", {}).get("port", 8001))
    return (name, agent_context(config), None) if name else ("<model-name>", 32768, None)


def dry_run_task(config, task, agent, allow, cwd=None):
    """`hearthwork task --dry-run`: print what the task would run, start nothing."""
    name, context, remote = preview_model(config)
    arguments, extra = HARNESSES[agent]["task"](allow, "<last-message-file>", "<task-file>", cwd=os.path.abspath(cwd or os.getcwd()))
    prompt = HARNESSES[agent]["prompt"]
    preview(agent, name, context, args=arguments, extra_env=extra, remote=remote,
            note=f"The task ({len(task)} characters, plus a short closing instruction) is given to the agent "
                 + ("on stdin." if prompt == "stdin" else "in a temporary file (<task-file>)."))


def run_task(config, task, agent="claude", allow=(), cwd=None, timeout=900):
    """Run one task and return the report dict."""
    folder = os.path.abspath(cwd or os.getcwd())
    if not os.path.isdir(folder):
        sys.exit(f"Folder not found: {folder}")
    if not installed(agent):
        sys.exit(f"{HARNESSES[agent]['title']} is not installed. Install: {HARNESSES[agent]['install']}")
    port, model, context, remote = model_for_task(config)
    host = (remote.get("hostName") or remote["host"]) if remote else "local"
    temporary = []  # the file the final message goes to (Codex), and the task for agents that cannot read stdin
    for prefix in ("last", "task"):
        handle, path = tempfile.mkstemp(prefix=f"hearthwork-{prefix}-", suffix=".txt")
        os.close(handle)
        temporary.append(path)
    last_message, prompt_file = temporary
    text = task.rstrip() + HARNESSES[agent].get("suffix", SUFFIX)
    try:
        arguments, extra = HARNESSES[agent]["task"](allow, last_message, prompt_file, cwd=folder)
        stdin_text = text
        if HARNESSES[agent]["prompt"] == "file":
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(text)
            stdin_text = ""
        prepared = prepare(agent, port, model, context, args=arguments, extra_env=extra, remote=remote, clean=True)
        if not prepared:
            sys.exit(1)
        command, env = prepared
        log(f"hearthwork task: {HARNESSES[agent]['title']} with {model} ({host}) in {folder}")
        before = snapshot(folder)
        started = time.time()
        returncode, out, err, timed_out = run_process(command, env, folder, stdin_text, timeout)
        duration = time.time() - started
        after = snapshot(folder)
        message, error = parse_result(agent, out, err, returncode, last_message)
    finally:
        for path in temporary:
            with contextlib.suppress(OSError):
                os.unlink(path)
    status = "timeout" if timed_out else "error" if error else "ok"
    if timed_out:
        error = f"timed out after {timeout} s; the agent was killed"
    created, modified, deleted = diff_snapshots(before, after)
    return {"agent": agent, "model": model, "host": host, "cwd": folder, "duration_seconds": round(duration, 1),
            "status": status, "exit_code": returncode, "error": error, "created": created, "modified": modified,
            "deleted": deleted, "files_capped": len(before) >= MAX_FILES or len(after) >= MAX_FILES, "message": message}


def format_report(report):
    def files(label, items):
        if not items:
            return []
        return [f"{label}:"] + [f"  {i['path']}" + (f" ({i['size']} bytes)" if "size" in i else "") for i in items]

    lines = [f"agent: {report['agent']}   model: {report['model']}   host: {report['host']}",
             f"folder: {report['cwd']}",
             f"duration: {report['duration_seconds']} s   status: {report['status']}   exit code: {report['exit_code']}"]
    if report["error"]:
        lines.append(f"error: {report['error']}")
    changes = files("created", report["created"]) + files("modified", report["modified"]) + files("deleted", report["deleted"])
    lines += changes or ["files: no changes"]
    if report.get("files_capped"):
        lines.append(f"(only the first {MAX_FILES} files were compared)")
    lines += ["", "final message:", report["message"] or "(none)"]
    return "\n".join(lines)


def exit_code(report):
    return 0 if report["status"] == "ok" else 124 if report["status"] == "timeout" else 1


def main(argv):
    parser = argparse.ArgumentParser(prog="hearthwork task", description="Run one coding task with the local model through a coding agent.")
    parser.add_argument("task", help='what to do; "-" reads it from stdin')
    parser.add_argument("--agent", choices=sorted(HARNESSES), default="claude")
    parser.add_argument("--allow", action="append", default=[], metavar="PATTERN",
                        help='command pattern the agent may run, e.g. "pytest *" (not Codex or Aider; repeatable)')
    parser.add_argument("--cwd", help="folder to work in (default: here)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds before the agent is killed (default 900)")
    parser.add_argument("--json", action="store_true", help="report as JSON")
    parser.add_argument("--dry-run", action="store_true", help="print the command, environment and generated files; start nothing")
    args = parser.parse_args(argv)
    task = sys.stdin.read() if args.task == "-" else args.task
    if not task.strip():
        parser.error("the task is empty")
    from .cli import configured
    from .onboard import load_config
    config = load_config()
    if args.dry_run:
        dry_run_task(config, task, args.agent, args.allow, args.cwd)
        return 0
    if not config.get("remote"):
        from .onboard import setup_complete
        if not setup_complete(config):
            sys.exit("Hearthwork is not set up yet. Run `hearthwork` once in a terminal.")
        config = configured(interactive=False)
    report = run_task(config, task, args.agent, args.allow, args.cwd, args.timeout)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # an agent's reply can hold characters the console code page lacks
    print(json.dumps(report, indent=2) if args.json else format_report(report), flush=True)
    return exit_code(report)
