"""Coding agents ("harnesses") that can use the local model, and the relay that makes their requests fit any model.

Each harness is one entry in HARNESSES: how to find it and how to launch it against the local server. Both
launch through a relay (a small HTTP server in this process) that cleans up requests on their way to llama.cpp:
strict chat templates (e.g. Qwen3.5/3.8-style) reject a second system/developer message, unknown roles, or two
user messages in a row, all of which these agents send. To add another harness, add a command function to HARNESSES.
"""
import http.client
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from . import paths
from .paths import HOME


# ---------- request normalization ----------

CODEX_MAX_OUTPUT = 8192  # tokens per reply; ends a runaway generation instead of letting it run forever

def _reminder(text):
    return f"<system-reminder>\n{text}\n</system-reminder>"


BILLING_HEADER = "x-anthropic-billing-header:"  # Claude Code's first system block; its `cc_version=<ver>.<hash>` hash changes with the session's first message
TOKENS_LEFT = re.compile(r"\s*<total_tokens>[^<]*</total_tokens>\s*")  # "N tokens left", appended after every user turn


def _strip_volatile(text):
    """Remove the parts of Claude Code's text that carry no meaning for a local model but differ between sessions
    or turns: the billing-header line and the token counter."""
    kept = [line for line in text.split("\n") if not line.startswith(BILLING_HEADER)]
    return TOKENS_LEFT.sub("", "\n".join(kept)).strip()


def normalize_anthropic(body):
    """Anthropic Messages (Claude Code). Claude Code puts a `system` message *inside* the conversation (its
    environment info, after the user's first message); its text moves into the neighbouring user message, at
    the same position, so roles still alternate and the server's prompt cache still matches turn to turn.

    Also drops what makes the prompt differ for no reason, since llama.cpp reuses only the part before the first
    difference. Measured on Claude Code 2.1.292 (captured requests, 17K-token prompt): the billing header is the
    very first system block and its hash differs per session, so every new session reprocessed the whole prompt;
    the per-turn `<total_tokens>` messages stay unchanged once sent (they cost no cache within a session) but are
    noise for a local model (it shows 15,000,000 tokens left), so they are dropped too."""
    system = body.get("system")
    if isinstance(system, str):
        body = dict(body, system=_strip_volatile(system))
    elif isinstance(system, list):
        cleaned = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = _strip_volatile(block.get("text", ""))
                if text:
                    cleaned.append(dict(block, text=text))
            else:
                cleaned.append(block)
        body = dict(body, system=cleaned)
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(m.get("role") == "system" for m in messages):
        return body

    def blocks(content):
        return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])

    out, pending = [], []  # pending: system blocks waiting for the next user message
    for message in messages:
        if message.get("role") == "system":
            text = _strip_volatile("\n".join(b.get("text", "") for b in blocks(message.get("content")) if b.get("type") == "text"))
            if not text:
                continue
            wrapped = {"type": "text", "text": _reminder(text)}
            if out and out[-1]["role"] == "user":
                out[-1] = dict(out[-1], content=blocks(out[-1]["content"]) + [wrapped])
            else:
                pending.append(wrapped)
            continue
        if pending and message.get("role") == "user":
            message = dict(message, content=pending + blocks(message.get("content")))
            pending = []
        out.append(message)
    if pending:
        out.append({"role": "user", "content": pending})
    return dict(body, messages=out)


def normalize_responses(body):
    """OpenAI Responses (Codex). Codex sends `instructions` plus a `developer` message, and two user messages
    in a row. Leading system/developer messages join `instructions` (one system prompt); later ones become
    user text; consecutive user messages merge into one."""
    # Codex sets no reply limit, and a local model can get stuck generating forever (seen: 20+ minutes on one
    # turn). Claude Code caps its replies itself (CLAUDE_CODE_MAX_OUTPUT_TOKENS).
    if not body.get("max_output_tokens"):
        body = dict(body, max_output_tokens=CODEX_MAX_OUTPUT)
    items = body.get("input")
    if not isinstance(items, list):
        return body

    def parts(content):
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        return [dict(p, type="input_text") if p.get("type") in ("text", "output_text") else p for p in content or []]

    def text_of(content):
        return "\n".join(p.get("text", "") for p in parts(content) if p.get("type") == "input_text")

    instructions = [body["instructions"]] if body.get("instructions") else []
    out = []
    for item in items:
        role = item.get("role")
        is_message = role and item.get("type", "message") == "message"
        if is_message and role in ("system", "developer"):
            if not out:
                instructions.append(text_of(item.get("content")))
                continue
            item, role = {"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": _reminder(text_of(item.get("content")))}]}, "user"
        if is_message and role == "user" and out and out[-1].get("role") == "user" and out[-1].get("type", "message") == "message":
            out[-1] = dict(out[-1], content=parts(out[-1].get("content")) + parts(item.get("content")))
            continue
        out.append(item)
    return dict(body, input=out, instructions="\n\n".join(i for i in instructions if i) or None)


# ---------- relay ----------

_dump_count = 0

def dump_request(incoming, normalized):
    """HEARTHWORK_RELAY_DUMP=<dir>: write each request as it arrived and as sent on (NNN-in.json, NNN-out.json),
    to see what changes between turns."""
    global _dump_count
    folder = os.environ.get("HEARTHWORK_RELAY_DUMP")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    _dump_count += 1
    stamp = f"{int(time.time() * 1000)}-{_dump_count:03d}"
    for kind, data in (("in", incoming), ("out", normalized)):
        with open(os.path.join(folder, f"{stamp}-{kind}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)


def make_relay(upstream_port, address=("127.0.0.1", 0), intercept=None):
    """HTTP relay to llama-server on `upstream_port`, bound to `address` (not yet serving). `intercept(handler)`, when
    given, runs first for every request and returns True once it has answered it itself: `hearthwork share` uses it
    for the device check and the pairing endpoints, and everything else goes through the same normalization."""

    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the body ends when the connection closes: simple, works for streams

        def _send_json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_body(self, limit=None):
            length = int(self.headers.get("content-length") or 0)
            return self.rfile.read(min(length, limit) if limit else length)

        def _relay(self):
            if intercept and intercept(self):
                return
            body = self.rfile.read(int(self.headers.get("content-length") or 0)) or None
            path = self.path.split("?")[0]
            if self.command == "GET" and path.rstrip("/").endswith("/models"):
                # llama.cpp's list also carries an Ollama-style "models" array that Codex tries (and fails) to
                # read as its own model catalogue; an empty catalogue keeps Codex quiet, "data" serves the rest.
                try:
                    with urllib_open(upstream_port, "/v1/models") as response:
                        data = json.load(response).get("data", [])
                    return self._send_json(200, {"object": "list", "data": data, "models": []})
                except Exception as error:
                    return self._send_json(502, {"error": {"message": str(error)}})
            if body and self.command == "POST":
                try:
                    payload = json.loads(body)
                    raw = payload
                    if path.endswith("/messages"):
                        payload = normalize_anthropic(payload)
                    elif path.endswith("/responses"):
                        payload = normalize_responses(payload)
                    body = json.dumps(payload).encode()
                    dump_request(raw, payload)
                except ValueError:
                    pass
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
            if body:
                headers["content-length"] = str(len(body))
            upstream = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=3600)
            try:
                upstream.request(self.command, self.path, body=body, headers=headers)
                response = upstream.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (ConnectionError, OSError):
                pass  # the client went away (e.g. Esc in the agent)
            finally:
                upstream.close()

        do_GET = do_POST = _relay

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(address, Relay)
    server.daemon_threads = True
    return server


def start_relay(upstream_port):
    """Local HTTP relay to llama-server on `upstream_port`. Returns the relay's port."""
    server = make_relay(upstream_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def urllib_open(port, path):
    import urllib.request
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)


# ---------- Claude Code settings ----------

USER_OVERRIDES = os.path.join("~", ".claude", "settings-hearthwork.json")  # optional, yours


def overrides_path():
    # HEARTHWORK_CLAUDE_OVERRIDES points elsewhere (used by tests)
    return os.path.expanduser(os.environ.get("HEARTHWORK_CLAUDE_OVERRIDES") or USER_OVERRIDES)


def context_label(tokens):
    return f"{round(tokens / 1024)}K"


def statusline_command(name, context):
    """Shell command for Claude Code's status line. Claude runs it through Git Bash, PowerShell or sh depending on
    the machine, so it must parse the same in all three: forward slashes, and no quotes around the program (a
    PowerShell command can't start with a quoted path). If the Python path needs quotes, use the `hearthwork`
    launcher on PATH instead."""
    exe = sys.executable.replace("\\", "/")
    base = f"{exe} -m hearthwork statusline"
    if any(c in exe for c in " ()&'"):
        launcher = shutil.which("hearthwork")
        base = f"{launcher.replace(chr(92), '/')} statusline" if launcher and " " not in launcher else f'"{exe}" -m hearthwork statusline'
    return f'{base} "{name}" {context_label(context)}'


def deep_merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def claude_settings(name, context):
    """The settings passed with `claude --settings`; they layer on top of yours, which stay untouched. Returns
    (settings, auto): auto says the user's overrides ask for auto mode."""
    settings = {
        # "default" is Claude Code's ask-before-risky-actions mode ("manual" in its UI). Newer versions start in
        # auto, where the (slow, local) model reviews every command first and those checks time out.
        "permissions": {"defaultMode": "default"},
        "statusLine": {"type": "command", "command": statusline_command(name, context)},
    }
    path = overrides_path()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                deep_merge(settings, json.load(f))
        except (OSError, ValueError, AttributeError) as error:
            print(f"\033[33mIgnoring {path}: {error}\033[0m")
    return settings, settings.get("permissions", {}).get("defaultMode") == "auto"


# ---------- harnesses ----------

def claude_command(binary, base_url, token, name, context, max_output, args):
    # The connection stays in process env vars, not the settings file: the relay port changes every session, and
    # env vars are what already works. The settings file only holds what is the same for every session.
    settings, auto = claude_settings(name, context)
    # One file per model: two sessions on different models must not share (and overwrite) a status line.
    settings_file = HOME / f"claude-settings-{name}.json"
    settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if auto:
        print("\033[2mYour settings-hearthwork.json starts Claude Code in auto mode: every command is first checked "
              "by the local model, which can be slow or time out and block it.\033[0m")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.update({
        "ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": token or "local-model",
        "ANTHROPIC_MODEL": name, "ANTHROPIC_DEFAULT_OPUS_MODEL": name, "ANTHROPIC_DEFAULT_SONNET_MODEL": name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": name, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context), "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    return [binary, "--settings", str(settings_file), "--model", name, *args], env


def codex_catalog(binary, name, context):
    """Write the model catalogue entry that tells Codex about the local model, and return its path.

    Without an entry Codex warns "Model metadata for <model> not found" and uses fallback metadata. The format
    is Codex's own ModelInfo (see `codex debug models --bundled`); it is the same for every model, so only the
    name and context window are filled in. Written per launch so a changed model or context is always current."""
    # Required field: Codex's system prompt. Reuse one Codex ships, so the model gets the normal Codex prompt.
    instructions = ""
    try:
        shipped = subprocess.run([binary, "debug", "models", "--bundled"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=20)
        for entry in json.loads(shipped.stdout).get("models", []):
            instructions = entry.get("base_instructions") or ""
            if instructions:
                break
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    entry = {
        "slug": name, "display_name": name, "description": "Local model served by Hearthwork",
        # Local models here do not take a reasoning-effort setting; "none" keeps Codex from sending one.
        "default_reasoning_level": "none",
        "supported_reasoning_levels": [{"effort": "none", "description": "No separate reasoning step"}],
        "shell_type": "shell_command", "visibility": "hide", "supported_in_api": True, "priority": 99,
        "availability_nux": None, "upgrade": None, "support_verbosity": False, "default_verbosity": None,
        "apply_patch_tool_type": None, "truncation_policy": {"mode": "tokens", "limit": 10000},
        "context_window": context, "max_context_window": context, "experimental_supported_tools": [],
        "supports_parallel_tool_calls": True,
        "base_instructions": instructions or "You are Codex, a coding agent running in the user's terminal.",
    }
    path = paths.HOME / "codex-models.json"
    temporary = path.with_name(f"{path.name}.{os.getpid()}")  # replace, so two launches never read half a file
    temporary.write_text(json.dumps({"models": [entry]}, indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return path


def codex_command(binary, base_url, token, name, context, max_output, args):
    # A one-off model provider given with -c overrides: your ~/.codex/config.toml is not changed. With a shared
    # model (token is not None) Codex reads the device key from the HEARTHWORK_KEY variable and sends it as Bearer.
    key = ', env_key="HEARTHWORK_KEY"' if token else ""
    overrides = [
        "model_provider=llamacpp",
        f'model_providers.llamacpp={{name="llama.cpp (local)", base_url="{base_url}/v1", wire_api="responses"{key}}}',
        f"model_catalog_json={json.dumps(str(codex_catalog(binary, name, context)))}",  # a TOML string
        f"model_context_window={context}",
        f"model_auto_compact_token_limit={int(context * 0.8)}",
    ]
    # Codex drops every root-level -c when a subcommand (e.g. `exec`) gets its own -c, and would then send the
    # request to OpenAI instead of the local model. So -c settings in `args` are moved up next to ours.
    rest, i = [], 0
    while i < len(args):
        if args[i] in ("-c", "--config") and i + 1 < len(args):
            overrides.append(args[i + 1])
            i += 2
        elif args[i].startswith("--config="):
            overrides.append(args[i].split("=", 1)[1])
            i += 1
        else:
            rest.append(args[i])
            i += 1
    command = [binary]
    for override in overrides:
        command += ["-c", override]
    return [*command, "-m", name, *rest], ({**os.environ, "HEARTHWORK_KEY": token} if token else None)


HARNESSES = {
    "claude": {"title": "Claude Code", "binary": "claude", "command": claude_command,
               "install": "https://code.claude.com"},
    "codex": {"title": "Codex", "binary": "codex", "command": codex_command,
              "install": "https://developers.openai.com/codex (or: npm install -g @openai/codex)"},
}


def installed(key):
    return shutil.which(HARNESSES[key]["binary"])


def launch(key, port, name, context, max_output=4096, args=(), capture=False, cwd=None, timeout=None, extra_env=None,
           remote=None):
    """Run harness `key` against the server on `port`: in this terminal, or with `capture` its output is
    returned as a CompletedProcess (for the benchmark). `extra_env` is added to the agent's environment. Returns the
    exit code otherwise. With `remote` (config["remote"]: a model shared by another computer) the agent talks to that
    computer directly: no local relay (the host normalizes), no local prompt-cache saves."""
    harness = HARNESSES[key]
    binary = installed(key)
    if not binary:
        print(f"{harness['title']} is not installed. Get it from {harness['install']}")
        return 1
    if remote:
        base_url, token = f"http://{remote['host']}:{remote['port']}", remote["key"]
    else:
        base_url, token = f"http://127.0.0.1:{start_relay(port)}", None
    command, env = harness["command"](binary, base_url, token, name, context, max_output, list(args))
    if extra_env:
        env = {**(os.environ if env is None else env), **extra_env}
    try:
        if capture:
            return subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", stdin=subprocess.DEVNULL, timeout=timeout)
        return subprocess.call(command, env=env, cwd=cwd)
    except KeyboardInterrupt:
        return 130
    finally:
        if not capture and not remote:
            from .server import save_slots  # the session's prompt cache makes the next start's first message quick
            save_slots(port)
