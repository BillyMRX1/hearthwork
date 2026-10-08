"""Coding agents ("harnesses") that can use the local model, and the relay that makes their requests fit any model.

Each harness is one entry in HARNESSES: how to find it, how to launch it against the local server, and how to run it
headless. All launch through a relay (a small HTTP server in this process) that cleans up requests on their way to llama.cpp:
strict chat templates (e.g. Qwen3.5/3.8-style) reject a second system/developer message, unknown roles, or two
user messages in a row, all of which these agents send. To add another harness, add an entry to HARNESSES (see there).
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
from pathlib import Path

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


def normalize_chat(body):
    """OpenAI Chat Completions (SDKs, Aider, ...). Leading system/developer messages join into one system message;
    later ones move into the neighbouring user message as a reminder, so strict chat templates accept them."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(isinstance(m, dict) and m.get("role") in ("system", "developer")
                                                   for m in messages):
        return body

    def text_of(content):
        if isinstance(content, str):
            return content
        return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")

    def add(message, text, front=False):
        content = message.get("content")
        if isinstance(content, str) or content is None:
            joined = [text, content or ""] if front else [content or "", text]
            return dict(message, content="\n\n".join(t for t in joined if t))
        part = {"type": "text", "text": text}
        return dict(message, content=[part, *content] if front else [*content, part])

    system, out, pending = [], [], []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("system", "developer"):
            text = text_of(message.get("content"))
            if not out:
                system.append(text)
            elif out[-1].get("role") == "user":
                out[-1] = add(out[-1], _reminder(text))
            else:
                pending.append(_reminder(text))
            continue
        if pending and role == "user":
            message = add(message, "\n\n".join(pending), front=True)
            pending = []
        out.append(message)
    if pending:
        out.append({"role": "user", "content": "\n\n".join(pending)})
    if system:
        out.insert(0, {"role": "system", "content": "\n\n".join(t for t in system if t)})
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


def make_relay(upstream_port, address=("127.0.0.1", 0), intercept=None, extra_headers=None):
    """HTTP relay to llama-server on `upstream_port`, bound to `address` (not yet serving). `intercept(handler)`, when
    given, runs first for every request and returns True once it has answered it itself: `hearthwork share` uses it
    for the device check and the pairing endpoints, and everything else goes through the same normalization.
    `extra_headers(handler)` returns [(name, value)] added to every response (CORS in `hearthwork api`)."""

    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the body ends when the connection closes: simple, works for streams

        def _send_json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self._extra()
            self.end_headers()
            self.wfile.write(data)

        def _extra(self):
            for name, value in (extra_headers(self) if extra_headers else ()):
                self.send_header(name, value)

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
                    elif path.endswith("/chat/completions"):
                        payload = normalize_chat(payload)
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
                self._extra()
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (ConnectionError, OSError):
                pass  # the client went away (e.g. Esc in the agent)
            finally:
                upstream.close()

        do_GET = do_POST = do_OPTIONS = _relay

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


_remote = None  # config["remote"] while prepare()/preview() build a command for a shared model (read by claude_command)


def host_parts(remote):
    """(host name, route like "LAN"/"Tailscale" or "") of a shared model, for the status line."""
    if not remote:
        return "", ""
    return str(remote.get("hostName") or remote.get("host") or ""), str((remote.get("via") or "").split(" ")[0])


def statusline_command(name, context, host="", via=""):
    """Shell command for Claude Code's status line. Claude runs it through Git Bash, PowerShell or sh depending on
    the machine, so it must parse the same in all three: forward slashes, and no quotes around the program (a
    PowerShell command can't start with a quoted path). If the Python path needs quotes, use the `hearthwork`
    launcher on PATH instead."""
    exe = sys.executable.replace("\\", "/")
    base = f"{exe} -m hearthwork statusline"
    if any(c in exe for c in " ()&'"):
        launcher = shutil.which("hearthwork")
        base = f"{launcher.replace(chr(92), '/')} statusline" if launcher and " " not in launcher else f'"{exe}" -m hearthwork statusline'
    extra = f' "{host}"' + (f' "{via}"' if via else "") if host else ""
    return f'{base} "{name}" {context_label(context)}{extra}'


def statusline_text(args):
    """The status line for `hearthwork statusline <model> <context> [host [route]]`: "host · model · context", with
    "local" when the model runs on this computer and "host (LAN)" when a route is given."""
    args = list(args)
    where = "local"
    if len(args) > 2 and args[2]:
        where = args[2] + (f" ({args[3]})" if len(args) > 3 and args[3] else "")
    return " · ".join([where, *args[:2]])


def deep_merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def claude_settings(name, context, host="", via=""):
    """The settings passed with `claude --settings`; they layer on top of yours, which stay untouched. Returns
    (settings, auto): auto says the user's overrides ask for auto mode."""
    settings = {
        # "default" is Claude Code's ask-before-risky-actions mode ("manual" in its UI). Newer versions start in
        # auto, where the (slow, local) model reviews every command first and those checks time out.
        "permissions": {"defaultMode": "default"},
        "statusLine": {"type": "command", "command": statusline_command(name, context, host, via)},
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

_preview = None  # a list while `--dry-run` previews a launch: generated files are collected instead of written


def write_file(path, text):
    """Write a file Hearthwork generates for a session (in its data folder, never in the agent's own config). The new
    file replaces the old one in one step, so two launches never read half a file. During a preview nothing is written."""
    path = Path(path)
    if _preview is not None:
        _preview.append((str(path), text))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def safe_name(name):
    return re.sub(r"[^\w.-]+", "_", name)


def claude_command(binary, base_url, token, name, context, max_output, args):
    # The connection stays in process env vars, not the settings file: the relay port changes every session, and
    # env vars are what already works. The settings file only holds what is the same for every session.
    host, via = host_parts(_remote)
    settings, auto = claude_settings(name, context, host, via)
    # One file per model (and host): two sessions on different models or hosts must not share a status line.
    settings_file = write_file(HOME / f"claude-settings-{name}{'@' + safe_name(host) if host else ''}.json", json.dumps(settings, indent=2))
    if auto and _preview is None:
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
    instructions = "(Codex's own system prompt, read from `codex debug models --bundled` at launch)" if _preview is not None else ""
    try:
        if _preview is not None:
            raise OSError  # a preview starts nothing, not even `codex debug models`
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
    return write_file(paths.HOME / "codex-models.json", json.dumps({"models": [entry]}, indent=1))


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


def opencode_command(binary, base_url, token, name, context, max_output, args):
    # Docs: https://opencode.ai/docs/providers/ (custom @ai-sdk/openai-compatible provider with limit.context/output)
    # and https://opencode.ai/docs/config/ (OPENCODE_CONFIG_CONTENT: inline config with the highest priority of the
    # normal sources, so no file is written and ~/.config/opencode/opencode.json stays as it is).
    config = {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
        "provider": {"hearthwork": {
            "npm": "@ai-sdk/openai-compatible", "name": "Hearthwork (local model)",
            "options": {"baseURL": f"{base_url}/v1", "apiKey": "{env:HEARTHWORK_KEY}"},
            "models": {name: {"name": name, "limit": {"context": context, "output": max_output}}}}},
    }
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(config), "HEARTHWORK_KEY": token or "local-model"}
    model = ["-m", f"hearthwork/{name}"]
    return ([binary, "run", *model, *args[1:]] if args[:1] == ["run"] else [binary, *model, *args]), env


def aider_command(binary, base_url, token, name, context, max_output, args):
    # Docs: https://aider.chat/docs/llms/openai-compat.html (OPENAI_API_BASE / OPENAI_API_KEY, model openai/<name>),
    # https://aider.chat/docs/config/adv-model-settings.html (metadata file: context; settings file: extra_params).
    # --no-analytics only turns it off for this session (--analytics-disable would write into ~/.aider).
    model = f"openai/{name}"
    stem = safe_name(name)
    metadata = write_file(HOME / f"aider-metadata-{stem}.json", json.dumps({model: {
        "max_input_tokens": context, "max_tokens": max_output, "max_output_tokens": max_output,
        "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}}, indent=1))
    settings = write_file(HOME / f"aider-settings-{stem}.json", json.dumps(  # JSON is YAML: aider reads it as it is
        [{"name": model, "extra_params": {"max_tokens": max_output}}], indent=1))
    env = {**os.environ, "OPENAI_API_BASE": f"{base_url}/v1", "OPENAI_API_KEY": token or "local-model",
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # aider's console crashes on Windows code pages otherwise
    flags = ["--model", model, "--model-metadata-file", str(metadata), "--model-settings-file", str(settings),
             "--no-show-model-warnings", "--no-check-update", "--no-analytics", "--no-show-release-notes"]
    return [binary, *flags, *args], env


def qwen_command(binary, base_url, token, name, context, max_output, args):
    # Docs: https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/ (OPENAI_BASE_URL / OPENAI_API_KEY /
    # OPENAI_MODEL; model.generationConfig; QWEN_CODE_SYSTEM_SETTINGS_PATH: an extra settings file that layers over
    # ~/.qwen/settings.json, which stays untouched).
    settings = {"security": {"auth": {"selectedType": "openai"}},
                "model": {"name": name, "generationConfig": {"contextWindowSize": context,
                                                             "samplingParams": {"max_tokens": max_output}}},
                "general": {"disableAutoUpdate": True}, "privacy": {"usageStatisticsEnabled": False}}
    path = write_file(HOME / f"qwen-settings-{safe_name(name)}.json", json.dumps(settings, indent=2))
    env = {**os.environ, "OPENAI_BASE_URL": f"{base_url}/v1", "OPENAI_API_KEY": token or "local-model",
           "OPENAI_MODEL": name, "QWEN_CODE_SYSTEM_SETTINGS_PATH": str(path)}
    return [binary, "--model", name, *args], env


# ---------- headless runs (hearthwork task, bench) ----------
# Each returns (arguments after the agent's name, extra env). `allow`: command patterns the agent may run; `prompt`
# is the file holding the task for agents that cannot read it from stdin; `resume` continues the previous session
# (True, or its id when the entry has a `session` reader); `cwd` is the folder it works in.

CLAUDE_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "MultiEdit"]


def claude_task(allow, last_message, prompt, resume=False, cwd=None):
    tools = CLAUDE_TOOLS + [f"Bash({pattern})" for pattern in allow]
    return ["-p", "--output-format", "json", "--permission-mode", "acceptEdits", "--allowedTools", *tools], {}


def codex_task(allow, last_message, prompt, resume=False, cwd=None):
    return ["exec", "--skip-git-repo-check", "-c", 'sandbox_mode="workspace-write"', "-o", str(last_message), "-"], {}


def opencode_task(allow, last_message, prompt, resume=False, cwd=None):
    # Everything is denied except editing and the allowed commands (the last matching rule wins).
    permission = {"edit": "allow", "webfetch": "deny", "bash": {"*": "deny", **{pattern: "allow" for pattern in allow}}}
    # `--continue` takes the newest session of any folder, so a later turn names its session (see session_opencode)
    again = ["--session", resume] if isinstance(resume, str) else ["--continue"] if resume else []
    # --dir: a resumed session otherwise works in the folder it was first started from, not in this one
    where = ["--dir", str(cwd)] if cwd else []
    return ["run", "--format", "json", *again, *where], {"OPENCODE_PERMISSION": json.dumps(permission)}


def aider_task(allow, last_message, prompt, resume=False, cwd=None):
    # Aider edits files; the shell commands it suggests can't be limited to patterns, so they are switched off.
    return ["--message-file", str(prompt), "--yes-always", "--no-suggest-shell-commands", "--no-pretty", "--no-stream", "--no-fancy-input",
            *(["--restore-chat-history"] if resume else [])], {}


def qwen_task(allow, last_message, prompt, resume=False, cwd=None):
    tools = [f"run_shell_command({pattern.removesuffix(' *').removesuffix('*')})" for pattern in allow]
    return ["--approval-mode", "auto-edit", "--output-format", "json", *(["--allowed-tools", *tools] if tools else []),
            *(["--continue"] if resume else [])], {}


def text_of(value):
    """A final message as clean text: strings stripped, anything else (None, lists of blocks) made printable."""
    if value is None:
        return ""
    if isinstance(value, list):
        value = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in value)
    return str(value).replace(chr(0), "").strip()


def result_claude(stdout, stderr, returncode, last_message):
    """(final message, error text or None) from what Claude Code printed."""
    stdout, stderr = stdout or "", stderr or ""
    data = next((e for e in reversed(json_events(stdout)) if e.get("type") == "result" or "result" in e), None)
    if data is None:
        return "", f"Claude Code exited with {returncode}: {(stdout + stderr)[-400:].strip()}"
    message = text_of(data.get("result"))
    if data.get("is_error"):
        return message, f"Claude Code reported an error: {message or str(data)[:400]}"
    return message, (f"Claude Code exited with {returncode}" if returncode else None)


def result_codex(stdout, stderr, returncode, last_message):
    stdout, stderr = stdout or "", stderr or ""
    try:
        with open(last_message, encoding="utf-8", errors="replace") as f:
            message = text_of(f.read())
    except (OSError, TypeError):
        message = ""
    if returncode != 0:
        return message, f"Codex exited with {returncode}: {(stdout + stderr)[-400:].strip()}"
    return message, None if message else "Codex finished without a final message"


def json_events(text):
    """The JSON objects among the lines of `text` (other lines are skipped)."""
    events = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def result_opencode(stdout, stderr, returncode, last_message):
    """`opencode run --format json` prints one event per line; the final message is the text after the last tool call."""
    stdout, stderr = stdout or "", stderr or ""
    texts, errors = [], []
    for event in json_events(stdout):
        if event.get("type") == "tool_use":
            texts = []
        elif event.get("type") == "text":
            texts.append(text_of((event.get("part") or {}).get("text")))
        elif event.get("type") == "error":
            errors.append(str(event.get("error", {}).get("data", {}).get("message") or event.get("error"))[:300])
    message = text_of("".join(texts))
    if errors or returncode:
        return message, f"OpenCode failed (exit {returncode}): {'; '.join(errors) or (stdout + stderr)[-400:].strip()}"
    return message, None if message else "OpenCode finished without a final message"


def session_opencode(stdout):
    """The id of the session an `opencode run --format json` run used, or None."""
    return next((e["sessionID"] for e in json_events(stdout) if e.get("sessionID")), None)


def result_aider(stdout, stderr, returncode, last_message):
    stdout, stderr = stdout or "", stderr or ""
    message = text_of(stdout)[-4000:]  # aider prints the model's reply and what it changed, no tool calls to filter out
    if returncode != 0:
        return message, f"Aider exited with {returncode}: {(stdout + stderr)[-400:].strip()}"
    return message, None if message else "Aider finished without output"


def result_qwen(stdout, stderr, returncode, last_message):
    """`qwen --output-format json` prints a JSON array of messages; its last item is the result."""
    stdout, stderr = stdout or "", stderr or ""
    start = stdout.find("[{")
    try:
        items = json.loads(stdout[start:]) if start >= 0 else []
    except ValueError:
        items = []
    final = next((i for i in reversed(items) if isinstance(i, dict) and i.get("type") == "result"), None)
    if not final:
        return "", f"Qwen Code exited with {returncode}: {(stdout + stderr)[-400:].strip()}"
    message = text_of(final.get("result"))
    if final.get("is_error"):
        return message, f"Qwen Code reported an error: {message or str(final)[:400]}"
    return message, (f"Qwen Code exited with {returncode}" if returncode else None)


# Aider only edits files (no tools): told it can run things, a local model answers in a format Aider rejects.
AIDER_SUFFIX = ("\n\nYou can only create and edit files in this folder; you cannot run commands, so do not try to or "
                "write example output. Answer with the file contents in the required format, then one short sentence.")

# One entry per agent; adding an agent is adding an entry (and a command + task + result function above).
#   title/binary/version   display name, program name, arguments that print its version
#   install                official install command and URL, shown when it is missing
#   api                    what it speaks: "anthropic" (/v1/messages), "openai-responses" (/v1/responses) or
#                          "openai-chat" (/v1/chat/completions); the relay cleans up all three
#   connects               how Hearthwork points it at the model, for `hearthwork agents`
#   command                (binary, base_url, token, model, context, max_output, args) -> (command, env). base_url is the
#                          relay without /v1; token is the device key of a shared model (None: local); the context and
#                          max_output come from the running server. Never edits the agent's own config.
#   task/prompt/result     headless run for `hearthwork task`/MCP/bench: arguments, how the task goes in ("stdin" or a
#                          "file"), and how to read the final message from the output
#   session                (optional) reads the session id from a headless run's output, so bench can continue it
#   suffix                 (optional) closing instruction for `hearthwork task`, instead of the default one
#   headless               example flags shown when an agent is started without a terminal
HARNESSES = {
    "claude": {"title": "Claude Code", "binary": "claude", "version": ["--version"], "api": "anthropic",
               "install": "curl -fsSL https://claude.ai/install.sh | bash  (https://code.claude.com)",
               "connects": "ANTHROPIC_* env vars + --settings file", "command": claude_command,
               "task": claude_task, "prompt": "stdin", "result": result_claude, "headless": '-p "prompt"'},
    "codex": {"title": "Codex", "binary": "codex", "version": ["--version"], "api": "openai-responses",
              "install": "npm install -g @openai/codex  (https://developers.openai.com/codex)",
              "connects": "-c overrides + generated model catalogue", "command": codex_command,
              "task": codex_task, "prompt": "stdin", "result": result_codex, "headless": 'exec "prompt"'},
    "opencode": {"title": "OpenCode", "binary": "opencode", "version": ["--version"], "api": "openai-chat",
                 "install": "npm i -g opencode-ai  (https://opencode.ai)",
                 "connects": "OPENCODE_CONFIG_CONTENT env (provider + limits)", "command": opencode_command,
                 "task": opencode_task, "prompt": "stdin", "result": result_opencode, "session": session_opencode, "headless": 'run "prompt"'},
    "aider": {"title": "Aider", "binary": "aider", "version": ["--version"], "api": "openai-chat",
              "install": "uv tool install --python 3.12 aider-chat  (https://aider.chat/docs/install.html)",
              "connects": "OPENAI_API_* env + generated model metadata/settings files", "command": aider_command,
              "task": aider_task, "prompt": "file", "suffix": AIDER_SUFFIX, "result": result_aider, "headless": '--message "prompt" --yes-always'},
    "qwen": {"title": "Qwen Code", "binary": "qwen", "version": ["--version"], "api": "openai-chat",
             "install": "npm i -g @qwen-code/qwen-code  (https://github.com/QwenLM/qwen-code)",
             "connects": "OPENAI_* env + generated settings file (QWEN_CODE_SYSTEM_SETTINGS_PATH)", "command": qwen_command,
             "task": qwen_task, "prompt": "stdin", "result": result_qwen, "headless": '"prompt"'},
}
API_NAMES = {"anthropic": "Anthropic Messages", "openai-responses": "OpenAI Responses", "openai-chat": "OpenAI Chat Completions"}


def installed(key):
    return shutil.which(HARNESSES[key]["binary"])


def prepare(key, port, name, context, max_output=4096, args=(), extra_env=None, remote=None, clean=False):
    """(command, env) to run harness `key` against the server on `port` (starts the local relay), or None when the
    agent is not installed. With `remote` (config["remote"]: a model shared by another computer) the agent talks to that
    computer directly: no local relay (the host normalizes). With `clean`, the environment of a parent Claude Code /
    Codex session is dropped (see clean_env)."""
    harness = HARNESSES[key]
    binary = installed(key)
    if not binary:
        print(f"{harness['title']} is not installed. Install: {harness['install']}", file=sys.stderr if clean else sys.stdout)
        return None
    if remote:
        base_url, token = f"http://{remote['host']}:{remote['port']}", remote["key"]
    else:
        base_url, token = f"http://127.0.0.1:{start_relay(port)}", None
    global _remote
    _remote = remote
    try:
        command, env = harness["command"](binary, base_url, token, name, context, max_output, list(args))
    finally:
        _remote = None
    if extra_env:
        env = {**(os.environ if env is None else env), **extra_env}
    if clean:
        env = clean_env(os.environ if env is None else env)
    return command, env


# ---------- dry run ----------

SECRET = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$", re.I)  # not ..._TOKENS (a count)


def shown(arg):
    return arg if re.fullmatch(r"[\w@%+=:,./\\~-]+", arg) else json.dumps(arg)


def preview(key, name, context, max_output=4096, args=(), extra_env=None, remote=None, note=None):
    """Print what launching `key` would run, without starting the model, the relay or the agent: the command, the
    environment variables Hearthwork sets (secrets masked), and the files it generates (path and content). The relay
    port is a placeholder; nothing is written."""
    global _preview, _remote
    harness = HARNESSES[key]
    base_url, token = ((f"http://{remote['host']}:{remote['port']}", remote["key"]) if remote
                       else ("http://127.0.0.1:<relay-port>", None))
    binary = installed(key) or harness["binary"]
    _preview, _remote = [], remote
    try:
        command, env = harness["command"](binary, base_url, token, name, context, max_output, list(args))
        files = _preview
    finally:
        _preview, _remote = None, None
    changed = {k: v for k, v in {**(os.environ if env is None else env), **(extra_env or {})}.items() if os.environ.get(k) != v}
    removed = [k for k in os.environ if env is not None and k not in env]
    print(f"{harness['title']}: {'installed' if installed(key) else 'NOT installed (' + harness['install'] + ')'}   "
          f"API: {API_NAMES[harness['api']]}   model: {name}   context: {context}   max output: {max_output}")
    print("\ncommand:\n  " + " ".join(shown(a) for a in command))
    print("\nenvironment Hearthwork sets (secrets masked):")
    for k, v in sorted(changed.items()):
        if SECRET.search(k):
            v = "********"
        try:  # a JSON value (OpenCode's config) is easier to read indented
            if v.startswith("{"):
                v = "\n" + "\n".join("    " + line for line in json.dumps(json.loads(v), indent=2).splitlines())
        except ValueError:
            pass
        print(f"  {k}={v}")
    for k in removed:
        print(f"  {k} (removed)")
    if not changed and not removed:
        print("  (none)")
    print("\ngenerated files (written to Hearthwork's data folder at launch; your own agent config is never edited):")
    for path, text in files or []:
        print(f"  {path}\n" + "\n".join("    " + line for line in text.splitlines()))
    if not files:
        print("  (none)")
    print("\nHearthwork starts a relay on a free local port " + ("(not used: the model is on another computer)" if remote else
          "in front of the model server; <relay-port> stands for it") + ".")
    if note:
        print(note)


def agent_version(key, timeout=20):
    """Version text of an installed agent, or None."""
    binary = installed(key)
    if not binary:
        return None
    try:
        done = subprocess.run([binary, *HARNESSES[key]["version"]], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return "?"
    lines = [line.strip() for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    found = re.search(r"\d+\.\d+[\w.+-]*", lines[0]) if lines else None
    return found.group(0) if found else (lines[0][:30] if lines else "?")


def agents_table():
    """The `hearthwork agents` text: every supported agent, whether it is installed, and how it is connected."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(HARNESSES)) as pool:
        versions = dict(zip(HARNESSES, pool.map(agent_version, HARNESSES)))
    rows = [("agent", "installed", "API", "how Hearthwork connects it")]
    rows += [(f"{key}", (versions[key] or "-") if installed(key) else "no", API_NAMES[h["api"]], h["connects"])
             for key, h in HARNESSES.items()]
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[3] for r in rows]
    missing = [(h["title"], h["install"]) for key, h in HARNESSES.items() if not installed(key)]
    if missing:
        lines += ["", "Not installed:"] + [f"  {title}: {hint}" for title, hint in missing]
    lines += ["", "Start one with `hearthwork <agent>`; `hearthwork <agent> --dry-run` shows what would run, without starting anything."]
    return "\n".join(lines)


# Set by a running Claude Code / Codex for its own session; a child started inside one must not inherit them (it
# would think it is nested, or reuse the parent's session, socket or sandbox). Hearthwork's own are kept.
KEEP_ENV = {"CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_MAX_CONTEXT_TOKENS", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"}
DROP_ENV = ("CLAUDECODE", "AI_AGENT", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_", "CODEX_SANDBOX", "CODEX_THREAD_ID",
            "CODEX_CI", "CODEX_SESSION", "CODEX_INTERNAL")


def clean_env(env):
    """Copy of `env` without the markers of a parent Claude Code / Codex session. Config locations
    (CLAUDE_CONFIG_DIR, CODEX_HOME) stay."""
    return {k: v for k, v in env.items()
            if k in KEEP_ENV or not (k == "CLAUDECODE" or k.startswith(DROP_ENV))}


def launch(key, port, name, context, max_output=4096, args=(), capture=False, cwd=None, timeout=None, extra_env=None,
           remote=None, input_text=None):
    """Run harness `key` against the server on `port`: in this terminal, or with `capture` its output is
    returned as a CompletedProcess (for the benchmark). `extra_env` is added to the agent's environment. Returns the
    exit code otherwise. `input_text` goes to the agent's stdin (with `capture`). With `remote` the agent talks to the shared model directly, with no local prompt-cache saves."""
    prepared = prepare(key, port, name, context, max_output, args, extra_env, remote)
    if not prepared:
        return 1
    command, env = prepared
    try:
        if capture:
            return subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", input=input_text, stdin=None if input_text else subprocess.DEVNULL,
                                  timeout=timeout)
        return subprocess.call(command, env=env, cwd=cwd)
    except KeyboardInterrupt:
        return 130
    finally:
        if not capture and not remote:
            from .server import save_slots  # the session's prompt cache makes the next start's first message quick
            save_slots(port)
