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
import shutil
import subprocess
import threading


# ---------- request normalization ----------

CODEX_MAX_OUTPUT = 8192  # tokens per reply; ends a runaway generation instead of letting it run forever

def _reminder(text):
    return f"<system-reminder>\n{text}\n</system-reminder>"


def normalize_anthropic(body):
    """Anthropic Messages (Claude Code). Claude Code puts a `system` message *inside* the conversation (its
    environment info, after the user's first message); its text moves into the neighbouring user message, at
    the same position, so roles still alternate and the server's prompt cache still matches turn to turn."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(m.get("role") == "system" for m in messages):
        return body

    def blocks(content):
        return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])

    out, pending = [], []  # pending: system blocks waiting for the next user message
    for message in messages:
        if message.get("role") == "system":
            text = "\n".join(b.get("text", "") for b in blocks(message.get("content")) if b.get("type") == "text")
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

def start_relay(upstream_port):
    """Local HTTP relay to llama-server on `upstream_port`. Returns the relay's port."""

    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the body ends when the connection closes: simple, works for streams

        def _send_json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _relay(self):
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
                    if path.endswith("/messages"):
                        payload = normalize_anthropic(payload)
                    elif path.endswith("/responses"):
                        payload = normalize_responses(payload)
                    body = json.dumps(payload).encode()
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

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def urllib_open(port, path):
    import urllib.request
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)


# ---------- harnesses ----------

def claude_command(binary, relay, name, context, max_output, args):
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.update({
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{relay}", "ANTHROPIC_AUTH_TOKEN": "local-model",
        "ANTHROPIC_MODEL": name, "ANTHROPIC_DEFAULT_OPUS_MODEL": name, "ANTHROPIC_DEFAULT_SONNET_MODEL": name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": name, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context), "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    return [binary, "--model", name, *args], env


def codex_command(binary, relay, name, context, max_output, args):
    # A one-off model provider given with -c overrides: your ~/.codex/config.toml is not changed.
    overrides = [
        "model_provider=llamacpp",
        f'model_providers.llamacpp={{name="llama.cpp (local)", base_url="http://127.0.0.1:{relay}/v1", wire_api="responses"}}',
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
    return [*command, "-m", name, *rest], None


HARNESSES = {
    "claude": {"title": "Claude Code", "binary": "claude", "command": claude_command,
               "install": "https://code.claude.com"},
    "codex": {"title": "Codex", "binary": "codex", "command": codex_command,
              "install": "https://developers.openai.com/codex (or: npm install -g @openai/codex)"},
}


def installed(key):
    return shutil.which(HARNESSES[key]["binary"])


def launch(key, port, name, context, max_output=4096, args=(), capture=False, cwd=None, timeout=None):
    """Run harness `key` against the server on `port`: in this terminal, or with `capture` its output is
    returned as a CompletedProcess (for the benchmark). Returns the exit code otherwise."""
    harness = HARNESSES[key]
    binary = installed(key)
    if not binary:
        print(f"{harness['title']} is not installed. Get it from {harness['install']}")
        return 1
    relay = start_relay(port)
    command, env = harness["command"](binary, relay, name, context, max_output, list(args))
    if key == "claude" and not capture:
        print("\033[2mTip: in Claude Code's auto mode, every command is first checked by the local model, which can be "
              "slow or time out and block it. Shift+Tab switches to another permission mode.\033[0m")
    try:
        if capture:
            return subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", stdin=subprocess.DEVNULL, timeout=timeout)
        return subprocess.call(command, env=env, cwd=cwd)
    except KeyboardInterrupt:
        return 130
    finally:
        if not capture:
            from .server import save_slots  # the session's prompt cache makes the next start's first message quick
            save_slots(port)
