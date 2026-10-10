"""`hearthwork api`: one local OpenAI + Anthropic compatible endpoint for any tool (SDKs, Aider, OpenCode, Continue, curl).

It is the same relay the agent launchers use (harnesses.make_relay), so strict chat templates work: the request
cleanup is applied to /v1/messages and /v1/responses. Binds 127.0.0.1 only; `hearthwork share` is for other devices.
With --api-key every request needs it (Bearer or x-api-key). With --allow-switch a request whose "model" names another
local model file restarts the server with that model (it interrupts everyone else using the model).
"""
import argparse
import hmac
import io
import json
import os
import re
import threading
import time
from pathlib import Path

from . import context as ctx
from .harnesses import make_relay
from .onboard import CYAN, GREEN, RED, RESET, YELLOW, find_models, save_config
from .server import DIM, agent_context, port_open, served_model, start_background
from .share import request_key

DEFAULT_PORT = 8080
JSON_PATHS = ("/v1/",)  # everything the relay forwards; other paths (llama.cpp's admin endpoints) are not exposed


# ---------- pure logic (unit tested) ----------

def key_ok(expected, headers):
    """True when no key is required, or the client sent it (constant-time compare)."""
    if not expected:
        return True
    return hmac.compare_digest(request_key(headers).encode(), expected.encode())


LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$", re.IGNORECASE)
CORS_HEADERS = "authorization, x-api-key, content-type, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access"


def origin_allowed(origin, key):
    """Whether a browser page at `origin` may call the API. No Origin (curl, SDKs): yes. With an API key any origin
    may try (the key is the protection). Without one, only pages on localhost / 127.0.0.1, because any web page you
    visit could otherwise use your model."""
    if not origin or key:
        return True
    return bool(LOCAL_ORIGIN.match(origin))


def cors_headers(origin, key):
    """[(name, value)] to add to a response for a request from `origin` ([] when there is no Origin or it is refused)."""
    if not origin or not origin_allowed(origin, key):
        return []
    headers = [("Access-Control-Allow-Origin", "*" if key else origin), ("Access-Control-Expose-Headers", "*")]
    if not key:
        headers.append(("Vary", "Origin"))
    return headers


def preflight_headers(origin, key, requested_headers=None):
    return cors_headers(origin, key) + [
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", requested_headers or CORS_HEADERS), ("Access-Control-Max-Age", "600")]


def model_id(path):
    return Path(path).stem


def match_model(models, requested, current):
    """What a request's "model" asks for: ("current", None) for any name that is the running model or matches no file,
    ("switch", path) for one other model file (case-insensitive part of the file name; an exact name wins), or
    ("ambiguous", [names]) when several other files match."""
    wanted = str(requested or "").strip().lower()
    if not wanted:
        return "current", None
    if current and wanted in current.lower():
        return "current", None
    exact = [m for m in models if model_id(m).lower() == wanted]
    found = exact or [m for m in models if wanted in model_id(m).lower()]
    if len(found) == 1:
        return "switch", found[0]
    if found:
        return "ambiguous", [model_id(m) for m in found]
    return "current", None


def models_listing(models, running):
    """The /v1/models reply: the running model first, then every other local model (id = file stem)."""
    ids = ([running] if running else []) + [i for i in (model_id(m) for m in models) if i != running]
    data = [{"id": i, "object": "model", "type": "model", "display_name": i, "created": 0, "owned_by": "hearthwork"}
            for i in ids]
    # "models": [] keeps Codex quiet (see harnesses.make_relay); "data" is what everything else reads.
    return {"object": "list", "data": data, "models": [], "has_more": False,
            "first_id": ids[0] if ids else None, "last_id": ids[-1] if ids else None}


def error_body(kind, message):
    """An error both SDK families understand: OpenAI reads error.message, Anthropic reads type=error + error.type."""
    return {"type": "error", "error": {"type": kind, "message": message, "code": kind}}


# ---------- the server ----------

class Api:
    def __init__(self, config, port, key=None, allow_switch=False):
        self.config, self.port, self.key, self.allow_switch = config, port, key, allow_switch
        self.model_port = config["server"]["port"]
        self.lock = threading.Lock()  # one model switch at a time
        self.models = lambda: find_models(config["modelsDir"])

    def switch_to(self, model):
        """Restart the server with `model` (auto context) unless someone just did. True when it is running."""
        if served_model(self.model_port, timeout=5) == model_id(model):
            return True
        print(f"\n{YELLOW}Switching to {model.name} (a request asked for it)...{RESET}", flush=True)
        ok = start_background(self.config, model)
        if ok:
            self.config["lastModel"] = str(model)
            save_config(self.config)
        return ok

    def prepare_switch(self, handler):
        """Read a POST body for its "model"; switch when asked and allowed. Returns an error (status, body) or None.
        The body is put back for the relay."""
        body = handler.read_body()
        handler.rfile = io.BytesIO(body)
        try:
            requested = json.loads(body).get("model")
        except (ValueError, AttributeError):
            return None
        with self.lock:
            current = served_model(self.model_port, timeout=5)
            action, found = match_model(self.models(), requested, current)
            if action == "ambiguous":
                return 400, error_body("invalid_request_error",
                                       f"Model '{requested}' matches several models: {', '.join(found)}. Use a longer part of the name.")
            if action == "switch" and not self.switch_to(found):
                return 503, error_body("api_error", f"Could not start {found.name}; see the hearthwork api terminal.")
        return None

    def intercept(self, handler):
        """Answer the request itself (True) or let the relay forward it (False)."""
        path = handler.path.split("?")[0].rstrip("/")
        origin = handler.headers.get("Origin")
        if not origin_allowed(origin, self.key):
            handler._send_json(403, error_body("permission_error", f"Origin {origin} is not allowed. Without --api-key only "
                                               "pages on localhost / 127.0.0.1 may call this API; start it with --api-key to allow other sites."))
            return True
        if handler.command == "OPTIONS":  # CORS preflight: browsers send no key here
            handler.send_response(204)
            for name, value in preflight_headers(origin, self.key, handler.headers.get("Access-Control-Request-Headers")):
                handler.send_header(name, value)
            handler.send_header("content-length", "0")
            handler.end_headers()
            return True
        if not key_ok(self.key, handler.headers):
            handler._send_json(401, error_body("authentication_error", "Invalid or missing API key."))
            return True
        if path == "/health":
            return False
        if not path.startswith(JSON_PATHS):
            handler._send_json(404, error_body("not_found_error", f"No such endpoint: {path}"))
            return True
        if path == "/v1/embeddings":
            handler._send_json(501, error_body("not_implemented_error",
                                               "embeddings not enabled: the model server is not started with --embeddings, "
                                               "so `hearthwork api` offers chat, messages, responses and completions only."))
            return True
        if path == "/v1/models" and handler.command == "GET":
            handler._send_json(200, models_listing(self.models(), served_model(self.model_port)))
            return True
        if handler.command == "POST" and self.allow_switch:
            problem = self.prepare_switch(handler)
            if problem:
                handler._send_json(*problem)
                return True
        if not port_open(self.model_port):
            handler._send_json(503, error_body("api_error", "The model server is not running. Start it with `hearthwork start`."))
            return True
        return False


def start_last_model(config):
    """The running model's name; else start the last-used one without asking. None (with a message) when impossible."""
    port = config["server"]["port"]
    name = served_model(port) or (port_open(port) and served_model(port, timeout=60))
    if name:
        return name
    if port_open(port):
        print(f"{RED}Port {port} is in use but no model answers. Stop it with `hearthwork stop` first.{RESET}")
        return None
    last = config.get("lastModel")
    if not last or not Path(last).is_file():
        print(f"{RED}No last-used model is configured. Run `hearthwork start` once and pick one, then try again.{RESET}")
        return None
    return served_model(port) if start_background(config, Path(last)) else None


# ---------- setup snippets ----------

def snippets(port, key, model, windows=None):
    base, token = f"http://127.0.0.1:{port}", key or "local"
    if windows is None:
        windows = os.name == "nt"
    if windows:  # curl.exe in Windows PowerShell 5.1 mangles the \" quotes in -d, so show PowerShell instead
        examples = f"""PowerShell (OpenAI)
  $h = @{{ Authorization = "Bearer {token}" }}
  $body = @{{ model = "{model}"; messages = @(@{{ role = "user"; content = "Hello" }}) }} | ConvertTo-Json -Depth 5
  (Invoke-RestMethod {base}/v1/chat/completions -Method Post -Headers $h -ContentType "application/json" -Body $body).choices[0].message.content

PowerShell (Anthropic)
  $h = @{{ "x-api-key" = "{token}" }}
  $body = @{{ model = "{model}"; max_tokens = 300; messages = @(@{{ role = "user"; content = "Hello" }}) }} | ConvertTo-Json -Depth 5
  (Invoke-RestMethod {base}/v1/messages -Method Post -Headers $h -ContentType "application/json" -Body $body).content | Where-Object type -eq "text" | Select-Object -ExpandProperty text"""
    else:
        examples = f"""curl (OpenAI)
  curl {base}/v1/chat/completions -H "Authorization: Bearer {token}" -H "Content-Type: application/json" \\
    -d '{{"model": "{model}", "messages": [{{"role": "user", "content": "Hi"}}]}}'

curl (Anthropic)
  curl {base}/v1/messages -H "x-api-key: {token}" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" \\
    -d '{{"model": "{model}", "max_tokens": 512, "messages": [{{"role": "user", "content": "Hi"}}]}}'"""
    return f"""Hearthwork API: {base}   model: {model}   (this computer only; other devices: `hearthwork share`)
The model name is free: any name uses the running model{{switch_note}}.

OpenAI Python SDK
  from openai import OpenAI
  client = OpenAI(base_url="{base}/v1", api_key="{token}")
  print(client.chat.completions.create(model="{model}", messages=[{{"role": "user", "content": "Hi"}}]).choices[0].message.content)

Anthropic Python SDK
  from anthropic import Anthropic
  client = Anthropic(base_url="{base}", auth_token="{token}")
  print(client.messages.create(model="{model}", max_tokens=512, messages=[{{"role": "user", "content": "Hi"}}]).content[0].text)

{examples}

Environment variables
  OPENAI_BASE_URL={base}/v1   OPENAI_API_KEY={token}
  ANTHROPIC_BASE_URL={base}   ANTHROPIC_AUTH_TOKEN={token}

Aider
  aider --openai-api-base {base}/v1 --openai-api-key {token} --model openai/{model}

OpenCode (opencode.json)
  {{"provider": {{"hearthwork": {{"npm": "@ai-sdk/openai-compatible", "name": "Hearthwork",
    "options": {{"baseURL": "{base}/v1", "apiKey": "{token}"}}, "models": {{"{model}": {{}}}}}}}}}}

Continue (config.yaml)
  models:
    - name: Hearthwork
      provider: openai
      model: {model}
      apiBase: {base}/v1
      apiKey: {token}

Cursor
  Settings > Models > OpenAI API Key: {token}, Override OpenAI Base URL: {base}/v1, add model "{model}".
  Cursor sends some requests from its own servers, which cannot reach localhost: it may not work with localhost.
"""


# ---------- command ----------

def api_main(argv):
    parser = argparse.ArgumentParser(
        prog="hearthwork api",
        description="One local OpenAI + Anthropic compatible endpoint for the running model (Ctrl+C stops it). "
                    "It listens on this computer only (127.0.0.1); to use the model from other devices run `hearthwork share`.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--api-key", help="require this key (Authorization: Bearer or x-api-key); without it any program on this computer may use the model")
    parser.add_argument("--allow-switch", action="store_true",
                        help='a request whose "model" is part of another local model file name restarts the server with it '
                             "(interrupts everyone using the model); otherwise any model name uses the running model")
    parser.add_argument("--show", action="store_true", help="print copy-paste setup for the OpenAI/Anthropic SDKs, curl, Aider, OpenCode, Continue, Cursor and exit")
    args = parser.parse_args(argv)
    from .cli import configured
    from .onboard import load_config
    if args.show:
        config = load_config()
        model = served_model(config.get("server", {}).get("port", 8001)) or model_id(config.get("lastModel") or "<model>")
        print(snippets(args.port, args.api_key, model).replace("{switch_note}", " (with --allow-switch, a part of a local model file name switches to it)"
                                                               if args.allow_switch else ""))
        return 0
    config = configured(interactive=False)
    name = start_last_model(config)
    if not name:
        return 1
    api = Api(config, args.port, args.api_key, args.allow_switch)
    try:
        server = make_relay(api.model_port, ("127.0.0.1", args.port), api.intercept,
                            lambda handler: cors_headers(handler.headers.get("Origin"), api.key))
    except OSError as error:
        print(f"{RED}Cannot listen on port {args.port}: {error}{RESET}")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{args.port}"
    print(f"\n{GREEN}Hearthwork API ready.{RESET}  Ctrl+C stops it.")
    print(f"  OpenAI:     {base}/v1   (chat/completions, responses, completions, models; embeddings answer 501)")
    print(f"  Anthropic:  {base}   (v1/messages, v1/messages/count_tokens)")
    print(f"  Model:      {name}   context {agent_context(config):,}")
    print(f"  Key:        {'required' if args.api_key else 'none (any program on this computer can use it)'}"
          f"   Switching: {'on' if args.allow_switch else 'off (--allow-switch)'}")
    print(f"{CYAN}hearthwork api --show{RESET} for setup snippets.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped the API.")
    return 0
