"""`hearthwork doctor`: is this model + agent combination set up correctly, and if not, which layer is at fault?

Stages: environment (info only), server, protocol (per API, through the same relay the agents use), agent
(installed, configuration, headless reply) and a small coding task graded by Hearthwork itself. A protocol failure
points at the relay/template/API; a failed coding task with a sound protocol is the model's capability limit."""
import argparse
import contextlib
import hashlib
import http.client
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from . import __version__
from .harnesses import API_NAMES, HARNESSES, SECRET, installed, make_relay, preview, url_host

OK, FAIL, SKIP = "pass", "fail", "skip"
MARK = {OK: "✓", FAIL: "✗", SKIP: "–"}
SECRET_ANSWER = "ZEBRA-42"
MIN_CONTEXT = {"claude": 32768, "codex": 24576}  # tokens an agent's own prompt and tools need before any work
DEFAULT_MIN_CONTEXT = 16384
MAX_TOKENS = 1024  # thinking models spend part of it before they answer
PROTOCOL = ["plain reply", "streaming", "tool call", "tool result", "next turn", "late system message"]
TOOL_STAGES = ("tool call", "tool result", "next turn")
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}


# ---------- stages ----------

def stage(name, status, seconds=0.0, detail="", suggestion=None):
    found = {"name": name, "status": status, "seconds": round(seconds, 1), "detail": detail}
    if suggestion and status == FAIL:
        found["suggestion"] = suggestion
    return found


def timed(name, check):
    """Run `check() -> (status, detail, suggestion)` as a stage; an exception is a failed stage."""
    started = time.time()
    try:
        status, detail, suggestion = check()
    except (Exception, SystemExit) as error:
        plain = isinstance(error, (SystemExit, ProtocolError))
        status, detail, suggestion = FAIL, str(error) if plain else f"{type(error).__name__}: {error}", None
    return stage(name, status, time.time() - started, detail, suggestion)


# ---------- wire formats ----------
# A conversation is a list of ("system"|"user"|"assistant", text), ("call", id, name, arguments JSON), ("result", id, text).

TOOL = {"name": "get_secret", "description": "Look up a secret value by its name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}
PATHS = {"anthropic": "/v1/messages", "openai-responses": "/v1/responses", "openai-chat": "/v1/chat/completions"}


def wire(api, conv, model, tools=False, stream=False):
    """Request body for `conv` in the format of `api`."""
    body = {"model": model, "stream": stream}
    if api == "anthropic":
        system, messages, reminder = [], [], []
        for item in conv:
            kind = item[0]
            if kind == "system":
                (system if not messages else reminder).append(item[1])
            elif kind == "user":
                blocks = [{"type": "text", "text": f"<system-reminder>\n{t}\n</system-reminder>"} for t in reminder]
                messages.append({"role": "user", "content": blocks + [{"type": "text", "text": item[1]}]})
                reminder = []
            elif kind == "assistant":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": item[1]}]})
            elif kind == "call":
                messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": item[1], "name": item[2],
                                                                   "input": json.loads(item[3] or "{}")}]})
            else:
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": item[1], "content": item[2]}]})
        body.update(max_tokens=MAX_TOKENS, messages=messages)
        if system:
            body["system"] = [{"type": "text", "text": t} for t in system]
        if tools:
            body["tools"] = [{"name": TOOL["name"], "description": TOOL["description"], "input_schema": TOOL["parameters"]}]
    elif api == "openai-responses":
        items, seen_user = [], False
        for item in conv:
            kind = item[0]
            if kind in ("system", "user", "assistant"):
                role = "developer" if kind == "system" and seen_user else kind
                seen_user = seen_user or kind == "user"
                part = "output_text" if kind == "assistant" else "input_text"
                items.append({"role": role, "content": [{"type": part, "text": item[1]}]})
            elif kind == "call":
                items.append({"type": "function_call", "call_id": item[1], "name": item[2], "arguments": item[3]})
            else:
                items.append({"type": "function_call_output", "call_id": item[1], "output": item[2]})
        body.update(max_output_tokens=MAX_TOKENS, input=items)
        if tools:
            body["tools"] = [{"type": "function", **TOOL}]
    else:
        messages = []
        for item in conv:
            kind = item[0]
            if kind in ("system", "user", "assistant"):
                messages.append({"role": kind, "content": item[1]})
            elif kind == "call":
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": item[1], "type": "function", "function": {"name": item[2], "arguments": item[3]}}]})
            else:
                messages.append({"role": "tool", "tool_call_id": item[1], "content": item[2]})
        body.update(max_tokens=MAX_TOKENS, messages=messages)
        if tools:
            body["tools"] = [{"type": "function", "function": TOOL}]
    return body


def parse_reply(api, data):
    """(text, [(id, name, arguments JSON text)]) of a non-streaming reply."""
    text, calls = "", []
    if api == "anthropic":
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text") or ""
            elif block.get("type") == "tool_use":
                calls.append((block.get("id"), block.get("name"), json.dumps(block.get("input") or {})))
    elif api == "openai-responses":
        for item in data.get("output") or []:
            if item.get("type") == "message":
                text += "".join(c.get("text") or "" for c in item.get("content") or [])
            elif item.get("type") == "function_call":
                calls.append((item.get("call_id") or item.get("id"), item.get("name"), item.get("arguments") or ""))
    else:
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        text = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls.append((call.get("id"), function.get("name"), function.get("arguments") or ""))
    return text, calls


class Endpoint:
    """Where the protocol checks go: a base URL (the local relay or the host) and the device key, if any."""

    def __init__(self, host, port, key=None, timeout=120):
        self.host, self.port, self.key, self.timeout = host, port, key, timeout

    def post(self, api, body):
        """(status, response object) of a POST; the object streams (iterable of lines) when body["stream"]."""
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
            headers["x-api-key"] = self.key
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        connection.request("POST", PATHS[api], body=json.dumps(body).encode(), headers=headers)
        return connection.getresponse(), connection

    def send(self, api, body):
        response, connection = self.post(api, body)
        try:
            raw = response.read().decode("utf-8", "replace")
        finally:
            connection.close()
        if response.status != 200:
            raise ProtocolError(f"HTTP {response.status}: {short(raw)}")
        try:
            return json.loads(raw)
        except ValueError:
            raise ProtocolError(f"the reply is not JSON: {short(raw)}")

    def stream(self, api, body):
        """[(event name, data text)] of a streaming reply, read to its end."""
        response, connection = self.post(api, body)
        events, name = [], ""
        try:
            if response.status != 200:
                raise ProtocolError(f"HTTP {response.status}: {short(response.read().decode('utf-8', 'replace'))}")
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    events.append((name, line[5:].strip()))
                    name = ""
        finally:
            connection.close()
        return events


class ProtocolError(Exception):
    pass


def short(text, limit=160):
    return re.sub(r"\s+", " ", text).strip()[:limit]


def stream_summary(api, events):
    """(content chunks, ended properly) of a streamed reply."""
    chunks, ended = 0, False
    for name, data in events:
        if data == "[DONE]":
            ended = ended or api == "openai-chat"
            continue
        try:
            item = json.loads(data)
        except ValueError:
            continue
        kind = item.get("type") or name
        if api == "anthropic":
            delta = item.get("delta") or {}
            if kind == "content_block_delta" and (delta.get("text") or delta.get("thinking") or delta.get("partial_json")):
                chunks += 1
            ended = ended or kind == "message_stop"
        elif api == "openai-responses":
            chunks += kind.endswith(".delta") and bool(item.get("delta"))
            ended = ended or kind in ("response.completed", "response.incomplete")
        else:
            delta = ((item.get("choices") or [{}])[0]).get("delta") or {}
            chunks += bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))
    return chunks, ended


# ---------- protocol stages ----------

LOOKS_LIKE_CALL = re.compile(r"<tool_call|<function=|\"name\"\s*:|get_secret\s*[({]|\[TOOL_CALLS\]|tool_use", re.I)
TEXT_CALL = ("The model wrote the tool call as plain text instead of a structured call: its chat template or the model "
             "does not produce tool calls the server can parse. See Hearthwork issue #9 (tool-call repair); try a "
             "model/template built for tool use.")


def protocol_stages(api, endpoint, model):
    """The six protocol stages for `api`, in order; later ones are skipped when they would be meaningless."""
    found, state = [], {}

    def run(name, check, needs=None):
        if needs and any(s["status"] != OK for s in found if s["name"] == needs):
            found.append(stage(name, SKIP, 0, f"needs '{needs}' to pass first"))
            return
        found.append(timed(name, lambda: check()))

    def plain():
        text, _ = parse_reply(api, endpoint.send(api, wire(api, [("user", "Reply with the word ok.")], model)))
        if not text.strip():
            return FAIL, "HTTP 200 but the reply has no text", (
                "The model answered with an empty message: raise max tokens, or check the chat template "
                "(a thinking model may spend everything on thinking). See `hearthwork status` and the server log.")
        return OK, f"reply: {short(text, 40)!r}", None

    def streaming():
        conv = [("user", "Count from 1 to 5, one number per line.")]
        chunks, ended = stream_summary(api, endpoint.stream(api, wire(api, conv, model, stream=True)))
        if not ended:
            return FAIL, f"{chunks} chunks, then the stream ended without its end event", (
                f"The {API_NAMES[api]} stream ended without its end event: a relay/API issue (the connection was cut "
                "or the server stopped mid-reply). Retry; if it repeats, check the llama-server log and the relay.")
        if chunks < 2:
            return FAIL, f"only {chunks} content chunk(s): the reply was not streamed", (
                f"The {API_NAMES[api]} reply arrived as one piece, not a stream: a relay or proxy in between is "
                "buffering it. Agents will look frozen until the whole reply is done.")
        return OK, f"{chunks} chunks and an end event", None

    def tool_call():
        conv = [("user", 'Use the get_secret tool to look up the secret named "alpha". Do not answer without calling it.')]
        state["conv"] = conv
        text, calls = parse_reply(api, endpoint.send(api, wire(api, conv, model, tools=True)))
        for call_id, name, arguments in calls:
            try:
                parsed = json.loads(arguments or "{}")
            except ValueError:
                return FAIL, f"tool call with invalid JSON arguments: {short(arguments, 60)}", (
                    "The tool call arguments are not valid JSON: the model or its template produced a broken call. "
                    "See Hearthwork issue #9 (tool-call repair).")
            if name == TOOL["name"] and "alpha" in json.dumps(parsed).lower():
                state["call"] = (call_id or "call_1", name, json.dumps(parsed))
                return OK, f"get_secret({json.dumps(parsed)})", None
            return FAIL, f"tool call {name}({short(arguments, 40)}) does not ask for 'alpha'", (
                "The model called the tool with the wrong name or arguments: a capability limit of the model.")
        if LOOKS_LIKE_CALL.search(text):
            return FAIL, f"the call came back as text: {short(text, 80)!r}", TEXT_CALL
        return FAIL, f"no tool call; the model answered in text: {short(text, 60)!r}", (
            "The model did not call the offered tool. Check that the server runs with --jinja (Hearthwork does) and "
            "that the model supports tool use; otherwise it is a model capability limit.")

    def tool_result():
        conv = state["conv"] + [("call", *state["call"]), ("result", state["call"][0], SECRET_ANSWER)]
        text, _ = parse_reply(api, endpoint.send(api, wire(api, conv, model, tools=True)))
        state["conv"], state["answer"] = conv, text
        if SECRET_ANSWER not in text:
            return FAIL, f"final text lacks {SECRET_ANSWER}: {short(text, 60)!r}", (
                "After the tool result the model did not use it: the tool result may be dropped by the chat template "
                "(role mapping) or the model ignores it. Check the template with the Environment fingerprint.")
        return OK, "answer contains the tool result", None

    def next_turn():
        conv = state["conv"] + [("assistant", state["answer"]), ("user", "What was the secret?")]
        text, _ = parse_reply(api, endpoint.send(api, wire(api, conv, model, tools=True)))
        if SECRET_ANSWER not in text:
            return FAIL, f"second turn lacks {SECRET_ANSWER}: {short(text, 60)!r}", (
                "The history with the tool call and result is not carried into the next turn: a chat template "
                "problem with tool messages in history.")
        return OK, "history with the tool call survives", None

    def late_system():
        conv = [("system", "You are terse."), ("user", "Reply with the word ok."), ("assistant", "ok"),
                ("system", "Always answer in one word."), ("user", "Reply with the word fine.")]
        text, _ = parse_reply(api, endpoint.send(api, wire(api, conv, model)))
        if not text.strip():
            return FAIL, "HTTP 200 but the reply has no text", "An empty reply after a late system message: check the chat template."
        return OK, "accepted", None

    def late_system_suggestion(check):
        def wrapped():
            try:
                return check()
            except ProtocolError as error:
                return FAIL, str(error), (
                    "The server rejects a system/developer message after the first user turn (a strict chat template, "
                    "e.g. Qwen3.5). Hearthwork's relay normally fixes this: make sure the agent goes through it "
                    "(`hearthwork <agent> --dry-run`), or use a fixed template (`hearthwork` data folder, templates/).")
        return wrapped

    run("plain reply", plain)
    run("streaming", streaming)
    run("tool call", tool_call)
    run("tool result", tool_result, "tool call")
    run("next turn", next_turn, "tool result")
    run("late system message", late_system_suggestion(late_system))
    return found


def protocol_error_hint(found):
    """Failed stages that only carry an HTTP error get a generic, specific-enough suggestion."""
    for item in found:
        if item["status"] == FAIL and "suggestion" not in item:
            item["suggestion"] = (f"The model server returned an error ({short(item['detail'], 90)}). Check that the model is "
                                  "loaded (`hearthwork status`) and read its log; a 500 usually comes from the chat template.")
    return found


# ---------- environment ----------

def get_json(url, headers=None, timeout=3):
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def model_path(config, name):
    """The GGUF file of the served model, or None."""
    last = config.get("lastModel")
    if last and Path(last).stem == name and Path(last).exists():
        return Path(last)
    folder = config.get("modelsDir")
    if folder and Path(folder).is_dir():
        return next(iter(Path(folder).rglob(f"{name}.gguf")), None)
    return None


def quantization(path, name):
    from .context import read_metadata
    found = FILE_TYPES.get(read_metadata(path).get("general.file_type")) if path else None
    if found:
        return found
    match = re.search(r"(IQ\d_\w+|Q\d_\w+|Q\d|BF16|F16|F32)", name, re.I)
    return match.group(1).upper() if match else None


def environment(config, remote):
    from . import runtime
    from .server import agent_context, running_context
    env = {"hearthwork": __version__, "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
           "python": platform.python_version()}
    if remote:
        info = get_json(f"http://{url_host(remote['host'])}:{remote['port']}/hearthwork/info") or {}
        env.update(mode="remote", host=remote.get("hostName") or remote["host"], route=remote.get("via") or "?",
                   hostVersion=info.get("hearthwork"), model=info.get("model"), runningContext=info.get("context"),
                   slots=info.get("slots"))
        return env
    port = config["server"]["port"]
    chosen = runtime.selected(config)
    from .server import served_model
    name = served_model(port)
    path = model_path(config, name) if name else None
    env.update(mode="local", runtime=f"{chosen[0]} b{chosen[1]}" if chosen else None,
               llamaCppBuild=chosen[1] if chosen else None, model=name, modelFile=path.name if path else (name + ".gguf" if name else None),
               quantization=quantization(path, name or ""), runningContext=running_context(port) if name else None)
    if path:
        from .context import model_info
        info = model_info(path) or {}
        env.update(architecture=info.get("arch"), trainedContext=info.get("modelMax"))
    props = get_json(f"http://127.0.0.1:{port}/props") if name else None
    template = (props or {}).get("chat_template")
    env["chatTemplate"] = hashlib.sha256(template.encode()).hexdigest()[:12] if isinstance(template, str) and template else None
    slots = get_json(f"http://127.0.0.1:{port}/slots") if name else None
    env["slots"] = (props or {}).get("total_slots") or (len(slots) if isinstance(slots, list) else None)
    if env["runningContext"] is None and name:
        env["runningContext"] = agent_context(config)
    return env


def server_stage(config, remote):
    """(stage, model name, context)."""
    from .server import port_open, running_context, served_model

    def local():
        port = config["server"]["port"]
        name = served_model(port, timeout=5)
        if name:
            health = get_json(f"http://127.0.0.1:{port}/health")
            return OK, f"{name} on port {port}" + (f" (health: {health.get('status')})" if health else ""), None
        if port_open(port):
            return FAIL, f"port {port} is open but no model answers /v1/models", (
                f"Something listens on port {port} but is not serving a model (still loading, or not llama-server). "
                "Wait a minute, or run `hearthwork stop` and `hearthwork start`.")
        return FAIL, f"nothing answers on port {port}", (
            f"No model server on port {port}. Start one with `hearthwork start`, or point Hearthwork at another "
            "computer with `hearthwork connect`.")

    def remote_check():
        from .remote import RemoteError, session
        try:
            name, context = session(config)
        except RemoteError as error:
            return FAIL, short(str(error), 300), str(error).splitlines()[-1] if "\n" in str(error) else (
                "Run `hearthwork connect` to pair again, or `hearthwork disconnect` to go back to local models.")
        return OK, f"{name} on {config['remote'].get('hostName') or config['remote']['host']} ({config['remote'].get('via')})", None

    item = timed("server", remote_check if remote else local)
    return item


# ---------- agents ----------

def agent_stages(key, config, remote, work, coding, context):
    """Stages 3 and 4 for one agent: installed, configuration, headless reply, coding task."""
    from . import bench
    from .harnesses import agent_version
    from .server import agent_context
    from .task import run_task
    harness = HARNESSES[key]
    found = []
    if not installed(key):
        found.append(stage("installed", FAIL, 0, "not found on PATH", f"Install {harness['title']}: {harness['install']}"))
        return found + [stage(n, SKIP, 0, "agent not installed") for n in ("configuration", "headless reply") + (("coding task",) if coding else ())]
    started = time.time()
    version = agent_version(key)
    found.append(stage("installed", OK, time.time() - started, f"version {version}"))

    def configuration():
        nonlocal context
        context = context or agent_context(config)
        need = MIN_CONTEXT.get(key, DEFAULT_MIN_CONTEXT)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            preview(key, "doctor-model", context, remote=remote)
        text = out.getvalue()
        if remote and remote["key"] in text:
            return FAIL, "the dry run would print the device key", "Bug in the dry-run masking; please report."
        if context and context < need:
            return FAIL, f"context {context:,} is below the {need:,} {harness['title']} needs", (
                f"The model runs with {context:,} tokens of context, too little for {harness['title']}'s own prompt and "
                f"tools. Raise it: `hearthwork context` (then restart the model), needs at least {need // 1024}K.")
        return OK, f"builds ({len(text.splitlines())} lines of dry run, secrets masked), context {context:,}", None

    found.append(timed("configuration", configuration))
    if found[-1]["status"] != OK:
        return found + [stage(n, SKIP, 0, "configuration failed") for n in ("headless reply",) + (("coding task",) if coding else ())]

    def headless(task, folder, timeout, grade=None):
        def check():
            config_dir = Path(work) / f"{key}-config"
            env = bench.isolated_env(key, config_dir)
            before = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    report = run_task(config, task, key, cwd=folder, timeout=timeout)
            finally:
                for k, v in before.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            if report["status"] != "ok":
                return FAIL, f"{report['status']}: {short(report.get('error') or '', 200)}", (
                    f"{harness['title']} did not complete the run ({report['status']}). Run "
                    f"`hearthwork task --agent {key} \"Reply with the word ok\"` to see its output, and `hearthwork {key} --dry-run` for its configuration.")
            return grade(report, folder) if grade else (OK, f"replied {short(report['message'], 40)!r} in {report['duration_seconds']} s", None)
        return check

    folder = Path(work) / f"{key}-reply"
    folder.mkdir(parents=True, exist_ok=True)
    found.append(timed("headless reply", headless("Reply with the word ok.", str(folder), 240,
                                                   lambda r, f: (OK, f"replied {short(r['message'], 40)!r}", None) if (r["message"] or "").strip()
                                                   else (FAIL, "the agent finished without a reply", f"{harness['title']} produced no final message; see `hearthwork task --agent {key} \"Reply with the word ok\"`."))))
    if not coding:
        return found
    if found[-1]["status"] != OK:
        return found + [stage("coding task", SKIP, 0, "headless reply failed")]
    folder = Path(work) / f"{key}-coding"
    folder.mkdir(parents=True, exist_ok=True)
    text = "Create a file calc.py in this folder with a function add(a, b) that returns a + b" + (
        "." if harness.get("tools") is False else ", then run it to check that it works.")

    def grade(report, folder):
        code, output = bench.run_py(folder, "-c", "import calc; assert calc.add(2, 3) == 5; print('GRADE-OK')")
        if code == 0 and "GRADE-OK" in output:
            return OK, f"calc.add(2, 3) == 5 in {report['duration_seconds']} s", None
        return FAIL, f"graded by Hearthwork: {short(output.splitlines()[-1] if output.strip() else 'no calc.py created', 120)}", (
            "The protocol works, but the model did not solve the coding task: a capability limit of this model "
            f"(or its quantization), not a configuration problem. Try a larger model; compare with `hearthwork bench --agent {key}`.")

    found.append(timed("coding task", headless(text, str(folder), 600, grade)))
    return found


def uses_tools(key):
    return HARNESSES[key].get("tools", True)


# ---------- verdicts ----------

def verdict(key, server, protocol, agent):
    """(text, failing stage name or None) for one agent. `protocol` are the stages of its API."""
    api = API_NAMES[HARNESSES[key]["api"]]
    relevant = [s for s in protocol if uses_tools(key) or s["name"] not in TOOL_STAGES]
    ordered = [("configuration/connection problem", [server] + [s for s in agent if s["name"] in ("installed", "configuration")]),
               ("protocol problem", relevant),
               ("configuration/connection problem", [s for s in agent if s["name"] == "headless reply"]),
               ("coding", [s for s in agent if s["name"] == "coding task"])]
    for kind, items in ordered:
        for item in items:
            if item["status"] == FAIL:
                if kind == "coding":
                    return "protocol OK, but the model failed the coding task (model capability)", item["name"]
                where = f"{item['name']} ({api})" if kind == "protocol problem" else item["name"]
                return f"{kind} at {where}", item["name"]
    if any(s["status"] == SKIP and s["name"] != "coding task" for s in agent):
        return "not checked", None
    return "works", None


# ---------- run ----------

def scrub(value, secrets):
    """Copy of `value` without credentials: known keys and anything shaped like a key/token assignment."""
    if isinstance(value, dict):
        return {k: ("********" if SECRET.search(str(k)) and isinstance(v, str) else scrub(v, secrets)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "********")
        return re.sub(r"(?i)(bearer\s+|sk-)[\w.~+/=-]{8,}", r"\1********", value)
    return value


def run_doctor(config, agents, quick=False, keep=False, endpoint_override=None):
    """Run every stage; returns the report dict. `endpoint_override` = (Endpoint, model) is for tests."""
    remote = config.get("remote")
    started = time.time()
    report = {"time": datetime.now().isoformat(timespec="seconds"), "mode": "remote" if remote else "local"}
    report["server"] = server_stage(config, remote)
    report["environment"] = environment(config, remote) if report["server"]["status"] == OK else {"hearthwork": __version__, "os": platform.platform()}
    selected = [a for a in agents]
    report["protocol"], report["agents"] = {}, {}
    work = tempfile.mkdtemp(prefix="hearthwork-doctor-")
    relay = None
    try:
        if report["server"]["status"] == OK:
            model = report["environment"].get("model") or "model"
            if endpoint_override:
                endpoint, model = endpoint_override
            elif remote:
                endpoint = Endpoint(remote["host"], remote["port"], remote["key"])
            else:
                relay = make_relay(config["server"]["port"])
                threading.Thread(target=relay.serve_forever, daemon=True).start()
                endpoint = Endpoint("127.0.0.1", relay.server_address[1])
            for api in sorted({HARNESSES[a]["api"] for a in selected}):
                found = protocol_error_hint(protocol_stages(api, endpoint, model))
                report["protocol"][api] = found
        for key in selected:
            harness = HARNESSES[key]
            if report["server"]["status"] != OK:
                agent = [stage("not run", SKIP, 0, "the model server check failed")]
            else:
                agent = agent_stages(key, config, remote, work, not quick, report["environment"].get("runningContext"))
            protocol = report["protocol"].get(harness["api"], [])
            text, at = verdict(key, report["server"], protocol, agent)
            report["agents"][key] = {"title": harness["title"], "api": harness["api"], "version": None, "stages": agent,
                                     "verdict": text, "failedStage": at, "toolsApplicable": uses_tools(key)}
            for item in agent:
                if item["name"] == "installed" and item["status"] == OK:
                    report["agents"][key]["version"] = item["detail"].replace("version ", "")
    finally:
        if relay:
            relay.shutdown()
        if keep:
            report["kept"] = work
        else:
            shutil.rmtree(work, ignore_errors=True)
    report["seconds"] = round(time.time() - started, 1)
    report["suggestions"] = suggestions(report)
    report["ok"] = all(a["verdict"] == "works" for a in report["agents"].values()) and report["server"]["status"] == OK
    secrets = [remote["key"]] if remote and remote.get("key") else []
    return scrub(report, secrets)


def suggestions(report):
    """Distinct suggestions of failed stages, in the order they are met."""
    found = []
    items = [report["server"]] + [s for p in report["protocol"].values() for s in p] + [s for a in report["agents"].values() for s in a["stages"]]
    for item in items:
        text = item.get("suggestion")
        if item["status"] == FAIL and text and text not in found:
            found.append(text)
    return found


# ---------- output ----------

def table(stages):
    width = max(len(s["name"]) for s in stages)
    return [f"  {MARK[s['status']]} {s['name'].ljust(width)}  {str(s['seconds']).rjust(6)} s  {s['detail']}" for s in stages]


def format_report(report):
    env = report["environment"]
    labels = [("hearthwork", "hearthwork"), ("os", "OS"), ("runtime", "runtime"), ("modelFile", "model file"), ("model", "model"),
              ("quantization", "quantization"), ("architecture", "architecture"), ("trainedContext", "trained context"),
              ("chatTemplate", "template fingerprint"), ("runningContext", "running context"), ("slots", "slots"),
              ("host", "host"), ("route", "route"), ("hostVersion", "host version")]
    lines = ["Hearthwork doctor", ""]
    lines += [f"  {label.ljust(20)} {env[key]:,}" if isinstance(env.get(key), int) and key in ("trainedContext", "runningContext")
              else f"  {label.ljust(20)} {env[key]}" for key, label in labels if env.get(key) not in (None, "")]
    lines += ["", "Model server"] + table([report["server"]])
    for api, stages in report["protocol"].items():
        users = ", ".join(k for k, a in report["agents"].items() if a["api"] == api)
        lines += ["", f"Protocol: {API_NAMES[api]} (used by {users})"] + table(stages)
    for key, agent in report["agents"].items():
        lines += ["", f"{agent['title']}" + (f" {agent['version']}" if agent.get("version") else "")
                  + ("" if agent["toolsApplicable"] else "  (edits files itself, no tool calls: tool stages not applicable)")] + table(agent["stages"])
    lines += ["", "Verdict"] + [f"  {a['title']}: {a['verdict']}" for a in report["agents"].values()]
    if report["suggestions"]:
        lines += ["", "Suggestions"] + [f"  {n}. {text}" for n, text in enumerate(report["suggestions"], 1)]
    return "\n".join(lines)


def save(report):
    from .paths import HOME
    folder = HOME / "doctor"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv):
    parser = argparse.ArgumentParser(prog="hearthwork doctor", description="Check that the model server, the relay and your coding agents work together.")
    parser.add_argument("--agent", choices=sorted(HARNESSES), help="check one agent (default: every installed agent)")
    parser.add_argument("--all", action="store_true", help="every installed agent (the default)")
    parser.add_argument("--quick", action="store_true", help="skip the coding task")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folders")
    args = parser.parse_args(argv)
    from .procs import setup_stdio
    from .onboard import load_config
    setup_stdio()
    config = load_config()
    agents = [args.agent] if args.agent else [k for k in HARNESSES if installed(k)]
    if not agents:
        print("No supported agent is installed. `hearthwork agents` lists them with install commands.")
        return 1
    report = run_doctor(config, agents, args.quick, args.keep)
    path = save(report)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"saved: {path}", file=sys.stderr)
    else:
        print(format_report(report))
        print(f"\nsaved: {path}" + (f"\nkept: {report['kept']}" if report.get("kept") else ""))
    return 0 if report["ok"] else 1
