"""Tool-call repair: recover a tool call a model wrote as plain text.

Some models write their call in a format the server's chat template does not parse, so the reply arrives as text and
the agent never runs it. `find_tool_call` recognises the common text formats and validates the result against the
request's tools (name and JSON schema; never invented). The relay uses `repair_response` for whole replies and
`StreamRepair` for streams: it passes events through untouched until the assistant text starts to look like a call,
holds the text back from there, and at the end either emits proper tool-call events or releases the held text as it was.
"""
import ast
import json
import os
import re
import secrets
from collections import namedtuple

FORMATS = ("mistral", "calling-tool", "tool_call", "function", "xml", "pythonic", "json")
Repair = namedtuple("Repair", "calls text format")  # calls: [{"name", "arguments": dict}], text: what is left outside them


def enabled():
    return os.environ.get("HEARTHWORK_TOOL_REPAIR", "1").strip().lower() not in ("0", "false", "off", "no")


# ---------- the request's tools ----------

def request_tools(api, payload):
    """{tool name: JSON schema} of a request ("anthropic" / "responses" / "chat"); empty when it offers none
    (or forbids them with tool_choice "none")."""
    if not isinstance(payload, dict) or payload.get("tool_choice") == "none":
        return {}
    found = {}
    for tool in payload.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if api == "anthropic":
            name, schema = tool.get("name"), tool.get("input_schema")
        elif api == "chat":
            function = tool.get("function") or {}
            name, schema = function.get("name"), function.get("parameters")
        else:
            name, schema = tool.get("name"), tool.get("parameters")
            if tool.get("type") not in (None, "function"):
                continue
        if isinstance(name, str) and name:
            found[name] = schema if isinstance(schema, dict) else {}
    return found


def _type_ok(value, kind):
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True  # a schema feature we do not check


def valid_call(name, arguments, tools):
    """True when `name` is one of `tools` and `arguments` is an object satisfying its schema's required keys and basic types."""
    if name not in tools or not isinstance(arguments, dict):
        return False
    schema = tools[name]
    if any(key not in arguments for key in schema.get("required") or ()):
        return False
    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        kind = (properties.get(key) or {}).get("type") if isinstance(properties.get(key), dict) else None
        if kind is None:
            continue
        kinds = kind if isinstance(kind, list) else [kind]
        if not any(_type_ok(value, k) for k in kinds):
            return False
    return True


# ---------- parsing the text formats ----------

class _Bad(Exception):
    """A call-looking text that is not a valid call: leave the reply alone."""


_JSON = json.JSONDecoder()
_ARG_KEYS = ("arguments", "parameters", "input", "args")


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # the template opened <think> in the prompt
        text = text.split("</think>")[-1]
    return text.split("<think>")[0]


def _decode(text, index):
    """(value, end) of the JSON value at `index` (after whitespace)."""
    while index < len(text) and text[index].isspace():
        index += 1
    try:
        return _JSON.raw_decode(text, index)
    except ValueError:
        raise _Bad()


def _json_call(item):
    """(name, arguments) of a decoded {"name": .., "arguments": ..} object (or the OpenAI {"function": {..}} wrapper)."""
    if not isinstance(item, dict):
        raise _Bad()
    if isinstance(item.get("function"), dict):
        item = item["function"]
    name = item.get("name")
    arguments = next((item[k] for k in _ARG_KEYS if k in item), {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            raise _Bad()
    if not isinstance(name, str):
        raise _Bad()
    return name, arguments


def _coerce(value, prop):
    """A text parameter value converted to the type its schema property declares."""
    kind = prop.get("type") if isinstance(prop, dict) else None
    kinds = kind if isinstance(kind, list) else [kind]
    if "string" in kinds:
        return value
    try:
        parsed = json.loads(value.strip())
    except ValueError:
        return value
    if kind is None:
        return parsed if isinstance(parsed, (dict, list)) else value
    return parsed if any(_type_ok(parsed, k) for k in kinds) else value


def _params(body, pattern, schema):
    properties = (schema or {}).get("properties") or {}
    out = {}
    for match in re.finditer(pattern, body, re.S):
        value = match.group("v")
        value = value[1:] if value.startswith("\n") else value
        value = value[:-1] if value.endswith("\n") else value
        out[match.group("k")] = _coerce(value, properties.get(match.group("k")))
    return out


def _names(tools):
    return "|".join(re.escape(n) for n in sorted(tools, key=len, reverse=True))


def _mistral(text, tools):
    marker = text.find("[TOOL_CALLS]")
    if marker < 0:
        return None
    start = marker + len("[TOOL_CALLS]")
    rest = text[start:].lstrip()
    if rest[:1] in ("[", "{"):
        value, end = _decode(text, start)
        items = value if isinstance(value, list) else [value]
        if not items:
            raise _Bad()
        return [_json_call(i) for i in items], [(marker, end)]
    calls, spans, index = [], [], start  # v3 form: name[ARGS]{json}
    pattern = re.compile(r"\s*([\w.\-]+)\s*(?:\[ARGS\])?\s*(?=\{)")
    while (match := pattern.match(text, index)):
        value, index = _decode(text, match.end())
        calls.append((match.group(1), value))
    if not calls:
        raise _Bad()
    return calls, [(marker, index)]


def _calling_tool(text, tools):
    calls, spans = [], []
    for match in re.finditer(r"\[Calling tool:\s*([\w.\-]+)\s*\(", text):
        if spans and match.start() < spans[-1][1]:
            continue
        index = match.end()
        arguments = {}
        if not text[index:].lstrip().startswith(")"):
            arguments, index = _decode(text, index)
        close = re.compile(r"\s*\)\s*\]").match(text, index)
        if not close:
            raise _Bad()
        calls.append((match.group(1), arguments))
        spans.append((match.start(), close.end()))
    return (calls, spans) if calls else None


def _function_body(name, body, tools):
    body = body.strip()
    if "<parameter=" in body:
        return name, _params(body, r"<parameter=(?P<k>[^>\s]+)>(?P<v>.*?)</parameter>", tools.get(name))
    if not body:
        return name, {}
    value, _ = _decode(body, 0)
    return name, value


def _functions(text, tools, within=None):
    calls, spans = [], []
    for match in re.finditer(r"<function=(?P<n>[^>\s]+)>(?P<b>.*?)(?:</function>|\Z)", text, re.S):
        calls.append(_function_body(match.group("n"), match.group("b"), tools))
        spans.append((match.start(), match.end()))
    return (calls, spans) if calls else None


def _tool_call(text, tools):
    calls, spans = [], []
    for match in re.finditer(r"<tool_call>(?P<b>.*?)(?:</tool_call>|\Z)", text, re.S):
        inner = match.group("b").strip()
        if inner.startswith("{") or inner.startswith("["):
            value, _ = _decode(inner, 0)
            calls.extend(_json_call(i) for i in (value if isinstance(value, list) else [value]))
        elif "<function=" in inner:
            found = _functions(inner, tools)
            if not found:
                raise _Bad()
            calls.extend(found[0])
        elif "<name>" in inner:
            name = re.search(r"<name>(.*?)</name>", inner, re.S)
            args = re.search(r"<arguments>(.*?)</arguments>", inner, re.S)
            if not name:
                raise _Bad()
            value = _decode(args.group(1), 0)[0] if args and args.group(1).strip() else {}
            calls.append((name.group(1).strip(), value))
        else:
            raise _Bad()
        spans.append((match.start(), match.end()))
    return (calls, spans) if calls else None


def _xml(text, tools):
    calls, spans = [], []
    for match in re.finditer(r"<invoke\s+name=[\"'](?P<n>[^\"']+)[\"']\s*>(?P<b>.*?)</invoke>", text, re.S):
        args = _params(match.group("b"), r"<parameter\s+name=[\"'](?P<k>[^\"']+)[\"']\s*>(?P<v>.*?)</parameter>",
                       tools.get(match.group("n")))
        calls.append((match.group("n"), args))
        spans.append((match.start(), match.end()))
    if tools:
        for match in re.finditer(rf"<(?P<n>{_names(tools)})>(?P<b>.*?)</(?P=n)>", text, re.S):
            body = match.group("b").strip()
            if not body:
                args = {}
            elif body.startswith("{"):
                args = _decode(body, 0)[0]
            else:
                args = _params(body, r"<(?P<k>\w+)>(?P<v>.*?)</(?P=k)>", tools.get(match.group("n")))
                if not args:
                    raise _Bad()
            calls.append((match.group("n"), args))
            spans.append((match.start(), match.end()))
    return (calls, sorted(spans)) if calls else None


def _call_end(text, index):
    """Index after the parenthesis that closes the one at `index`, skipping strings; -1 if it never closes."""
    depth, quote, i = 0, None, index
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        elif c in "\"'":
            quote = c * 3 if text.startswith(c * 3, i) else c
            i += len(quote)
            continue
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _literal(node):
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise _Bad()


def _python_call(source):
    try:
        node = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        raise _Bad()
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args
            and all(k.arg for k in node.keywords)):
        raise _Bad()
    return node.func.id, {k.arg: _literal(k.value) for k in node.keywords}


def _pythonic(text, tools):
    if not tools:
        return None
    names = _names(tools)
    first = re.compile(rf"^[ \t]*`*\[?[ \t]*(?P<n>{names})[ \t]*\(", re.M)
    more = re.compile(rf"\s*,\s*(?P<n>{names})[ \t]*\(")
    calls, spans, pos = [], [], 0
    while (match := first.search(text, pos)):
        index = match.start("n")
        start, bracket = match.start(), "[" in text[match.start():index]
        while match:
            open_at = match.end() - 1
            end = _call_end(text, open_at)
            if end < 0:
                raise _Bad()
            calls.append(_python_call(text[match.start("n"):end]))
            index = end
            match = more.match(text, index) if bracket else None
        tail = re.compile(r"\s*\]?`*").match(text, index)
        index = tail.end() if tail else index
        spans.append((start, index))
        pos = index
    return (calls, spans) if calls else None


def _json_objects(text, tools):
    calls, spans, pos = [], [], 0
    while (index := text.find("{", pos)) >= 0:
        try:
            value, end = _JSON.raw_decode(text, index)
        except ValueError:
            pos = index + 1
            continue
        pos = end
        inner = value.get("function") if isinstance(value, dict) and isinstance(value.get("function"), dict) else value
        if isinstance(inner, dict) and isinstance(inner.get("name"), str) and any(k in inner for k in _ARG_KEYS):
            calls.append(_json_call(value))
            spans.append((index, end))
    return (calls, spans) if calls else None


_DETECTORS = (("mistral", _mistral), ("calling-tool", _calling_tool), ("tool_call", _tool_call),
              ("function", _functions), ("xml", _xml), ("pythonic", _pythonic), ("json", _json_objects))
_LEFTOVER = re.compile(r"</?(?:tool_call|tool_calls|function_calls)>|```(?:json)?\s*```")


def find_tool_call(text, tools):
    """Repair(calls, text, format) when `text` holds valid tool calls written as plain text, else None.
    `tools` is {name: schema} (see request_tools). Calls are validated against it; any invalid one means None."""
    if not tools or not isinstance(text, str) or not text.strip():
        return None
    text = strip_think(text)
    for fmt, detect in _DETECTORS:
        try:
            found = detect(text, tools)
            if not found:
                continue
            raw, spans = found
            calls = []
            for name, arguments in raw:
                if not valid_call(name, arguments, tools):
                    return None
                calls.append({"name": name, "arguments": arguments})
        except _Bad:
            return None
        outside, last = [], 0
        for start, end in sorted(spans):
            outside.append(text[last:start])
            last = end
        outside.append(text[last:])
        return Repair(calls, _LEFTOVER.sub("", "".join(outside)).strip(), fmt)
    return None


# ---------- streaming: when does text look like the start of a call? ----------

MARKERS = ("[TOOL_CALLS]", "[Calling tool:", "<tool_call>", "<function=", "<invoke ")
NO, MAYBE, YES = 0, 1, 2


def hold_state(text, names, window=0):
    """YES when `text` contains the start of a call, MAYBE when it ends in what could become one, NO otherwise.
    `window`: only look at the last `window` characters (aligned to a line start)."""
    if window and len(text) > window:
        text = text[text.rfind("\n", 0, len(text) - window) + 1:]
    if any(m in text for m in MARKERS):
        return YES
    for name in names:
        if f"<{name}>" in text:
            return YES
    if names and re.search(rf"(?m)^[ \t]*`*\[?[ \t]*(?:{_names(names)})[ \t]*\(", text):
        return YES
    for match in re.finditer(r"(?m)^[ \t]*\{", text):
        rest = text[match.start():]
        if '"name"' in rest:
            return YES
    state = NO
    for marker in MARKERS + tuple(f"<{n}>" for n in names):
        if any(text.endswith(marker[:k]) for k in range(1, len(marker))):
            state = MAYBE
    last_line = text.rsplit("\n", 1)[-1].strip().lstrip("`[").strip()
    if last_line and any(n.startswith(last_line) for n in names):
        state = MAYBE
    brace = [m.start() for m in re.finditer(r"(?m)^[ \t]*\{", text)]
    if brace and len(text) - brace[-1] < 400:
        state = MAYBE
    return state


# ---------- rewriting whole replies ----------

def _hex():
    return secrets.token_hex(12)


def repair_response(api, response, tools):
    """A copy of the JSON reply `response` with a text-written tool call turned into a proper one, as (new reply,
    Repair); None when there is nothing to repair."""
    if not tools or not isinstance(response, dict):
        return None
    if api == "anthropic":
        blocks = response.get("content")
        if not isinstance(blocks, list) or any(b.get("type") == "tool_use" for b in blocks if isinstance(b, dict)):
            return None
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        found = find_tool_call(text, tools)
        if not found:
            return None
        out, placed = [], False
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                if not placed and found.text:
                    out.append({**block, "text": found.text})
                placed = True
            else:
                out.append(block)
        out += [{"type": "tool_use", "id": "toolu_hw_" + _hex(), "name": c["name"], "input": c["arguments"]}
                for c in found.calls]
        return {**response, "content": out, "stop_reason": "tool_use"}, found
    if api == "chat":
        choices = response.get("choices")
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or message.get("tool_calls") or not isinstance(message.get("content"), str):
            return None
        found = find_tool_call(message["content"], tools)
        if not found:
            return None
        message = {**message, "content": found.text or None, "tool_calls": [
            {"id": "call_hw_" + _hex(), "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in found.calls]}
        choice = {**choices[0], "message": message, "finish_reason": "tool_calls"}
        return {**response, "choices": [choice] + choices[1:]}, found
    output = response.get("output")
    if not isinstance(output, list) or any(i.get("type") == "function_call" for i in output if isinstance(i, dict)):
        return None
    text = "".join(p.get("text", "") for i in output if isinstance(i, dict) and i.get("type") == "message"
                   for p in i.get("content") or () if isinstance(p, dict) and p.get("type") == "output_text")
    found = find_tool_call(text, tools)
    if not found:
        return None
    return {**response, "output": responses_output(output, found), "status": "completed"}, found


def _function_call_items(found, status="completed"):
    return [{"type": "function_call", "id": "fc_hw_" + _hex(), "call_id": "call_hw_" + _hex(), "status": status,
             "name": c["name"], "arguments": json.dumps(c["arguments"])} for c in found.calls]


def responses_output(output, found, items=None):
    """`output` (Responses items) with the message text replaced by what is outside the calls, and the calls added."""
    out, placed = [], False
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            if not placed and found.text:
                parts = [{**p, "text": found.text} if isinstance(p, dict) and p.get("type") == "output_text" else p
                         for p in item.get("content") or ()]
                out.append({**item, "content": parts})
            placed = True
        else:
            out.append(item)
    return out + (items if items is not None else _function_call_items(found))


# ---------- streaming ----------

def _sse(api, name, obj):
    body = "data: " + json.dumps(obj, separators=(",", ":")) + "\n\n"
    return (body if api == "chat" else f"event: {name}\n{body}").encode()


_BOUNDARY = re.compile(rb"\r?\n\r?\n")


class StreamRepair:
    """Filter for a streamed reply: `feed(bytes)` returns the bytes to send on, `finish()` the rest.

    Events pass through byte for byte until the assistant text looks like the start of a call; from then on the events
    that carry text (and, for Anthropic and Responses, everything after them) are held back. At the end of the reply,
    valid held text becomes proper tool-call events; anything else is released unchanged. `repaired` is then the
    Repair. A structured tool call from the server switches the filter off."""

    def __init__(self, api, tools):
        self.api, self.tools = api, tools
        self.buffer = b""
        self.text = ""
        self.mode = "pass"      # pass | hold | off
        self.state = NO
        self.queue = []         # [(raw bytes, text delta or None, event object)]
        self.held_from = 0      # length of the text already sent on when holding began
        self.top_index = -1     # highest Anthropic content block index seen
        self.template = {}      # last Chat chunk: id, model, ... for the chunks we make
        self.repaired = None

    # --- input
    def feed(self, data):
        self.buffer += data
        out = []
        while (match := _BOUNDARY.search(self.buffer)):
            raw, self.buffer = self.buffer[:match.end()], self.buffer[match.end():]
            out.append(self._event(raw))
        return b"".join(out)

    def finish(self):
        out = [self._event(self.buffer)] if self.buffer else []
        self.buffer = b""
        out.append(self._flush())
        return b"".join(out)

    def _flush(self):
        out = b"".join(raw for raw, _, _ in self.queue)
        self.queue = []
        if self.mode == "hold":
            self.mode = "pass"
        return out

    # --- one event
    def _parse(self, raw):
        data = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.startswith("data:"):
                data.append(line[5:].strip())
        joined = "\n".join(data)
        try:
            obj = json.loads(joined)
        except ValueError:
            return {"_raw": joined}
        return obj if isinstance(obj, dict) else {"_raw": joined}

    def _classify(self, obj):
        """("structured" | "text" | "final" | "other", text delta)."""
        if "_raw" in obj:
            return ("final" if self.api == "chat" and obj["_raw"] == "[DONE]" else "other"), None
        kind = obj.get("type")
        if self.api == "anthropic":
            if "index" in obj and isinstance(obj["index"], int):
                self.top_index = max(self.top_index, obj["index"])
            if kind == "content_block_start" and (obj.get("content_block") or {}).get("type") == "tool_use":
                return "structured", None
            delta = obj.get("delta") or {}
            if kind == "content_block_delta" and delta.get("type") == "text_delta":
                return "text", delta.get("text") or ""
            return ("final" if kind in ("message_delta", "message_stop") else "other"), None
        if self.api == "responses":
            if kind == "response.output_item.added" and (obj.get("item") or {}).get("type") == "function_call":
                return "structured", None
            if kind == "response.output_text.delta":
                return "text", obj.get("delta") or ""
            return ("final" if kind in ("response.completed", "response.incomplete", "response.failed") else "other"), None
        choice = (obj.get("choices") or [{}])[0] if isinstance(obj.get("choices"), list) else {}
        delta = choice.get("delta") or {}
        self.template = {k: obj[k] for k in ("id", "created", "model", "object", "system_fingerprint") if k in obj}
        if delta.get("tool_calls"):
            return "structured", None
        if delta.get("content"):
            return "text", delta["content"]
        return ("final" if choice.get("finish_reason") else "other"), None

    def _event(self, raw):
        if self.mode == "off":
            return raw
        obj = self._parse(raw)
        kind, delta = self._classify(obj)
        if kind == "structured":
            out = self._flush() + raw
            self.mode = "off"
            return out
        if kind == "text":
            self.text += delta
            if self.mode == "pass":
                self.state = hold_state(self.text, self.tools, window=len(delta) + 500)
                if self.state == NO:
                    return raw
                self.mode, self.held_from = "hold", len(self.text) - len(delta)
            elif self.state != YES:
                self.state = hold_state(self.text, self.tools)
                if self.state == NO:
                    self.queue.append((raw, delta, obj))
                    return self._flush()
            self.queue.append((raw, delta, obj))
            return b""
        if self.mode == "hold":
            if kind == "final":
                return self._finish_reply(raw, obj)
            if self.api == "chat":
                return raw  # reasoning and role chunks do not wait
            self.queue.append((raw, None, obj))
            return b""
        return raw

    # --- the end of the reply: repair or release
    def _finish_reply(self, raw, obj):
        found = find_tool_call(self.text, self.tools) if self._normal_end(obj) else None
        if not found:
            return self._flush() + raw
        queue, self.queue, self.mode = self.queue, [], "off"
        self.repaired = found
        sent = strip_think(self.text[:self.held_from]).strip()
        extra = found.text[len(sent):] if found.text.startswith(sent) else ""
        return getattr(self, "_emit_" + self.api)(found, queue, extra, raw, obj)

    def _normal_end(self, obj):
        """False for a reply cut short (token limit, failure): its text may be an unfinished call."""
        if self.api == "anthropic":
            return (obj.get("delta") or {}).get("stop_reason") in (None, "end_turn", "stop_sequence") or obj.get("type") == "message_stop"
        if self.api == "responses":
            return obj.get("type") == "response.completed"
        if "_raw" in obj:
            return True
        return obj["choices"][0].get("finish_reason") == "stop"

    def _kept(self, queue, extra, make_text_event):
        """The held events without their text deltas; `extra` (text before the call that was not sent yet) takes the
        place of the first of them."""
        out, put = [], not extra
        for raw, delta, obj in queue:
            if delta is None:
                out.append(raw)
            elif not put:
                out.append(make_text_event(obj, extra))
                put = True
        return out

    def _emit_anthropic(self, found, queue, extra, raw, obj):
        def text_event(old, text):
            return _sse("anthropic", "content_block_delta", {**old, "delta": {"type": "text_delta", "text": text}})
        out = self._kept(queue, extra, text_event)
        if obj.get("type") == "message_delta":
            final, end = obj, None
        else:  # message_stop without a message_delta
            final, end = None, raw
        index = self.top_index
        for call in found.calls:
            index += 1
            block = {"type": "tool_use", "id": "toolu_hw_" + _hex(), "name": call["name"], "input": {}}
            out.append(_sse("anthropic", "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}))
            out.append(_sse("anthropic", "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(call["arguments"])}}))
            out.append(_sse("anthropic", "content_block_stop", {"type": "content_block_stop", "index": index}))
        if final is not None:
            delta = {**(final.get("delta") or {}), "stop_reason": "tool_use"}
            out.append(_sse("anthropic", "message_delta", {**final, "delta": delta}))
        else:
            out.append(_sse("anthropic", "message_delta", {"type": "message_delta", "delta": {
                "stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 0}}))
            out.append(end)
        return b"".join(out)

    def _chunk(self, delta, finish=None, base=None):
        return _sse("chat", "", {**self.template, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def _emit_chat(self, found, queue, extra, raw, obj):
        out = [self._chunk({"content": extra})] if extra else []
        if "_raw" in obj:  # [DONE] without a finish chunk
            tail, finish = raw, None
        else:
            finish = obj
        for number, call in enumerate(found.calls):
            out.append(self._chunk({"tool_calls": [{"index": number, "id": "call_hw_" + _hex(), "type": "function",
                                                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}))
        if finish is not None:
            choice = {**finish["choices"][0], "finish_reason": "tool_calls"}
            out.append(_sse("chat", "", {**finish, "choices": [choice] + finish["choices"][1:]}))
        else:
            out.append(self._chunk({}, "tool_calls"))
            out.append(tail)
        return b"".join(out)

    def _emit_responses(self, found, queue, extra, raw, obj):
        text = found.text

        def text_event(old, value):
            return _sse("responses", "response.output_text.delta", {**old, "delta": value})
        out = []
        put = not extra
        for held, delta, event in queue:
            kind = event.get("type")
            if delta is not None:
                if not put:
                    out.append(text_event(event, extra))
                    put = True
            elif kind == "response.output_text.done":
                out.append(_sse("responses", kind, {**event, "text": text}))
            elif kind == "response.content_part.done":
                out.append(_sse("responses", kind, {**event, "part": {**(event.get("part") or {}), "text": text}}))
            elif kind == "response.output_item.done" and (event.get("item") or {}).get("type") == "message":
                parts = [{**p, "text": text} if isinstance(p, dict) and p.get("type") == "output_text" else p
                         for p in event["item"].get("content") or ()]
                out.append(_sse("responses", kind, {**event, "item": {**event["item"], "content": parts}}))
            else:
                out.append(held)
        items = _function_call_items(found)
        for item in items:
            started = {**item, "status": "in_progress", "arguments": ""}
            out.append(_sse("responses", "response.output_item.added", {"type": "response.output_item.added", "item": started}))
            out.append(_sse("responses", "response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta", "delta": item["arguments"], "item_id": item["id"]}))
            out.append(_sse("responses", "response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done", "arguments": item["arguments"], "item_id": item["id"]}))
            out.append(_sse("responses", "response.output_item.done", {"type": "response.output_item.done", "item": item}))
        if obj.get("type") == "response.completed":
            response = obj.get("response") or {}
            response = {**response, "status": "completed",
                        "output": responses_output(response.get("output") or [], found, items)}
            out.append(_sse("responses", "response.completed", {**obj, "response": response}))
        else:
            out.append(raw)
        return b"".join(out)
