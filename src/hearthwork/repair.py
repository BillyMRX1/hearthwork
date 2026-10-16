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
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
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


MAX_LEAD = 300    # characters of prose allowed before a call
MAX_HOLD = 4096   # characters a stream holds back while waiting to see whether a call follows
_CLOSERS = re.compile(r"(?:\s|</tool_call>|</function>|</function_calls>|</tool_calls>|</invoke>|<\|im_end\|>|<\|eot_id\|>|</s>|\[/TOOL_CALLS\])*\Z")
_BETWEEN = re.compile(r"(?:\s|,|</?(?:tool_call|tool_calls|function_calls|function)>)*\Z")


def lead_in_ok(lead):
    lead = _LEFTOVER.sub("", lead)
    return "```" not in lead and len(lead.strip()) <= MAX_LEAD


def find_tool_call(text, tools):
    """Repair(calls, text, format) when `text` is (a short lead-in and) valid tool calls written as plain text, else None.
    `tools` is {name: schema} (see request_tools). Deliberately conservative: the calls must end the reply (only closing
    markers may follow), the lead-in is short and has no code fence, nothing inside a ``` fence counts, and every call
    must be valid, or the reply stays as it is."""
    if not tools or not isinstance(text, str) or not text.strip():
        return None
    text = strip_think(text)
    if "```" in text:
        return None
    try:
        for fmt, detect in _DETECTORS:
            found = detect(text, tools)
            if not found:
                continue
            raw, spans = found
            spans = sorted(spans)
            calls = []
            for name, arguments in raw:
                if not valid_call(name, arguments, tools):
                    return None
                json.dumps(arguments, allow_nan=False)
                calls.append({"name": name, "arguments": arguments})
            if not lead_in_ok(text[:spans[0][0]]) or not _CLOSERS.match(text, spans[-1][1]):
                return None
            if any(not _BETWEEN.match(text, a[1], b[0]) or a[1] > b[0] for a, b in zip(spans, spans[1:])):
                return None
            return Repair(calls, _LEFTOVER.sub("", text[:spans[0][0]]).strip(), fmt)
    except Exception:  # anything odd in the text: leave the reply alone
        return None
    return None


# ---------- streaming: could the text so far still turn out to be a trailing call? ----------

MARKERS = ("[TOOL_CALLS]", "[Calling tool:", "<tool_call>", "<function=", "<invoke ")
_CALL_KEYS = ("name", "function", "type", "arguments", "parameters", "input", "args")
_FIRST_NAME = re.compile(r'(?:^\{|<tool_call>\s*\{|\[TOOL_CALLS\]\s*\[?\s*\{)\s*"name"\s*:\s*"([^"]*)"')
_NAME_CHECKS = (re.compile(r"<function=([^>\s]+)>"), re.compile(r"\[Calling tool:\s*([\w.\-]+)\s*\("),
                re.compile(r"<invoke\s+name=[\"']([^\"']+)[\"']"), re.compile(r"<name>([^<]*)</name>"),
                re.compile(r"\[TOOL_CALLS\]\s*([\w.\-]+)\s*\[ARGS\]"))


def _candidate(s, names):
    """0: `s` does not begin like a call; 1: it does; 2: it is too short to say yet (the start of a marker or name)."""
    if not s:
        return 0
    if s[0] == "{" or any(s.startswith(m) for m in MARKERS) or any(s.startswith(f"<{n}>") for n in names):
        return 1
    if names and re.match(rf"`*\[?\s*(?:{_names(names)})\s*\(", s):
        return 1
    if any(m.startswith(s) for m in MARKERS + tuple(f"<{n}>" for n in names)):
        return 2
    word = s.strip("`[ \t")
    if re.fullmatch(r"`*\[?\s*\w*\s*", s) and word and any(n.startswith(word) for n in names):
        return 2
    return 0


def _unclosed(s):
    depth, quote, escape = 0, False, False
    for c in s:
        if quote:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                quote = False
        elif c == '"':
            quote = True
        elif c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
    if quote or depth > 0:
        return True
    return any(s.count(a) > s.count(b) for a, b in (("<tool_call>", "</tool_call>"), ("<function=", "</function>"),
                                                    ("<invoke ", "</invoke>")))


def can_be_call(t, start, tools):
    """True while `t[start:]` (the held text; `t` is the whole reply so far) could still end as a valid trailing call."""
    tail = t[start:]
    if len(tail) > MAX_HOLD or "```" in t:
        return False
    if find_tool_call(t, tools):
        return True
    s = tail.lstrip()
    kind = _candidate(s, tools)
    if not kind:
        return False
    if kind == 2:
        return True
    named = _FIRST_NAME.search(s)
    if named and named.group(1) not in tools:
        return False
    if any((m := rx.search(s)) and m.group(1) not in tools for rx in _NAME_CHECKS):
        return False
    key = re.match(r'\{\s*"([^"]*)"', s)
    if key and key.group(1) not in _CALL_KEYS:
        return False
    if s.startswith("[TOOL_CALLS]") and "{" not in s:
        return True
    return _unclosed(s)


def find_hold_start(t, scan_from, tools):
    """Index in `t` of a line start where a trailing call may begin, among the lines from the one holding `scan_from`
    on; None when there is none (the normal case)."""
    if "```" in t:
        return None
    pos = t.rfind("\n", 0, scan_from) + 1
    while pos <= len(t):
        q = pos
        while q < len(t) and t[q] in " \t":
            q += 1
        if q >= scan_from and _candidate(t[q:], tools) and lead_in_ok(t[:q]) and can_be_call(t, q, tools):
            return q
        end = t.find("\n", pos)
        if end < 0:
            return None
        pos = end + 1
    return None


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


def responses_output(output, found, items=None, keep=None):
    """`output` (Responses items) with the message text replaced by what is outside the calls, and the calls added."""
    out, placed = [], False
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            if not placed and (found.text if keep is None else keep):
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

    Events pass through byte for byte. Text is held back only from a line start where the rest of the reply could still be
    a trailing call (see `can_be_call`), and released as soon as it cannot; at the end of the reply valid held text
    becomes proper tool-call events. `repaired` is then the Repair. A structured tool call from the server, or any
    internal error, switches the filter off (everything held is sent as it was)."""

    def __init__(self, api, tools):
        self.api, self.tools = api, tools
        self.buffer = b""
        self.text = ""
        self.mode = "pass"      # pass | hold | off
        self.queue = []         # held [(raw bytes, text delta or None, event object)]
        self.pending = []       # events that open a text container, waiting for its first text
        self.hold_pos = 0       # where the held text starts, in the visible text
        self.held_from = 0      # length of the text already sent on when holding began
        self.no_hold_before = 0  # text before this was released: no new hold starts there
        self.top_index = -1     # highest Anthropic content block index seen
        self.next_output = 0    # Responses: index of the next output item
        self.seq = None         # Responses: last sequence_number seen
        self.template = {}      # last Chat chunk: id, model, ... for the chunks we make
        self.repaired = None

    # --- input
    def feed(self, data):
        self.buffer += data
        out = []
        while (match := _BOUNDARY.search(self.buffer)):
            raw, self.buffer = self.buffer[:match.end()], self.buffer[match.end():]
            out.append(self._safe(raw))
        return b"".join(out)

    def finish(self):
        out = [self._safe(self.buffer)] if self.buffer else []
        self.buffer = b""
        out.append(self._flush())
        return b"".join(out)

    def _safe(self, raw):
        try:
            return self._event(raw)
        except Exception:  # never lose a reply over a repair: send what we hold and stop filtering
            self.mode = "off"
            out = b"".join(r for r, _, _ in self.pending + self.queue) + raw
            self.pending, self.queue = [], []
            return out

    def _flush(self):
        out = b"".join(raw for raw, _, _ in self.pending + self.queue)
        self.pending, self.queue = [], []
        if self.mode == "hold":
            self.mode = "pass"
        return out

    def _visible(self):
        return strip_think(self.text) if "think>" in self.text else self.text

    # --- one event
    def _parse(self, raw):
        data = [line[5:].strip() for line in raw.decode("utf-8", "replace").splitlines() if line.startswith("data:")]
        joined = "\n".join(data)
        try:
            obj = json.loads(joined)
        except ValueError:
            return {"_raw": joined}
        return obj if isinstance(obj, dict) else {"_raw": joined}

    def _classify(self, obj):
        """("structured" | "container" | "text" | "final" | "other", text delta)."""
        if "_raw" in obj:
            return ("final" if self.api == "chat" and obj["_raw"] == "[DONE]" else "other"), None
        kind = obj.get("type")
        if self.api == "anthropic":
            if isinstance(obj.get("index"), int):
                self.top_index = max(self.top_index, obj["index"])
            block = obj.get("content_block") or {}
            if kind == "content_block_start":
                return ("structured" if block.get("type") == "tool_use" else
                        "container" if block.get("type") == "text" else "other"), None
            delta = obj.get("delta") or {}
            if kind == "content_block_delta" and delta.get("type") == "text_delta":
                return "text", delta.get("text") or ""
            return ("final" if kind in ("message_delta", "message_stop") else "other"), None
        if self.api == "responses":
            if isinstance(obj.get("sequence_number"), int):
                self.seq = max(self.seq or 0, obj["sequence_number"])
            item = obj.get("item") or {}
            if kind == "response.output_item.added":
                self.next_output = max(self.next_output, obj["output_index"] + 1) if isinstance(
                    obj.get("output_index"), int) else self.next_output + 1
                if item.get("type") == "function_call":
                    return "structured", None
                return ("container" if item.get("type") == "message" else "other"), None
            if kind == "response.content_part.added":
                return "container", None
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

    def _take_pending(self):
        out = b"".join(raw for raw, _, _ in self.pending)
        self.pending = []
        return out

    def _event(self, raw):
        if self.mode == "off":
            return raw
        obj = self._parse(raw)
        before = len(self._visible())
        kind, delta = self._classify(obj)
        if kind == "structured":
            out = self._flush() + raw
            self.mode = "off"
            return out
        if kind == "container":
            (self.pending if self.mode == "pass" else self.queue).append((raw, None, obj))
            return b""
        if kind == "text":
            self.text += delta
            t = self._visible()
            if self.mode == "pass":
                start = find_hold_start(t, max(before, self.no_hold_before), self.tools)
                if start is None:
                    return self._take_pending() + raw
                self.mode, self.hold_pos, self.held_from = "hold", start, len(self.text) - len(delta)
                self.queue, self.pending = self.pending + [(raw, delta, obj)], []
                return b""
            self.queue.append((raw, delta, obj))
            if len(t) < self.hold_pos or not can_be_call(t, self.hold_pos, self.tools):
                self.no_hold_before = len(t)
                return self._flush()
            return b""
        if self.mode == "hold":
            if kind == "final":
                return self._finish_reply(raw, obj)
            if self.api == "chat":
                return raw  # reasoning and role chunks do not wait
            self.queue.append((raw, None, obj))
            return b""
        return self._take_pending() + raw

    # --- the end of the reply: repair or release
    def _finish_reply(self, raw, obj):
        held = b"".join(r for r, _, _ in self.queue)
        queue, self.queue = self.queue, []
        try:
            found = find_tool_call(self.text, self.tools) if self._normal_end(obj) else None
            if not found:
                self.mode = "pass"
                return held + raw
            sent = strip_think(self.text[:self.held_from]).strip()
            extra = found.text[len(sent):] if found.text.startswith(sent) else ""
            out = getattr(self, "_emit_" + self.api)(found, queue, extra, raw, obj)
        except Exception:
            self.mode = "off"
            return held + raw
        self.mode, self.repaired = "off", found
        return out

    def _normal_end(self, obj):
        """False for a reply cut short (token limit, failure): its text may be an unfinished call."""
        if self.api == "anthropic":
            return (obj.get("delta") or {}).get("stop_reason") in (None, "end_turn", "stop_sequence") or obj.get("type") == "message_stop"
        if self.api == "responses":
            return obj.get("type") == "response.completed"
        if "_raw" in obj:
            return True
        return obj["choices"][0].get("finish_reason") == "stop"

    def _emit_anthropic(self, found, queue, extra, raw, obj):
        def text_event(old, text):
            return _sse("anthropic", "content_block_delta", {**old, "delta": {"type": "text_delta", "text": text}})
        # a text block that would end up empty is dropped altogether (its start was held back for this)
        starts = [ev for _, _, ev in queue if ev.get("type") == "content_block_start"
                  and (ev.get("content_block") or {}).get("type") == "text"]
        dropped = starts[0]["index"] if starts and not found.text else None
        out, put = [], not extra
        for held, delta, ev in queue:
            if delta is not None:
                if not put:
                    out.append(text_event(ev, extra))
                    put = True
            elif dropped is not None and ev.get("index") == dropped and ev.get("type") in ("content_block_start", "content_block_stop"):
                continue
            else:
                out.append(held)
        index = (dropped if dropped is not None else self.top_index + 1) - 1
        for call in found.calls:
            index += 1
            block = {"type": "tool_use", "id": "toolu_hw_" + _hex(), "name": call["name"], "input": {}}
            out.append(_sse("anthropic", "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}))
            out.append(_sse("anthropic", "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(call["arguments"])}}))
            out.append(_sse("anthropic", "content_block_stop", {"type": "content_block_stop", "index": index}))
        if obj.get("type") == "message_delta":
            delta = {**(obj.get("delta") or {}), "stop_reason": "tool_use"}
            out.append(_sse("anthropic", "message_delta", {**obj, "delta": delta}))
        else:  # message_stop without a message_delta
            out.append(_sse("anthropic", "message_delta", {"type": "message_delta", "delta": {
                "stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 0}}))
            out.append(raw)
        return b"".join(out)

    def _chunk(self, delta, finish=None):
        return _sse("chat", "", {**self.template, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def _emit_chat(self, found, queue, extra, raw, obj):
        out = [self._chunk({"content": extra})] if extra else []
        for number, call in enumerate(found.calls):
            out.append(self._chunk({"tool_calls": [{"index": number, "id": "call_hw_" + _hex(), "type": "function",
                                                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}))
        if "_raw" in obj:  # [DONE] without a finish chunk
            out.append(self._chunk({}, "tool_calls"))
            out.append(raw)
        else:
            choice = {**obj["choices"][0], "finish_reason": "tool_calls"}
            out.append(_sse("chat", "", {**obj, "choices": [choice] + obj["choices"][1:]}))
        return b"".join(out)

    def _emit_responses(self, found, queue, extra, raw, obj):
        text = found.text
        message_events = ("response.content_part.added", "response.output_text.delta", "response.output_text.done",
                          "response.content_part.done")
        # the message item would end up empty: drop all of it (its start was held back for this)
        dropped = not text and any(ev.get("type") == "response.output_item.added" and (ev.get("item") or {}).get("type") == "message"
                                   for _, _, ev in queue)

        def number(event):
            if self.seq is not None:
                self.seq += 1
                event["sequence_number"] = self.seq
            return event

        out, put = [], not extra
        for held, delta, event in queue:
            kind = event.get("type")
            is_message = (event.get("item") or {}).get("type") == "message"
            if dropped and (kind in message_events or (kind in ("response.output_item.added", "response.output_item.done") and is_message)):
                continue
            if delta is not None:
                if not put:
                    out.append(_sse("responses", kind, {**event, "delta": extra}))
                    put = True
            elif kind == "response.output_text.done":
                out.append(_sse("responses", kind, {**event, "text": text}))
            elif kind == "response.content_part.done":
                out.append(_sse("responses", kind, {**event, "part": {**(event.get("part") or {}), "text": text}}))
            elif kind == "response.output_item.done" and is_message:
                parts = [{**p, "text": text} if isinstance(p, dict) and p.get("type") == "output_text" else p
                         for p in event["item"].get("content") or ()]
                out.append(_sse("responses", kind, {**event, "item": {**event["item"], "content": parts}}))
            else:
                out.append(held)
        items = _function_call_items(found)
        base = self.next_output - (1 if dropped else 0)
        for n, item in enumerate(items):
            index = {"output_index": base + n}
            for kind, fields in (
                    ("response.output_item.added", {"item": {**item, "status": "in_progress", "arguments": ""}}),
                    ("response.function_call_arguments.delta", {"delta": item["arguments"], "item_id": item["id"]}),
                    ("response.function_call_arguments.done", {"arguments": item["arguments"], "item_id": item["id"]}),
                    ("response.output_item.done", {"item": item})):
                out.append(_sse("responses", kind, number({"type": kind, **index, **fields})))
        if obj.get("type") == "response.completed":
            response = obj.get("response") or {}
            response = {**response, "status": "completed",
                        "output": responses_output(response.get("output") or [], found, items, keep=not dropped)}
            out.append(_sse("responses", "response.completed", number({**obj, "response": response})))
        else:
            out.append(raw)
        return b"".join(out)
