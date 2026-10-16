import http.client
import http.server
import json
import threading
import unittest

from hearthwork import repair
from hearthwork.harnesses import make_relay

TOOLS = {
    "get_secret": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "read_file": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"},
                                                   "deep": {"type": "boolean"}}, "required": ["path"]},
    "noargs": {"type": "object", "properties": {}},
}


def one(text, tools=TOOLS):
    found = repair.find_tool_call(text, tools)
    assert found, text
    return found


class Formats(unittest.TestCase):
    def check(self, text, fmt, name="get_secret", arguments=None, rest=""):
        found = one(text)
        self.assertEqual(found.format, fmt)
        self.assertEqual(found.calls, [{"name": name, "arguments": {"name": "alpha"} if arguments is None else arguments}])
        self.assertEqual(found.text, rest)

    def test_mistral(self):
        self.check('[TOOL_CALLS] [{"name": "get_secret", "arguments": {"name": "alpha"}}]', "mistral")
        self.check('Sure.[TOOL_CALLS][{"name": "get_secret", "arguments": {"name": "alpha"}}]', "mistral", rest="Sure.")
        self.check('[TOOL_CALLS]get_secret[ARGS]{"name": "alpha"}', "mistral")

    def test_calling_tool(self):
        self.check('[Calling tool: get_secret({"name": "alpha"})]', "calling-tool")
        self.check("[Calling tool: noargs()]", "calling-tool", "noargs", {})

    def test_tool_call_json(self):
        self.check('<tool_call>\n{"name": "get_secret", "arguments": {"name": "alpha"}}\n</tool_call>', "tool_call")
        self.check('ok <tool_call>{"name": "get_secret", "arguments": {"name": "alpha"}}', "tool_call", rest="ok")

    def test_tool_call_wrapping_function(self):
        found = one("<tool_call>\n<function=get_secret>\n<parameter=name>\nalpha\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual(found.calls[0]["arguments"], {"name": "alpha"})
        self.assertEqual(found.text, "")

    def test_function_parameters_are_typed(self):
        found = one("<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=limit>\n20\n</parameter>\n"
                    "<parameter=deep>\ntrue\n</parameter>\n</function>")
        self.assertEqual(found.format, "function")
        self.assertEqual(found.calls[0]["arguments"], {"path": "a.py", "limit": 20, "deep": True})

    def test_function_json(self):
        self.check('<function=get_secret>{"name": "alpha"}</function>', "function")

    def test_xml_tags(self):
        self.check("<get_secret><name>alpha</name></get_secret>", "xml")
        self.check('<invoke name="get_secret"><parameter name="name">alpha</parameter></invoke>', "xml")
        self.check('<get_secret>{"name": "alpha"}</get_secret>', "xml")

    def test_pythonic(self):
        self.check('get_secret(name="alpha")', "pythonic")
        self.check('`get_secret(name="alpha")`', "pythonic")
        found = one("[read_file(path='a.py', limit=3), get_secret(name=\"x)\")]")
        self.assertEqual([c["name"] for c in found.calls], ["read_file", "get_secret"])
        self.assertEqual(found.calls[1]["arguments"], {"name": "x)"})

    def test_raw_json(self):
        self.check('{"name": "get_secret", "arguments": {"name": "alpha"}}', "json")
        self.check('  {"name": "get_secret", "parameters": {"name": "alpha"}}\n', "json")
        self.check('{"name": "get_secret", "arguments": "{\\"name\\": \\"alpha\\"}"}', "json")
        self.check('{"type": "function", "function": {"name": "get_secret", "arguments": {"name": "alpha"}}}', "json")

    def test_multiple_calls(self):
        text = ('<tool_call>{"name": "get_secret", "arguments": {"name": "a"}}</tool_call>\n'
                '<tool_call>{"name": "read_file", "arguments": {"path": "b"}}</tool_call>')
        self.assertEqual([c["name"] for c in one(text).calls], ["get_secret", "read_file"])
        text = '{"name": "get_secret", "arguments": {"name": "a"}}\n{"name": "noargs", "arguments": {}}'
        self.assertEqual(len(one(text).calls), 2)

    def test_think_is_stripped(self):
        self.check('<think>I could write <tool_call>{"name": "nope"}</tool_call></think>\nget_secret(name="alpha")', "pythonic")
        self.check('reasoning...</think>\nget_secret(name="alpha")', "pythonic")
        self.assertIsNone(repair.find_tool_call('<think>get_secret(name="alpha")', TOOLS))


class Rejected(unittest.TestCase):
    def none(self, text, tools=TOOLS):
        self.assertIsNone(repair.find_tool_call(text, tools), text)

    def test_no_tools(self):
        self.none('get_secret(name="alpha")', {})
        self.none('<tool_call>{"name": "get_secret", "arguments": {"name": "a"}}</tool_call>', {})

    def test_unknown_tool(self):
        self.none('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>')
        self.none('[TOOL_CALLS] [{"name": "rm_rf", "arguments": {}}]')
        self.none('<function=rm_rf>{}</function>')
        self.none('{"name": "rm_rf", "arguments": {}}')

    def test_schema_violations(self):
        self.none('<tool_call>{"name": "get_secret", "arguments": {}}</tool_call>')  # required key missing
        self.none('<tool_call>{"name": "get_secret", "arguments": {"name": 5}}</tool_call>')
        self.none('<tool_call>{"name": "get_secret", "arguments": [1]}</tool_call>')
        self.none('<tool_call>{"name": "read_file", "arguments": {"path": "a", "limit": true}}</tool_call>')
        self.none('<tool_call>{"name": "get_secret", "arguments": "not json"}</tool_call>')
        self.none('<tool_call>{"name": "get_secret", "arguments": {"name": "a"}}</tool_call><tool_call>{"name": "x"}</tool_call>')
        self.none('get_secret("alpha")')  # positional arguments

    def test_ordinary_text(self):
        for text in ("Here is the answer: use a function to read the file.",
                     'The config is {"name": "bob", "age": 3} as you see.',
                     "def get_secret(name):\n    return name\n",
                     "Call get_secret like this: it takes a name.",
                     "if (x) { return 1; }\nfunction foo() { return {a: 1}; }",
                     '{"error": "no"}',
                     "[1, 2, 3] and print(x)",
                     ""):
            self.none(text)

    def test_unsupported_schema_features_are_ignored(self):
        tools = {"t": {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}, "b": {"type": ["integer", "null"]}},
                       "required": ["a"], "additionalProperties": False}}
        found = repair.find_tool_call('t(a="x", b=None, c=1)', tools)
        self.assertEqual(found.calls[0]["arguments"], {"a": "x", "b": None, "c": 1})


class Conservative(unittest.TestCase):
    BASH = {"bash": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
    CALL = '{"name": "bash", "arguments": {"command": "rm -rf build"}}'

    def none(self, text, tools=None):
        self.assertIsNone(repair.find_tool_call(text, tools or self.BASH), text)

    def test_call_shaped_examples_in_prose_are_not_repaired(self):
        self.none("The model emits:\n```json\n" + self.CALL + "\n```\nThat is the format.")
        self.none("The model emits: " + self.CALL + " That is the format.")
        self.none(self.CALL + "\nThat is the format.")
        self.none("Output:\n```\n" + self.CALL + "\n```")
        self.none("<tool_call>" + self.CALL + "</tool_call> is how Qwen writes it, and Mistral writes it differently.")
        self.none("x" * 400 + "\n<tool_call>" + self.CALL + "</tool_call>")

    def test_bare_formats_allow_no_lead_in(self):
        self.none('Example: {"name": "bash", "arguments": {"command": "ls"}}')
        self.none('I will look it up.\nget_secret(name="alpha")', TOOLS)
        self.none('Here you go: {"name": "get_secret", "arguments": {"name": "alpha"}}', TOOLS)

    def test_one_fence_around_the_trailing_call(self):
        for text, fmt in (('```json\n{"name": "get_secret", "arguments": {"name": "alpha"}}\n```', "json"),
                          ('```\nget_secret(name="alpha")\n```\n', "pythonic"),
                          ('Checking.\n```\n<tool_call>{"name": "get_secret", "arguments": {"name": "alpha"}}</tool_call>\n```', "tool_call")):
            found = repair.find_tool_call(text, TOOLS)
            self.assertEqual((found.format, found.calls[0]["arguments"]), (fmt, {"name": "alpha"}))
        self.assertIsNone(repair.find_tool_call('```\nx\n```\n```json\n{"name": "get_secret", "arguments": {"name": "a"}}\n```', TOOLS))
        self.assertIsNone(repair.find_tool_call('```json\n{"name": "get_secret", "arguments": {"name": "a"}}\n```\nDone.', TOOLS))
        self.assertIsNone(repair.find_tool_call('Look: ```json\n{"name": "get_secret", "arguments": {"name": "a"}}\n```', TOOLS))

    def test_fenced_call_streams_are_repaired(self):
        tools = repair.request_tools("chat", {"tools": REQUEST_TOOLS["chat"]})
        text = '```json\n{"name": "get_secret", "arguments": {"name": "alpha"}}\n```'
        stream = repair.StreamRepair("chat", tools)
        raw = STREAMS["chat"](text)
        out = b"".join(stream.feed(e + b"\n\n") for e in raw.split(b"\n\n") if e) + stream.finish()
        self.assertTrue(stream.repaired)
        self.assertIn(b"tool_calls", out)

    def test_a_short_lead_in_and_closing_markers_are_fine(self):
        found = repair.find_tool_call("Cleaning up.\n<tool_call>" + self.CALL + "</tool_call>\n", self.BASH)
        self.assertEqual((found.format, found.text), ("tool_call", "Cleaning up."))

    def test_odd_values_do_not_raise(self):
        tools = {"t": {"type": "object", "properties": {}}}
        for text in ("t(a={[]: 1})", "t(a={1, 2})", "t(a=b'x')", "t(a=1j)", "t(a=" + "[" * 5000 + ")",
                     '{"name": "t", "arguments": {"a": NaN}}'):
            self.assertIsNone(repair.find_tool_call(text, tools), text)

    def test_hold_start(self):
        names = TOOLS
        for text in ("Hello there, I will help.", "Use <b>bold</b> text", "mention [TOOL_CALLS] in prose",
                     "x = read(fd, buf, n);", '{"name": "bob"}'):
            self.assertIsNone(repair.find_hold_start(text, 0, names), text)
        self.assertEqual(repair.find_hold_start("text\n[TOOL_CALLS]", 0, names), 5)
        self.assertEqual(repair.find_hold_start("text\n[TOOL", 0, names), 5)
        self.assertIsNone(repair.find_hold_start("a\nget_secret(", 0, names))  # raw/pythonic only at the very start
        self.assertIsNone(repair.find_hold_start('a\n{"name": "get_secret", "arg', 0, names))
        self.assertEqual(repair.find_hold_start("get_secret(", 0, names), 0)
        self.assertEqual(repair.find_hold_start('```json\n{"name": "get_secret", "arg', 0, names), 8)
        self.assertEqual(repair.find_hold_start("a\n<tool_call>", 0, names), 2)
        self.assertIsNone(repair.find_hold_start("x" * 400 + "\n[TOOL_CALLS]", 0, names))  # lead-in too long

    def test_can_be_call_releases(self):
        names = TOOLS
        text = '{"name": "get_secret", "arguments": {"name": "a"}}'
        self.assertTrue(repair.can_be_call(text, 0, names))
        self.assertFalse(repair.can_be_call(text + "\nand more prose", 0, names))
        self.assertFalse(repair.can_be_call('{"name": "my-app", "version": "1"', 0, names))
        self.assertFalse(repair.can_be_call("get_secret(1, 2)", 0, names))
        self.assertFalse(repair.can_be_call("[TOOL_CALLS]" + "x" * 5000, 0, names))


def event_stream(api, text):
    return STREAMS[api](text)


class StreamTiming(unittest.TestCase):
    def feed_all(self, api, text, tools=None):
        """(bytes sent before the stream ended, bytes at the end, the filter)."""
        stream = repair.StreamRepair(api, repair.request_tools(api, {"tools": REQUEST_TOOLS[api]}) if tools is None else tools)
        raw = STREAMS[api](text)
        early = b""
        # feed everything but the last event(s), as the relay does while the model is still writing
        events = raw.split(b"\n\n")
        cut = len(events) - {"anthropic": 4, "chat": 3, "responses": 2}[api]
        for event in events[:cut]:
            early += stream.feed(event + b"\n\n")
        late = b"".join(stream.feed(e + b"\n\n") for e in events[cut:] if e) + stream.finish()
        return raw, early, late, stream

    def test_package_json_is_released_while_streaming(self):
        text = 'Here is package.json:\n{\n  "name": "my-app",\n  "version": "1.0.0"\n}\nRun npm install now.'
        for api in PATHS:
            with self.subTest(api):
                raw, early, late, stream = self.feed_all(api, text)
                self.assertEqual(early + late, raw)
                self.assertIn(b"rsion", early)  # not held until the end
                self.assertIsNone(stream.repaired)

    def test_function_call_in_code_is_released(self):
        text = "Example:\nread(fd, buf, n);\nThat reads n bytes."
        tools = {"read": {"type": "object", "properties": {}}}
        for api in PATHS:
            with self.subTest(api):
                raw, early, late, stream = self.feed_all(api, text, tools)
                self.assertEqual(early + late, raw)
                self.assertIn(b"hat rea", early)

    def test_prose_example_with_a_call_is_not_repaired_in_a_stream(self):
        text = 'The model emits:\n<tool_call>{"name": "get_secret", "arguments": {"name": "a"}}</tool_call>\nThat is the format.'
        for api in PATHS:
            with self.subTest(api):
                raw, early, late, stream = self.feed_all(api, text)
                self.assertEqual(early + late, raw)
                self.assertIsNone(stream.repaired)
                self.assertIn(b"rmat.", early)

    def test_marker_mentioned_in_prose_is_not_held(self):
        text = "Mistral uses [TOOL_CALLS] to start a call, and Qwen uses <tool_call> tags. Both are fine."
        for api in PATHS:
            with self.subTest(api):
                raw, early, late, stream = self.feed_all(api, text)
                self.assertEqual(early + late, raw)
                self.assertIn(b"th are", early)


class StreamShapes(unittest.TestCase):
    def run_stream(self, api, text, mutate=None):
        raw = STREAMS[api](text)
        stream = repair.StreamRepair(api, repair.request_tools(api, {"tools": REQUEST_TOOLS[api]}))
        out = b"".join(stream.feed(e + b"\n\n") for e in raw.split(b"\n\n") if e) + stream.finish()
        return parse_events(out), stream

    def test_anthropic_call_only_reply_has_no_empty_text_block(self):
        events, stream = self.run_stream("anthropic", CALL_TEXT)
        self.assertTrue(stream.repaired)
        starts = [(e["index"], e["content_block"]["type"]) for e in events if e["type"] == "content_block_start"]
        self.assertEqual(starts, [(0, "thinking"), (1, "tool_use")])
        stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
        self.assertEqual(stops, [0, 1])
        self.assertFalse(any(e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta" for e in events))

    def test_responses_call_only_reply_has_no_message_item(self):
        events, stream = self.run_stream("responses", CALL_TEXT)
        self.assertTrue(stream.repaired)
        kinds = [e["type"] for e in events]
        self.assertFalse(any(k.startswith("response.output_text") or k == "response.content_part.added" for k in kinds))
        items = [e["item"]["type"] for e in events if e["type"] == "response.output_item.added"]
        self.assertEqual(items, ["reasoning", "function_call"])
        self.assertEqual([i["type"] for i in events[-1]["response"]["output"]], ["function_call"])

    def test_responses_events_carry_output_index_and_sequence_numbers(self):
        raw = responses_stream("Looking.\n" + CALL_TEXT)
        numbered, number = [], 0
        for block in raw.split(b"\n\n"):
            if not block:
                continue
            number += 1
            line = block.split(b"\n")[-1]
            event = json.loads(line[5:])
            event["sequence_number"] = number
            if event["type"].startswith("response.output_item"):
                event["output_index"] = 0 if event["item"]["type"] == "reasoning" else 1
            numbered.append(event)
        data = b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in numbered)
        stream = repair.StreamRepair("responses", repair.request_tools("responses", {"tools": REQUEST_TOOLS["responses"]}))
        events = parse_events(stream.feed(data) + stream.finish())
        calls = [e for e in events if "function_call" in json.dumps(e.get("item", "")) or e["type"].startswith("response.function_call_arguments")]
        self.assertTrue(calls)
        self.assertEqual({e["output_index"] for e in calls}, {2})
        sequence = [e["sequence_number"] for e in events]
        self.assertEqual(len(sequence), len(set(sequence)))
        self.assertEqual(sequence, sorted(sequence))
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_responses_events_get_output_index_without_upstream_numbers(self):
        events, _ = self.run_stream("responses", "Looking.\n" + CALL_TEXT)
        added = [e for e in events if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"]
        self.assertEqual([e["output_index"] for e in added], [2])
        for kind in ("response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done"):
            self.assertTrue(any(e["type"] == kind and e.get("output_index") == 2 for e in events), kind)

    def test_errors_never_lose_the_reply(self):
        from unittest import mock
        for api in PATHS:
            with self.subTest(api):
                raw = STREAMS[api]("Looking.\n" + CALL_TEXT)
                stream = repair.StreamRepair(api, repair.request_tools(api, {"tools": REQUEST_TOOLS[api]}))
                with mock.patch.object(repair, "find_tool_call", side_effect=RuntimeError("boom")):
                    out = b"".join(stream.feed(e + b"\n\n") for e in raw.split(b"\n\n") if e) + stream.finish()
                self.assertEqual(out, raw)
        from hearthwork import harnesses
        with mock.patch.object(repair, "repair_response", side_effect=TypeError("boom")):
            self.assertIsNone(harnesses.repaired_reply("chat", json.dumps(whole("chat", CALL_TEXT)).encode(), TOOLS))


# ---------- the relay, against a fake llama-server ----------

CALL_TEXT = '<tool_call>{"name": "get_secret", "arguments": {"name": "alpha"}}</tool_call>'


def sse(*events):
    return b"".join((f"event: {n}\n" if n else "").encode() + b"data: " + json.dumps(d).encode() + b"\n\n" if d != "[DONE]"
                    else b"data: [DONE]\n\n" for n, d in events)


def pieces(text, size=7):
    return [text[i:i + size] for i in range(0, len(text), size)]


def anthropic_stream(text=None, structured=False):
    ev = [("message_start", {"type": "message_start", "message": {"id": "m1", "role": "assistant", "content": []}}),
          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
          ("content_block_stop", {"type": "content_block_stop", "index": 0})]
    if structured:
        ev += [("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t", "name": "get_secret", "input": {}}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
               ("content_block_stop", {"type": "content_block_stop", "index": 1})]
        stop = "tool_use"
    else:
        ev.append(("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}))
        ev += [("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": p}}) for p in pieces(text)]
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": 1}))
        stop = "end_turn"
    ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 5}}),
           ("message_stop", {"type": "message_stop"})]
    return sse(*ev)


def chat_chunk(delta, finish=None):
    return (None, {"id": "c1", "model": "m", "object": "chat.completion.chunk", "created": 1,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def chat_stream(text=None, structured=False):
    ev = [chat_chunk({"role": "assistant", "content": None}), chat_chunk({"reasoning_content": "hmm"})]
    if structured:
        ev.append(chat_chunk({"tool_calls": [{"index": 0, "id": "x", "type": "function", "function": {"name": "get_secret", "arguments": "{}"}}]}))
        ev.append(chat_chunk({}, "tool_calls"))
    else:
        ev += [chat_chunk({"content": p}) for p in pieces(text)]
        ev.append(chat_chunk({}, "stop"))
    return sse(*ev, (None, "[DONE]"))


def responses_stream(text=None, structured=False):
    ev = [("response.created", {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}),
          ("response.output_item.added", {"type": "response.output_item.added", "item": {"id": "rs", "type": "reasoning", "summary": []}}),
          ("response.output_item.done", {"type": "response.output_item.done", "item": {"id": "rs", "type": "reasoning", "summary": []}})]
    if structured:
        item = {"id": "fc1", "type": "function_call", "call_id": "c1", "name": "get_secret", "arguments": "{}", "status": "completed"}
        ev += [("response.output_item.added", {"type": "response.output_item.added", "item": {**item, "arguments": ""}}),
               ("response.output_item.done", {"type": "response.output_item.done", "item": item})]
        output = [item]
    else:
        msg = {"id": "msg1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text, "annotations": []}]}
        ev += [("response.output_item.added", {"type": "response.output_item.added", "item": {"id": "msg1", "type": "message", "role": "assistant", "content": []}}),
               ("response.content_part.added", {"type": "response.content_part.added", "item_id": "msg1", "part": {"type": "output_text", "text": ""}})]
        ev += [("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg1", "delta": p}) for p in pieces(text)]
        ev += [("response.output_text.done", {"type": "response.output_text.done", "item_id": "msg1", "text": text}),
               ("response.content_part.done", {"type": "response.content_part.done", "item_id": "msg1", "part": msg["content"][0]}),
               ("response.output_item.done", {"type": "response.output_item.done", "item": msg})]
        output = [msg]
    ev.append(("response.completed", {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": output}}))
    return sse(*ev)


def whole(api, text=None, structured=False):
    if api == "anthropic":
        content = [{"type": "tool_use", "id": "t", "name": "get_secret", "input": {}}] if structured else [{"type": "text", "text": text}]
        return {"id": "m", "type": "message", "role": "assistant", "content": content, "stop_reason": "tool_use" if structured else "end_turn"}
    if api == "chat":
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "get_secret", "arguments": "{}"}}]} if structured else {"role": "assistant", "content": text}
        return {"id": "c", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if structured else "stop"}]}
    output = [{"type": "function_call", "id": "fc", "call_id": "c", "name": "get_secret", "arguments": "{}"}] if structured else [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    return {"id": "r", "status": "completed", "output": output}


PATHS = {"anthropic": "/v1/messages", "chat": "/v1/chat/completions", "responses": "/v1/responses"}
REQUEST_TOOLS = {
    "anthropic": [{"name": "get_secret", "description": "d", "input_schema": TOOLS["get_secret"]}],
    "chat": [{"type": "function", "function": {"name": "get_secret", "parameters": TOOLS["get_secret"]}}],
    "responses": [{"type": "function", "name": "get_secret", "parameters": TOOLS["get_secret"]}],
}
STREAMS = {"anthropic": anthropic_stream, "chat": chat_stream, "responses": responses_stream}


def request_body(api, stream, tools=True):
    body = {"model": "m", "stream": stream}
    if api == "anthropic":
        body.update(max_tokens=50, messages=[{"role": "user", "content": "hi"}])
    elif api == "chat":
        body.update(messages=[{"role": "user", "content": "hi"}])
    else:
        body.update(input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
    if tools:
        body["tools"] = REQUEST_TOOLS[api]
    return body


def parse_events(raw):
    events = []
    for block in raw.decode().split("\n\n"):
        if block.strip():
            data = [l[5:].strip() for l in block.split("\n") if l.startswith("data:")]
            events.append(data[0] if data[0] == "[DONE]" else json.loads(data[0]))
    return events


class RelayRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reply = None  # (content type, body bytes)

        class Upstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                kind, body = cls.reply
                self.send_response(200)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        cls.relay = make_relay(cls.upstream.server_address[1])
        for server in (cls.upstream, cls.relay):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.upstream, cls.relay):
            server.shutdown()
            server.server_close()

    def ask(self, api, stream, reply, tools=True):
        type(self).reply = reply
        connection = http.client.HTTPConnection("127.0.0.1", self.relay.server_address[1], timeout=20)
        connection.request("POST", PATHS[api], json.dumps(request_body(api, stream, tools)), {"content-type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        headers = dict(response.getheaders())
        connection.close()
        return data, headers

    def each(self):
        for api in PATHS:
            for stream in (False, True):
                yield api, stream, f"{api} {'stream' if stream else 'whole'}"

    def reply_for(self, api, stream, text=None, structured=False):
        if stream:
            return "text/event-stream", STREAMS[api](text, structured)
        return "application/json", json.dumps(whole(api, text, structured)).encode()

    def test_repairs_a_call_written_as_text(self):
        for api, stream, label in self.each():
            with self.subTest(label):
                data, headers = self.ask(api, stream, self.reply_for(api, stream, "Looking.\n" + CALL_TEXT))
                if not stream:
                    self.assertEqual(headers.get("X-Hearthwork-Repaired"), "tool_call")
                    out = json.loads(data)
                    if api == "anthropic":
                        self.assertEqual(out["stop_reason"], "tool_use")
                        self.assertEqual([b["type"] for b in out["content"]], ["text", "tool_use"])
                        self.assertEqual(out["content"][0]["text"], "Looking.")
                        call = out["content"][1]
                        self.assertTrue(call["id"].startswith("toolu_hw_"))
                        self.assertEqual((call["name"], call["input"]), ("get_secret", {"name": "alpha"}))
                    elif api == "chat":
                        choice = out["choices"][0]
                        self.assertEqual(choice["finish_reason"], "tool_calls")
                        self.assertEqual(choice["message"]["content"], "Looking.")
                        function = choice["message"]["tool_calls"][0]["function"]
                        self.assertEqual((function["name"], json.loads(function["arguments"])), ("get_secret", {"name": "alpha"}))
                    else:
                        kinds = [i["type"] for i in out["output"]]
                        self.assertEqual(kinds, ["message", "function_call"])
                        call = out["output"][1]
                        self.assertTrue(call["call_id"])
                        self.assertEqual((call["name"], json.loads(call["arguments"])), ("get_secret", {"name": "alpha"}))
                else:
                    self.check_stream(api, parse_events(data))

    def check_stream(self, api, events):
        if api == "anthropic":
            self.assertEqual(events[-1]["type"], "message_stop")
            starts = [e for e in events if e["type"] == "content_block_start"]
            self.assertEqual([s["content_block"]["type"] for s in starts], ["thinking", "text", "tool_use"])
            self.assertEqual(starts[2]["index"], 2)
            text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta")
            self.assertEqual(text, "Looking.")
            args = "".join(e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
            self.assertEqual(json.loads(args), {"name": "alpha"})
            self.assertEqual([e for e in events if e["type"] == "message_delta"][0]["delta"]["stop_reason"], "tool_use")
            self.assertEqual([e["type"] for e in events if e["type"] == "content_block_stop"].__len__(), 3)
            self.assertTrue(any(e["delta"]["type"] == "thinking_delta" for e in events if e["type"] == "content_block_delta"))
        elif api == "chat":
            self.assertEqual(events[-1], "[DONE]")
            chunks = [e for e in events if e != "[DONE]"]
            text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
            self.assertEqual(text, "Looking.")
            calls = [t for c in chunks for t in c["choices"][0]["delta"].get("tool_calls", [])]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["function"]["name"], "get_secret")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"name": "alpha"})
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
            self.assertTrue(any(c["choices"][0]["delta"].get("reasoning_content") for c in chunks))
        else:
            kinds = [e["type"] for e in events]
            self.assertEqual(kinds[-1], "response.completed")
            self.assertEqual(kinds[kinds.index("response.function_call_arguments.delta") - 1], "response.output_item.added")
            self.assertEqual(kinds[kinds.index("response.function_call_arguments.done") + 1], "response.output_item.done")
            done = [e for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"][0]["item"]
            self.assertEqual((done["name"], json.loads(done["arguments"])), ("get_secret", {"name": "alpha"}))
            output = events[-1]["response"]["output"]
            self.assertEqual([i["type"] for i in output], ["message", "function_call"])
            self.assertEqual(output[0]["content"][0]["text"], "Looking.")
            self.assertEqual(output[1]["call_id"], done["call_id"])
            text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            self.assertEqual(text, "Looking.")
            self.assertEqual([e for e in events if e["type"] == "response.output_text.done"][0]["text"], "Looking.")
            self.assertEqual(kinds.count("response.output_item.done"), 3)

    def test_normal_text_passes_byte_identical(self):
        for text in ("Here is the answer, with {braces} and a function.", "Use <b>bold</b> and [1, 2] then get_sec"):
            for api, stream, label in self.each():
                with self.subTest(label, text=text):
                    reply = self.reply_for(api, stream, text)
                    data, headers = self.ask(api, stream, reply)
                    self.assertEqual(data, reply[1])
                    self.assertNotIn("X-Hearthwork-Repaired", headers)

    def test_held_text_that_is_not_a_call_is_released_unchanged(self):
        text = 'Example: <tool_call>{"name": "rm_rf", "arguments": {}}</tool_call> is what it looks like.'
        for api, stream, label in self.each():
            with self.subTest(label):
                reply = self.reply_for(api, stream, text)
                data, _ = self.ask(api, stream, reply)
                self.assertEqual(data, reply[1])

    def test_structured_call_passes_untouched(self):
        for api, stream, label in self.each():
            with self.subTest(label):
                reply = self.reply_for(api, stream, structured=True)
                data, headers = self.ask(api, stream, reply)
                self.assertEqual(data, reply[1])
                self.assertNotIn("X-Hearthwork-Repaired", headers)

    def test_no_tools_in_the_request_means_no_repair(self):
        for api, stream, label in self.each():
            with self.subTest(label):
                reply = self.reply_for(api, stream, CALL_TEXT)
                data, _ = self.ask(api, stream, reply, tools=False)
                self.assertEqual(data, reply[1])

    def test_off_switch(self):
        import os
        os.environ["HEARTHWORK_TOOL_REPAIR"] = "0"
        try:
            for api, stream, label in self.each():
                with self.subTest(label):
                    reply = self.reply_for(api, stream, CALL_TEXT)
                    data, _ = self.ask(api, stream, reply)
                    self.assertEqual(data, reply[1])
        finally:
            del os.environ["HEARTHWORK_TOOL_REPAIR"]


if __name__ == "__main__":
    unittest.main()
