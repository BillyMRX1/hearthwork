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
        self.check('I will look it up.\n`get_secret(name="alpha")`', "pythonic", rest="I will look it up.")
        found = one("[read_file(path='a.py', limit=3), get_secret(name=\"x)\")]")
        self.assertEqual([c["name"] for c in found.calls], ["read_file", "get_secret"])
        self.assertEqual(found.calls[1]["arguments"], {"name": "x)"})

    def test_raw_json(self):
        self.check('{"name": "get_secret", "arguments": {"name": "alpha"}}', "json")
        self.check('```json\n{"name": "get_secret", "parameters": {"name": "alpha"}}\n```', "json")
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


class Holding(unittest.TestCase):
    def test_states(self):
        names = list(TOOLS)
        self.assertEqual(repair.hold_state("Hello there, I will help.", names), repair.NO)
        self.assertEqual(repair.hold_state("Use <b>bold</b> text", names), repair.NO)
        self.assertEqual(repair.hold_state("text [TOOL_CALLS]", names), repair.YES)
        self.assertEqual(repair.hold_state("text [TOOL", names), repair.MAYBE)
        self.assertEqual(repair.hold_state("text <tool_c", names), repair.MAYBE)
        self.assertEqual(repair.hold_state("a\nget_secret(", names), repair.YES)
        self.assertEqual(repair.hold_state("a\nget_sec", names), repair.MAYBE)
        self.assertEqual(repair.hold_state('a\n{"name"', names), repair.YES)
        self.assertEqual(repair.hold_state("a\n{\n  ", names), repair.MAYBE)


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
