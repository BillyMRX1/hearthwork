import http.server
import json
import threading
import unittest

from hearthwork import doctor
from hearthwork.harnesses import HARNESSES

OK, FAIL, SKIP = doctor.OK, doctor.FAIL, doctor.SKIP


class Fake(http.server.BaseHTTPRequestHandler):
    mode = "good"  # good | text_call | no_end | late500
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        messages = body["messages"]
        roles = [m["role"] for m in messages]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for word in ("1", "2", "3"):
                self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": word}}]})}\n\n'.encode())
            if self.mode != "no_end":
                self.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')
            return
        if self.mode == "late500" and "system" in roles[roles.index("user"):]:
            return self.reply(500, {"error": {"message": "Jinja: System message must be at the beginning."}})
        last = messages[-1]
        message = {"role": "assistant", "content": "ok"}
        if last["role"] == "tool":
            message["content"] = "The secret is ZEBRA-42."
        elif last["content"] == "What was the secret?":
            message["content"] = "It was ZEBRA-42."
        elif "alpha" in str(last["content"]) and body.get("tools"):
            if self.mode == "text_call":
                message["content"] = '<tool_call>{"name": "get_secret", "arguments": {"name": "alpha"}}</tool_call>'
            else:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_secret", "arguments": '{"name": "alpha"}'}}]}
        self.reply(200, {"choices": [{"message": message}]})


def run_against(mode):
    Fake.mode = mode
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        return doctor.protocol_error_hint(doctor.protocol_stages("openai-chat", doctor.Endpoint("127.0.0.1", server.server_address[1], timeout=10), "m"))
    finally:
        server.shutdown()
        server.server_close()


def by_name(stages):
    return {s["name"]: s for s in stages}


class ProtocolTests(unittest.TestCase):
    def test_good_server_passes_everything(self):
        stages = run_against("good")
        self.assertEqual([s["status"] for s in stages], [OK] * 6, stages)

    def test_tool_call_as_text_points_at_issue_9(self):
        stages = by_name(run_against("text_call"))
        self.assertEqual(stages["tool call"]["status"], FAIL)
        self.assertIn("#9", stages["tool call"]["suggestion"])
        self.assertEqual(stages["tool result"]["status"], SKIP)
        self.assertEqual(stages["next turn"]["status"], SKIP)
        self.assertEqual(stages["late system message"]["status"], OK)

    def test_stream_without_end_event(self):
        stages = by_name(run_against("no_end"))
        self.assertEqual(stages["streaming"]["status"], FAIL)
        self.assertIn("end event", stages["streaming"]["suggestion"])
        self.assertEqual(stages["plain reply"]["status"], OK)

    def test_500_on_late_system_message(self):
        stages = by_name(run_against("late500"))
        self.assertEqual(stages["late system message"]["status"], FAIL)
        self.assertIn("HTTP 500", stages["late system message"]["detail"])
        self.assertIn("system", stages["late system message"]["suggestion"])
        self.assertEqual(stages["tool call"]["status"], OK)


class WireTests(unittest.TestCase):
    CONV = [("user", "hi"), ("call", "c1", "get_secret", '{"name": "alpha"}'), ("result", "c1", "ZEBRA-42"),
            ("assistant", "ok"), ("system", "late"), ("user", "again")]

    def test_anthropic_blocks(self):
        body = doctor.wire("anthropic", self.CONV, "m", tools=True)
        kinds = [(m["role"], [b["type"] for b in m["content"]]) for m in body["messages"]]
        self.assertEqual(kinds[1], ("assistant", ["tool_use"]))
        self.assertEqual(kinds[2], ("user", ["tool_result"]))
        self.assertEqual(kinds[-1], ("user", ["text", "text"]))  # the late system text rides along as a reminder
        self.assertNotIn("system", [m["role"] for m in body["messages"]])

    def test_responses_items(self):
        body = doctor.wire("openai-responses", self.CONV, "m", tools=True)
        self.assertEqual([i.get("type") or i["role"] for i in body["input"]],
                         ["message", "function_call", "function_call_output", "message", "message", "message"])
        self.assertEqual([i.get("role") for i in body["input"] if i["type"] == "message"], ["user", "assistant", "developer", "user"])

    def test_chat_roles(self):
        body = doctor.wire("openai-chat", self.CONV, "m")
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "tool", "assistant", "system", "user"])

    def test_stream_summary(self):
        events = [("content_block_delta", json.dumps({"type": "content_block_delta", "delta": {"text": t}})) for t in "ab"]
        self.assertEqual(doctor.stream_summary("anthropic", events), (2, False))
        events.append(("message_stop", '{"type": "message_stop"}'))
        self.assertEqual(doctor.stream_summary("anthropic", events), (2, True))


class VerdictTests(unittest.TestCase):
    good = [doctor.stage(n, OK) for n in doctor.PROTOCOL]
    server = doctor.stage("server", OK)

    def agent(self, **status):
        names = ["installed", "configuration", "headless reply", "coding task"]
        return [doctor.stage(n, status.get(n.replace(" ", "_"), OK), cause="model") for n in names]

    def test_works(self):
        self.assertEqual(doctor.verdict("claude", self.server, self.good, self.agent())[0], "works")

    def test_server_down(self):
        self.assertIn("connection problem at server", doctor.verdict("claude", doctor.stage("server", FAIL), [], self.agent())[0])

    def test_protocol_problem_names_api(self):
        bad = [dict(s) for s in self.good]
        bad[1]["status"] = FAIL
        text = doctor.verdict("codex", self.server, bad, self.agent())[0]
        self.assertEqual(text, "protocol problem at streaming (OpenAI Responses)")

    def test_coding_failure_is_capability(self):
        text = doctor.verdict("claude", self.server, self.good, self.agent(coding_task=FAIL))[0]
        self.assertIn("model capability", text)

    def test_aider_ignores_tool_stages(self):
        bad = [dict(s) for s in self.good]
        bad[2]["status"] = FAIL
        self.assertEqual(doctor.verdict("aider", self.server, bad, self.agent())[0], "works")
        self.assertIn("protocol problem at tool call", doctor.verdict("qwen", self.server, bad, self.agent())[0])

    def test_headless_failure_is_configuration(self):
        text = doctor.verdict("claude", self.server, self.good, self.agent(headless_reply=FAIL))[0]
        self.assertEqual(text, "configuration/connection problem at headless reply")

    def test_not_installed_suggests_install(self):
        self.assertTrue(HARNESSES["aider"]["install"])


class SecretTests(unittest.TestCase):
    def test_report_has_no_credentials(self):
        report = {"a": {"detail": "failed with Bearer abcdef123456789 and key hw-secret-key-1"}, "ANTHROPIC_API_KEY": "x"}
        text = json.dumps(doctor.scrub(report, ["hw-secret-key-1"]))
        self.assertNotIn("abcdef123456789", text)
        self.assertNotIn("hw-secret-key-1", text)
        self.assertNotIn('"x"', text)


if __name__ == "__main__":
    unittest.main()


class ReviewFixTests(unittest.TestCase):
    good = [doctor.stage(n, OK) for n in doctor.PROTOCOL]
    server = doctor.stage("server", OK)

    def agent(self, coding=None):
        found = [doctor.stage(n, OK) for n in ("installed", "configuration", "headless reply")]
        return found + [coding or doctor.stage("coding task", OK)]

    def with_failure(self, index, cause):
        stages = [dict(s) for s in self.good]
        stages[index].update(status=FAIL, cause=cause)
        return stages

    def test_context_policy_is_the_shared_minimum(self):
        from hearthwork import context
        self.assertIs(doctor.MIN_CONTEXT, context.MIN_CONTEXT)

    def test_model_cause_is_capability_not_protocol(self):
        text = doctor.verdict("claude", self.server, self.with_failure(2, "model"), self.agent())[0]
        self.assertEqual(text, "protocol OK, but the model failed at tool call (model capability)")

    def test_protocol_cause_is_protocol(self):
        text = doctor.verdict("claude", self.server, self.with_failure(1, "protocol"), self.agent())[0]
        self.assertEqual(text, "protocol problem at streaming (Anthropic Messages)")

    def test_wrong_tool_args_and_text_answer_are_model_causes(self):
        for mode, expected in (("text_call", "protocol"),):
            stages = by_name(run_against(mode))
            self.assertEqual(stages["tool call"]["cause"], expected)

    def test_coding_run_error_vs_grader_failure(self):
        crashed = doctor.stage("coding task", FAIL, 0, "timeout: timed out after 600 s", "x", "agent")
        self.assertEqual(doctor.verdict("claude", self.server, self.good, self.agent(crashed))[0],
                         "configuration/connection problem at coding task (timeout)")
        graded = doctor.stage("coding task", FAIL, 0, "graded", "x", "model")
        self.assertIn("model capability", doctor.verdict("claude", self.server, self.good, self.agent(graded))[0])

    def test_truncation_detected_per_api(self):
        self.assertTrue(doctor.truncated("anthropic", {"stop_reason": "max_tokens"}))
        self.assertTrue(doctor.truncated("openai-chat", {"choices": [{"finish_reason": "length"}]}))
        self.assertTrue(doctor.truncated("openai-responses", {"status": "incomplete"}))
        self.assertFalse(doctor.truncated("openai-chat", {"choices": [{"finish_reason": "stop"}]}))
        with self.assertRaises(doctor.OutputLimit):
            doctor.parse_reply("openai-chat", {"choices": [{"finish_reason": "length", "message": {"content": "x"}}]})
        item = doctor.timed("plain reply", lambda: doctor.parse_reply("anthropic", {"stop_reason": "max_tokens", "content": []}))
        self.assertEqual(item["cause"], "limit")
        self.assertIn("output limit", item["detail"])
        text = doctor.verdict("claude", self.server, self.with_failure(0, "limit"), self.agent())[0]
        self.assertIn("output limit", text)

    def test_empty_answer_after_think_is_model_cause(self):
        original = doctor.Endpoint.send
        doctor.Endpoint.send = lambda self, api, body: {"choices": [{"finish_reason": "stop", "message": {"content": "<think>hmm</think>"}}]}
        try:
            stages = doctor.protocol_stages("openai-chat", doctor.Endpoint("x", 1), "m")
        finally:
            doctor.Endpoint.send = original
        self.assertEqual((stages[0]["status"], stages[0]["cause"]), (FAIL, "model"))

    def test_suggestions_are_strings_and_causes_set(self):
        stages = by_name(run_against("text_call"))
        self.assertIsInstance(stages["tool call"]["suggestion"], str)
        self.assertEqual(stages["tool call"]["cause"], "protocol")
        self.assertTrue(all(isinstance(s.get("suggestion", ""), str) for s in stages.values()))

    def test_think_blocks_are_stripped(self):
        self.assertEqual(doctor.strip_think("<think>ZEBRA-42 maybe</think>ok"), "ok")
        self.assertEqual(doctor.strip_think("answer<think>ZEBRA-42 unfinished"), "answer")
        text, _ = doctor.parse_reply("openai-chat", {"choices": [{"message": {"content": "<think>ZEBRA-42</think>nope"}}]})
        self.assertNotIn("ZEBRA-42", text)

    def test_remote_never_asks_for_local_context(self):
        from unittest import mock
        config = {"remote": {"host": "h", "port": 1, "key": "k"}}
        with mock.patch("hearthwork.remote.session", return_value=("m", 131072)):
            item, context = doctor.server_stage(config, config["remote"])
        self.assertEqual((item["status"], context), (OK, 131072))

    def test_environment_failure_is_recorded_and_run_continues(self):
        from unittest import mock
        config = {"server": {"port": 1}}
        relay = mock.Mock(server_address=("127.0.0.1", 1))
        with mock.patch.object(doctor, "server_stage", return_value=(doctor.stage("server", OK), None)), \
                mock.patch.object(doctor, "environment", side_effect=RuntimeError("boom")), \
                mock.patch.object(doctor, "make_relay", return_value=relay), \
                mock.patch.object(doctor, "protocol_stages", return_value=[]), \
                mock.patch.object(doctor, "agent_stages", return_value=[]):
            report = doctor.run_doctor(config, ["claude"], quick=True)
        self.assertIn("boom", report["environment"]["error"])
        with mock.patch.object(doctor, "server_stage", return_value=(doctor.stage("server", FAIL, 0, "x", "y"), None)):
            report = doctor.run_doctor(config, ["claude"], quick=True)
        self.assertEqual(report["agents"]["claude"]["verdict"], "configuration/connection problem at server")
