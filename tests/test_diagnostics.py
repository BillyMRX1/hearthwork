"""Session diagnostics: timings extraction per API, percentiles, findings, capacity errors, sanitized export, passive recording."""
import http.client
import http.server
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from hearthwork import diagnostics as d, harnesses, host

TIMINGS = {"cache_n": 7, "prompt_n": 4, "prompt_ms": 120, "prompt_per_second": 32.5, "predicted_n": 5, "predicted_ms": 110, "predicted_per_second": 35.8}
CHAT_USAGE = {"completion_tokens": 5, "prompt_tokens": 11, "prompt_tokens_details": {"cached_tokens": 7}}


class Extract(unittest.TestCase):
    def test_chat_json(self):
        text = json.dumps({"choices": [], "usage": CHAT_USAGE, "timings": TIMINGS})
        self.assertEqual(d.extract(text), (11, 7, 5, 32.5, 35.8))

    def test_chat_stream_final_chunk(self):
        text = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: ' + json.dumps({"choices": [], "usage": CHAT_USAGE, "timings": TIMINGS}) + "\n\ndata: [DONE]\n\n"
        self.assertEqual(d.extract(text), (11, 7, 5, 32.5, 35.8))

    def test_responses_stream_timings_beside_response(self):
        done = {"type": "response.completed", "response": {"usage": {"input_tokens": 11, "output_tokens": 16, "input_tokens_details": {"cached_tokens": 7}}}, "timings": TIMINGS}
        self.assertEqual(d.extract("event: response.completed\ndata: " + json.dumps(done) + "\n\n"), (11, 7, 16, 32.5, 35.8))

    def test_responses_json_without_timings_has_no_speeds(self):
        text = json.dumps({"usage": {"input_tokens": 11, "output_tokens": 16, "input_tokens_details": {"cached_tokens": 7}}})
        self.assertEqual(d.extract(text), (11, 7, 16, None, None))

    def test_messages_input_is_uncached_part(self):
        text = json.dumps({"usage": {"cache_read_input_tokens": 7, "input_tokens": 4, "output_tokens": 5}})
        self.assertEqual(d.extract(text), (11, 7, 5, None, None))

    def test_messages_stream_merges_start_and_delta(self):
        text = ('event: message_start\ndata: {"type":"message_start","message":{"usage":{"cache_read_input_tokens":7,"input_tokens":4,"output_tokens":0}}}\n\n'
                'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":5}}\n\n')
        self.assertEqual(d.extract(text), (11, 7, 5, None, None))

    def test_partial_and_missing(self):
        whole = json.dumps({"usage": CHAT_USAGE, "timings": TIMINGS})
        self.assertEqual(d.extract(whole[:-20]), (11, 7, 5, None, None))  # cut inside timings: usage still read, speeds not guessed
        self.assertEqual(d.extract("data: [DONE]"), (None,) * 5)

    def test_content_that_looks_like_usage_is_ignored(self):
        text = json.dumps({"choices": [{"message": {"content": '"usage": {"prompt_tokens": 999}'}}], "usage": CHAT_USAGE})
        self.assertEqual(d.extract(text)[0], 11)

    def test_requested_output(self):
        self.assertEqual(d.requested_output(b'{"max_output_tokens": 64}'), 64)
        self.assertIsNone(d.requested_output(b'{"x": 1}'))
        self.assertIsNone(d.requested_output(b"not json"))


class Numbers(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(d.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(d.percentile([10, 20], 50), 15)
        self.assertAlmostEqual(d.percentile(list(range(1, 101)), 95), 95.05)
        self.assertIsNone(d.percentile([None], 50))

    def test_durations(self):
        self.assertEqual(d.parse_duration("30m"), 1800)
        self.assertEqual(d.parse_duration("2h"), 7200)
        with self.assertRaises(ValueError):
            d.parse_duration("soon")


def rec(**kw):
    base = {"ts": time.time(), "client": "local-opencode-1", "api": "/v1/chat/completions", "model": "m.gguf", "outcome": "ok", "status": 200, "wait_ms": 0,
            "ttfb_ms": 500, "total_ms": 3000, "prompt_tokens": 1000, "cached_tokens": 900, "gen_tokens": 50, "prompt_tps": 500.0, "gen_tps": 30.0, "gen_tps_source": "server",
            "n_ctx": 32768, "slots": 2, "max_output": 1024, "active": 1, "queued_at_arrival": 0, "capacity_error": False}
    base.update(kw)
    return base


class Findings(unittest.TestCase):
    def kinds(self, records):
        return [(f["kind"], f["text"]) for f in d.findings(d.summarize(records))]

    def test_healthy_session_has_no_findings(self):
        self.assertEqual(self.kinds([rec() for _ in range(5)]), [])

    def test_queue_time_is_observed(self):
        got = self.kinds([rec(wait_ms=2000, total_ms=3000, client=f"local-a-{i}") for i in range(4)])
        self.assertTrue(any(k == "observed" and "waiting in the queue" in t and "4 client" in t for k, t in got))

    def test_low_cache_is_observed_and_cause_is_possible(self):
        got = self.kinds([rec(cached_tokens=10) for _ in range(4)])
        self.assertTrue(any(k == "observed" and "Cache hit ratio" in t for k, t in got))
        self.assertTrue(any(k == "possible" and "evicting" in t for k, t in got))
        self.assertFalse(any(k == "possible" and "Cache hit ratio" in t for k, t in got))

    def test_context_pressure(self):
        got = self.kinds([rec(prompt_tokens=30000) for _ in range(3)])
        self.assertTrue(any("80%" in t for k, t in got))

    def test_unavailable_stays_none(self):
        s = d.summarize([rec(prompt_tokens=None, cached_tokens=None, gen_tps=None, prompt_tps=None, ttfb_ms=None, n_ctx=None)])
        self.assertIsNone(s["cache"]["hit_ratio"])
        self.assertIsNone(s["gen_tps_median"])
        self.assertIsNone(s["context"]["share_above_80pct"])
        text = d.format_report(d.report([rec(gen_tps=None)]))
        self.assertIn("unavailable", text)
        self.assertIn("time spent in tools", text)  # says what it does not measure


class RelaySpeeds(unittest.TestCase):
    def test_medians_never_mix_sources(self):
        s = d.summarize([rec(gen_tps=30.0), rec(gen_tps=50.0), rec(gen_tps=5.0, gen_tps_source="relay-total")])
        self.assertEqual((s["gen_tps_median"], s["gen_tps_source"]), (40.0, "server"))
        self.assertEqual(s["gen_tps_by_source"]["relay-total"], 5.0)
        s = d.summarize([rec(gen_tps=5.0, gen_tps_source="relay-total")])
        self.assertEqual(s["gen_tps_source"], "relay-total")
        self.assertIn("relay-measured, includes prompt processing", d.format_report(d.report([rec(gen_tps=5.0, gen_tps_source="relay-total")])))
        self.assertFalse([f for f in d.findings(s) if "generation speed" in f["text"]])  # not judged: includes prompt time

    def probe(self, body, chunks, delay, status=200):
        recorder = d.Recorder(Path(tempfile.mkdtemp()) / "r.jsonl")
        p = d.begin("api", "/v1/messages", body, recorder=recorder)
        p.started = time.monotonic()
        p.response(status)
        for c in chunks:
            p.feed(c)
            time.sleep(delay)
        p.finish("ok")
        return recorder.recent()[0]

    def test_anthropic_stream_uses_relay_clock(self):
        chunks = [b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"cache_read_input_tokens":7,"input_tokens":4,"output_tokens":0}}}\n\n',
                  b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":40}}\n\n']
        r = self.probe(b'{"stream": true}', chunks, 0.2)
        self.assertEqual((r["prompt_tokens"], r["gen_tokens"], r["gen_tps_source"]), (11, 40, "relay-stream"))
        self.assertTrue(50 < r["gen_tps"] < 250)  # 40 tokens over ~0.2 s

    def test_nonstream_is_total_and_server_timings_win(self):
        r = self.probe(b"{}", [b'{"usage":{"input_tokens":4,"output_tokens":20,"cache_read_input_tokens":0}}'], 0.2)
        self.assertEqual(r["gen_tps_source"], "relay-total")
        r = self.probe(b'{"stream": true}', [json.dumps({"usage": CHAT_USAGE, "timings": TIMINGS}).encode()], 0.1)
        self.assertEqual((r["gen_tps_source"], r["gen_tps"]), ("server", 35.8))


class Capacity(unittest.TestCase):
    def test_openai_shape(self):
        err = {"error": {"code": 400, "message": "request (200010 tokens) exceeds the available context size (131072 tokens), try increasing it",
                         "type": "exceed_context_size_error", "n_prompt_tokens": 200010, "n_ctx": 131072}}
        new, prompt, n_ctx = d.rewrite_capacity(json.dumps(err).encode(), slots=2)
        out = json.loads(new)
        self.assertEqual((prompt, n_ctx), (200010, 131072))
        self.assertEqual(out["error"]["code"], 400)
        self.assertIn("200,010 tokens don't fit the model's 131,072 context (per session; 2 sessions share the pool)", out["error"]["message"])
        self.assertIn("hearthwork context <model> <N>", out["error"]["message"])

    def test_anthropic_shape_numbers_from_message(self):
        err = {"type": "error", "error": {"type": "invalid_request_error", "message": "request (9000 tokens) exceeds the available context size (8192 tokens), try increasing it"}}
        new, prompt, n_ctx = d.rewrite_capacity(json.dumps(err).encode())
        out = json.loads(new)
        self.assertEqual(out["type"], "error")
        self.assertEqual(out["error"]["type"], "invalid_request_error")
        self.assertIn("9,000 tokens don't fit the model's 8,192 context", out["error"]["message"])

    def test_other_errors_untouched(self):
        self.assertIsNone(d.rewrite_capacity(b'{"error": {"message": "bad json"}}'))
        self.assertIsNone(d.rewrite_capacity(b"oops"))


class Export(unittest.TestCase):
    def test_sanitized_export_has_no_content(self):
        records = [rec(client="share:Brians phone", model="D:\\Users\\brian\\models\\m.gguf", prompt="SECRET CODE", response="SECRET OUT")]
        text = json.dumps(d.report(records))
        for word in ("SECRET", "Brians", "brian", "D:\\\\"):
            self.assertNotIn(word, text)
        self.assertIn("share:device-1", text)

    def test_cli_export_rotation_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            recorder = d.Recorder(path, size=3, rotate_at=700)
            for i in range(6):
                recorder.add(rec(ts=time.time() + i))
            self.assertEqual(len(recorder.ring), 3)
            self.assertTrue(Path(str(path) + ".1").exists())  # rotated
            self.assertGreaterEqual(len(d.load_records(path)), 4)
            out = Path(folder) / "report.json"
            old = d.REQUESTS_FILE
            d.REQUESTS_FILE = path
            try:
                self.assertEqual(d.main(["--last", "4", "--export", str(out)]), 0)
            finally:
                d.REQUESTS_FILE = old
            self.assertEqual(json.loads(out.read_text())["summary"]["requests"], min(4, len(d.load_records(path))))


class FakeLlama(http.server.ThreadingHTTPServer):
    """Streams three chunks 0.4 s apart; a prompt marked "big" gets llama.cpp's context error."""
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n', b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
                  b"data: " + json.dumps({"choices": [], "usage": CHAT_USAGE, "timings": TIMINGS}).encode() + b"\n\ndata: [DONE]\n\n"]

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if b"big" in body:
                data = json.dumps({"error": {"code": 400, "message": "request (900 tokens) exceeds the available context size (512 tokens), try increasing it",
                                             "type": "exceed_context_size_error", "n_prompt_tokens": 900, "n_ctx": 512}}).encode()
                self.send_response(400)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for chunk in self.chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.4)


class Passive(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.old = d.RECORDER
        d.RECORDER = d.Recorder(Path(self.folder.name) / "requests.jsonl")
        d.RECORDER.context = lambda: (512, 2, "C:\\x\\model.gguf")
        self.llama = FakeLlama()
        self.relay = harnesses.make_relay(self.llama.server_address[1], gate=host.Gate(lambda: 2))
        threading.Thread(target=self.relay.serve_forever, daemon=True).start()

    def tearDown(self):
        d.RECORDER = self.old
        self.relay.shutdown()
        self.llama.shutdown()
        self.folder.cleanup()

    def post(self, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.relay.server_address[1], timeout=10)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"content-type": "application/json"})
        return conn.getresponse()

    def test_stream_is_byte_identical_and_not_buffered(self):
        response = self.post({"stream": True, "max_tokens": 64, "messages": [{"role": "user", "content": "hello"}]})
        first = response.read1(65536)
        first_at = time.monotonic()
        rest = response.read()
        self.assertEqual(first, FakeLlama.Handler.chunks[0])  # arrived on its own, before the stream ended
        self.assertGreater(time.monotonic() - first_at, 0.5)
        self.assertEqual(first + rest, b"".join(FakeLlama.Handler.chunks))
        time.sleep(0.3)
        (r,) = d.RECORDER.recent()
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["gen_tokens"], r["outcome"], r["status"]), (11, 7, 5, "ok", 200))
        self.assertEqual((r["max_output"], r["n_ctx"], r["model"]), (64, 512, "model.gguf"))
        self.assertLess(r["ttfb_ms"], 300)
        self.assertGreater(r["total_ms"], 800)
        self.assertNotIn("hello", json.dumps(r))
        self.assertNotIn("hello", (Path(self.folder.name) / "requests.jsonl").read_text())

    def test_capacity_error_rewritten_with_same_status(self):
        response = self.post({"messages": [{"role": "user", "content": "big"}]})
        data = response.read()
        self.assertEqual(response.status, 400)
        self.assertEqual(int(response.getheader("content-length")), len(data))
        self.assertIn("900 tokens don't fit the model's 512 context (per session; 2 sessions share the pool)", json.loads(data)["error"]["message"])
        time.sleep(0.2)
        (r,) = d.RECORDER.recent()
        self.assertTrue(r["capacity_error"])
        self.assertEqual(r["outcome"], "error")


class ExportFolders(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.records = Path(self.folder.name) / "requests.jsonl"
        self.records.write_text(json.dumps(rec(ts=time.time() - 3 * 86400)) + "\n" + json.dumps(rec(ts=time.time() - 60)) + "\n", encoding="utf-8")
        self.old = d.REQUESTS_FILE
        d.REQUESTS_FILE = self.records

    def tearDown(self):
        d.REQUESTS_FILE = self.old
        self.folder.cleanup()

    def run_main(self, argv):
        import contextlib
        import io
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = d.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_missing_folders_are_created_and_path_printed(self):
        target = Path(self.folder.name) / "new" / "deeper" / "d.json"
        code, out, err = self.run_main(["--export", str(target)])
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())
        self.assertIn("written to", out)
        self.assertIn(str(target.resolve()), out)
        self.assertEqual(err, "")

    def test_unwritable_target_is_one_clear_line_and_exit_1(self):
        blocker = Path(self.folder.name) / "file.txt"
        blocker.write_text("x", encoding="utf-8")
        code, out, err = self.run_main(["--export", str(blocker / "d.json")])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("hearthwork diagnose: cannot write"))
        self.assertEqual(err.count("\n"), 1)
        self.assertNotIn("Traceback", err)

    def test_default_window_is_last_24_hours(self):
        target = Path(self.folder.name) / "d.json"
        self.assertEqual(self.run_main(["--export", str(target)])[0], 0)
        report = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["requests"], 1)
        self.assertEqual(report["selection"]["label"], "last 24 h; use --since or --last to change")
        self.assertEqual(self.run_main(["--all", "--export", str(target)])[0], 0)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["summary"]["requests"], 2)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["selection"]["label"], "all recorded history")

    def test_explicit_selections_still_work(self):
        target = Path(self.folder.name) / "d.json"
        self.run_main(["--since", "2d", "--export", str(target)])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["summary"]["requests"], 1)
        self.run_main(["--last", "2", "--export", str(target)])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["summary"]["requests"], 2)
        self.run_main(["--client", "nobody", "--export", str(target)])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["summary"]["requests"], 0)
        with self.assertRaises(SystemExit):  # the two options conflict
            self.run_main(["--all", "--since", "1h", "--export", str(target)])

    def test_header_says_the_window(self):
        text = d.format_report(d.report([rec()], {"label": "last 24 h; use --since or --last to change"}))
        self.assertTrue(text.startswith("Selection: last 24 h; use --since or --last to change\n"))


class RunningNow(unittest.TestCase):
    def test_status_counts_running_requests(self):
        self.assertEqual(d.status_line({"requests": 0, "running": 2}), "last 5 min: 0 finished, 2 running now")
        self.assertEqual(d.status_line({"requests": 0}), "last 5 min: no requests")
        self.assertEqual(d.status_line(None), "last 5 min: no requests")
        text = d.status_line(dict(d.summarize([rec(), rec()]), running=1))
        self.assertTrue(text.startswith("last 5 min: 2 finished, median generation"))
        self.assertTrue(text.endswith(", 1 running now"))
        self.assertNotIn("running", d.status_line(d.summarize([rec()])))

    def test_recorder_tracks_admitted_requests_until_finished(self):
        recorder = d.Recorder(Path(tempfile.mkdtemp()) / "r.jsonl")
        p = d.begin("api", "/v1/messages", b"{}", recorder=recorder)
        self.assertEqual(recorder.running(), 0)  # queued, not admitted yet
        p.started = time.monotonic()
        self.assertEqual(recorder.running(), 1)
        p.finish("ok")
        p.finish("ok")  # idempotent
        self.assertEqual(recorder.running(), 0)


class PromptSpeeds(unittest.TestCase):
    def probe(self, chunks, delay_before_first=0.0):
        recorder = d.Recorder(Path(tempfile.mkdtemp()) / "r.jsonl")
        p = d.begin("api", "/v1/messages", b'{"stream": true}', recorder=recorder)
        p.started = time.monotonic()
        p.response(200)
        time.sleep(delay_before_first)
        for c in chunks:
            p.feed(c)
        p.finish("ok")
        return recorder.recent()[0]

    START = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"cache_read_input_tokens":7,"input_tokens":4,"output_tokens":0}}}\n\n')
    DELTA = b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":40}}\n\n'

    def test_estimated_from_uncached_tokens_over_first_byte(self):
        r = self.probe([self.START, self.DELTA], delay_before_first=0.4)
        self.assertEqual((r["prompt_tokens"], r["cached_tokens"], r["prompt_tps_source"]), (11, 7, "relay-estimated"))
        self.assertTrue(4 / 0.5 < r["prompt_tps"] < 4 / 0.35)  # 4 uncached tokens over about 0.4 s

    def test_skipped_when_first_byte_is_tiny(self):
        r = self.probe([self.START, self.DELTA])
        self.assertIsNone(r["prompt_tps"])
        self.assertIsNone(r["prompt_tps_source"])

    def test_skipped_when_tokens_missing(self):
        r = self.probe([b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":40}}\n\n'], delay_before_first=0.4)
        self.assertIsNone(r["prompt_tokens"])
        self.assertIsNone(r["prompt_tps"])

    def test_server_timings_win_over_estimate(self):
        r = self.probe([json.dumps({"usage": CHAT_USAGE, "timings": TIMINGS}).encode()], delay_before_first=0.4)
        self.assertEqual((r["prompt_tps"], r["prompt_tps_source"]), (32.5, "server"))

    def test_medians_never_mix_sources(self):
        s = d.summarize([rec(prompt_tps=500.0), rec(prompt_tps=600.0), rec(prompt_tps=10.0, prompt_tps_source="relay-estimated")])
        self.assertEqual((s["prompt_tps_median"], s["prompt_tps_source"]), (550.0, "server"))
        s = d.summarize([rec(prompt_tps=None), rec(prompt_tps=10.0, prompt_tps_source="relay-estimated")])
        self.assertEqual((s["prompt_tps_median"], s["prompt_tps_source"]), (10.0, "relay-estimated"))
        self.assertIn("prompt 10.0 tokens/s (relay-estimated, includes overhead)", d.format_report(d.report([rec(prompt_tps=None), rec(prompt_tps=10.0, prompt_tps_source="relay-estimated")])))
        self.assertIn("prompt unavailable", d.format_report(d.report([rec(prompt_tps=None)])))


if __name__ == "__main__":
    unittest.main()
