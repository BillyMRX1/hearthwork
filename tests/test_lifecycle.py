"""Host lifecycle (#15 step 2, #20): switch/drain/force, idle unload, crash and OOM recovery, restart limits, health checks,
retry before the request was accepted, mid-stream interruption. A fake llama-server runs as a real process."""
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import context as ctx, harnesses, host, server

FAKE = str(Path(__file__).with_name("fake_llama.py"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(check, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def post(port, path, payload, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, json.dumps(payload), {"content-type": "application/json"})
    response = conn.getresponse()
    return response.status, response.read().decode(errors="replace")


class Rig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fail_file = self.tmp / "fail"
        for name in ("A", "B"):
            (self.tmp / f"{name}.gguf").write_text("x")
        self.model_a, self.model_b = self.tmp / "A.gguf", self.tmp / "B.gguf"
        os.environ["FAKE_FAIL_FILE"] = str(self.fail_file)
        self.llama = free_port()

        def command(config, model, context=None, slot_path=None):
            return [sys.executable, FAKE, "--port", str(self.llama), "-m", str(model)]

        patches = [mock.patch.object(host, "STATE", self.tmp / "server.json"), mock.patch.object(server, "STATE", self.tmp / "server.json"),
                   mock.patch.object(host, "LOG", self.tmp / "server.log"), mock.patch.object(server, "LOG", self.tmp / "server.log"),
                   mock.patch.object(host, "HOST_FILE", self.tmp / "host.json"),
                   mock.patch.object(server, "command", command), mock.patch.object(ctx, "choose", lambda c, m, x=None: (x or 4096, "test", None)),
                   mock.patch.object(ctx, "describe", lambda *a: "context 4096"),
                   mock.patch.object(host, "BACKOFF", (0.1, 0.1, 0.1)), mock.patch.object(host, "LOAD_POLL", 0.1),
                   mock.patch.object(host, "HEALTH_EVERY", 0.2), mock.patch.object(host, "HEALTH_TIMEOUT", 0.5),
                   mock.patch.object(host, "HEALTH_FAILS", 3), mock.patch.object(host, "FORCE_GRACE", 0.5)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.host = host.Host({"server": {"port": self.llama, "slots": 2}}, lambda m: None)
        self.host.load(self.model_a, None)
        self.host.set_ready()
        self.listener = harnesses.make_relay(self.llama, ("127.0.0.1", 0), self.host.control, gate=self.host.gate)
        threading.Thread(target=self.listener.serve_forever, daemon=True).start()
        self.port = self.listener.server_address[1]
        self.running = True
        threading.Thread(target=self.monitor, daemon=True).start()
        self.addCleanup(self.teardown)

    def monitor(self):
        while self.running:
            self.host.tick()
            time.sleep(0.1)

    def teardown(self):
        self.running = False
        self.host.stop_event.set()
        time.sleep(0.3)
        self.listener.shutdown()
        self.listener.server_close()
        try:
            server.stop(self.host.config, quiet=True)
        except Exception:
            pass

    def stream(self, path="/v1/chat/completions", count=10, delay=0.1):
        """Start a streaming request; returns (thread, result) where result gets "status" and "body"."""
        result = {}

        def run():
            try:
                result["status"], result["body"] = post(self.port, path, {"stream": True, "count": count, "delay": delay})
            except Exception as error:  # noqa
                result["error"] = repr(error)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, result

    def llama_pid(self):
        return json.loads((self.tmp / "server.json").read_text())["pid"]


class SwitchTests(Rig):
    def test_drain_then_switch_and_queue_waits(self):
        t, result = self.stream(count=15, delay=0.1)
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        code, _ = self.host.switch(str(self.model_b), None, False)
        self.assertEqual(code, 202)
        status = self.host.status()
        self.assertEqual((status["transition"]["kind"], status["transition"]["phase"]), ("switch", "draining"))
        self.assertEqual(status["transition"]["draining"], 1)
        self.assertIn("switching to B (draining 1)", host.transition_text(status))
        waiting = {}
        q = threading.Thread(target=lambda: waiting.update(zip(("status", "body"), post(self.port, "/v1/chat/completions", {"n": 1}))), daemon=True)
        q.start()  # arrives during the switch: waits, is not refused
        t.join(20)
        self.assertEqual(result["status"], 200)
        self.assertIn("tok14", result["body"])  # the running request finished
        q.join(60)
        self.assertEqual(waiting["status"], 200)
        self.assertIn("B.gguf", waiting["body"])  # and the queued one was answered by the new model
        self.assertTrue(wait_for(lambda: self.host.state == "ready"))
        self.assertEqual(self.host.model, "B")

    def test_force_cancels_running(self):
        t, result = self.stream(count=300, delay=0.1)
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        time.sleep(0.3)
        self.assertEqual(self.host.switch(str(self.model_b), None, True)[0], 202)
        t.join(20)
        self.assertIn("event: error", result["body"].replace("data: {\"error\"", "event: error\ndata"))
        self.assertIn("cancelled", result["body"])
        self.assertNotIn("tok299", result["body"])
        self.assertTrue(wait_for(lambda: self.host.state == "ready" and self.host.model == "B", 30))

    def test_concurrent_switch_refused(self):
        t, _ = self.stream(count=10, delay=0.1)
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        self.assertEqual(self.host.switch(str(self.model_b), None, False)[0], 202)
        code, payload = self.host.switch(str(self.model_a), None, False)
        self.assertEqual(code, 409)
        self.assertIn("a switch is already in progress", payload["error"]["message"])
        t.join(20)
        self.assertTrue(wait_for(lambda: self.host.state == "ready" and self.host.model == "B", 30))

    def test_control_api_switch(self):
        info = {"port": self.port, "token": self.host.token}
        answer = host.call(info, "switch", "POST", payload={"model": str(self.model_b)}, errors=True)
        self.assertEqual(answer, {"switching": "B"})
        again = host.call(info, "switch", "POST", payload={"model": str(self.model_a)}, errors=True)
        if again.get("error"):  # only while the first is still running
            self.assertIn("in progress", again["error"]["message"])
        self.assertTrue(wait_for(lambda: self.host.state == "ready" and self.host.model == "B", 30))


class IdleTests(Rig):
    def test_unload_and_reload_on_request(self):
        self.host.idle_minutes = 0.02
        self.assertTrue(wait_for(lambda: self.host.state == "unloaded", 10))
        self.assertFalse(self.host.ready)
        self.assertIn("unloaded", host.transition_text(self.host.status()))
        self.assertFalse(host.health(self.llama, 0.5)[0])  # llama-server is gone
        self.host.idle_minutes = 0
        status, body = post(self.port, "/v1/chat/completions", {"n": 1}, timeout=60)  # waits while the model loads
        self.assertEqual(status, 200)
        self.assertEqual(self.host.state, "ready")

    def test_never_while_active_or_queued(self):
        self.host.idle_minutes = 0.01
        t, result = self.stream(count=25, delay=0.1)  # ~2.5 s, longer than the idle time
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        time.sleep(1.5)
        self.assertEqual(self.host.state, "ready")
        t.join(20)
        self.assertIn("tok24", result["body"])
        self.host.idle_minutes = 0
        entry = {"client": "x", "path": "/", "queued": time.monotonic()}
        self.host.gate.queue.append(entry)
        self.host.unload()
        self.assertEqual(self.host.state, "ready")
        self.host.gate.queue.remove(entry)


class RecoveryTests(Rig):
    def test_crash_restarts(self):
        first = self.llama_pid()
        with self.assertRaises(Exception):
            post(self.llama, "/__crash", {}, timeout=5)
        self.assertTrue(wait_for(lambda: self.host.recovered == 1 and self.host.state == "ready", 30))
        self.assertNotEqual(self.llama_pid(), first)
        failure = self.host.status()["recovery"]["lastFailure"]
        self.assertEqual((failure["kind"], failure["exitCode"]), ("crash", 1))
        self.assertIn("exit code 1", failure["reason"])

    def test_oom_classified_with_log_tail(self):
        with self.assertRaises(Exception):
            post(self.llama, "/__oom", {}, timeout=5)
        self.assertTrue(wait_for(lambda: self.host.failure is not None, 10))
        self.assertEqual(self.host.failure["kind"], "out of memory")
        self.assertTrue(any("out of memory" in line for line in self.host.failure["logTail"]))
        self.assertLessEqual(len(self.host.failure["logTail"]), 20)
        self.assertTrue(wait_for(lambda: self.host.state == "ready", 30))

    def test_restart_limit_then_failed_then_cleared(self):
        self.fail_file.write_text("x")  # every restart dies at startup
        with self.assertRaises(Exception):
            post(self.llama, "/__crash", {}, timeout=5)
        self.assertTrue(wait_for(lambda: self.host.state == "failed", 30))
        self.assertEqual(len(self.host.restarts), host.RESTART_LIMIT)
        status = self.host.status()
        self.assertIn("failed to allocate", " ".join(status["recovery"]["lastFailure"]["logTail"]))
        code, body = post(self.port, "/v1/messages", {"n": 1})
        self.assertEqual(code, 503)
        self.assertEqual(json.loads(body)["error"]["type"], "api_error")
        self.assertIn("hearthwork logs --server", body)
        code, body = post(self.port, "/v1/chat/completions", {"n": 1})
        self.assertEqual((code, json.loads(body)["error"]["type"]), (503, "server_error"))
        self.fail_file.unlink()
        self.assertEqual(self.host.request_load()[0], 200)  # `hearthwork start`
        self.assertTrue(wait_for(lambda: self.host.state == "ready", 30))
        self.assertEqual(post(self.port, "/v1/chat/completions", {"n": 1})[0], 200)

    def test_slow_health_is_not_restarted(self):
        pid = self.llama_pid()
        post(self.llama, "/__hang", {"seconds": 3})
        t, result = self.stream(count=30, delay=0.1)  # a stream is being served: never restart it
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        self.assertTrue(wait_for(lambda: self.host.last_health == "timeout", 10))
        time.sleep(2)  # many failed checks while the stream runs: activity resets the count, nothing restarts
        self.assertLess(self.host.health_fails, host.HEALTH_FAILS)
        self.assertEqual((self.host.state, self.host.recovered, self.llama_pid()), ("ready", 0, pid))
        t.join(20)
        self.assertIn("tok29", result["body"])

    def test_silent_and_idle_server_is_restarted(self):
        pid = self.llama_pid()
        post(self.llama, "/__hang", {"seconds": 3})  # no request at all during the whole window
        self.assertTrue(wait_for(lambda: self.host.recovered == 1, 30))
        self.assertNotEqual(self.llama_pid(), pid)
        self.assertEqual(self.host.failure["kind"], "timeout")

    def test_stream_that_ended_gets_no_error_event(self):
        result = post(self.port, "/v1/messages", {"stream": True, "count": 2, "final": "event: message_stop\ndata: {}\n\n", "die": True})
        self.assertIn("message_stop", result[1])
        self.assertNotIn("event: error", result[1])
        self.assertTrue(wait_for(lambda: self.host.recovered == 1, 30))

    def test_cancelled_request_is_not_retried(self):
        dead = free_port()
        gate = host.Gate(lambda: 2)
        calls = []

        def restart():
            calls.append(1)
            gate.cancel_all("cancelled by a switch")
            self.addCleanup(http_server(dead).shutdown)
            return True

        gate.on_refused = restart
        listener = harnesses.make_relay(dead, ("127.0.0.1", 0), None, gate=gate)
        threading.Thread(target=listener.serve_forever, daemon=True).start()
        self.addCleanup(listener.shutdown)
        status, body = post(listener.server_address[1], "/v1/chat/completions", {"n": 1}, timeout=30)
        self.assertEqual(status, 503)
        self.assertIn("cancelled by a switch", body)
        self.assertEqual(calls, [1])

    def test_load_deadline_kills_the_process(self):
        self.host.kill_server()
        os.environ["FAKE_LOAD_DELAY"] = "30"
        self.addCleanup(os.environ.pop, "FAKE_LOAD_DELAY", None)
        with mock.patch.object(host, "LOAD_DEADLINE", 1):
            with self.assertRaises(RuntimeError):
                self.host.load(self.model_a, None)
        self.assertIsNone(self.host.process)
        self.assertFalse((self.tmp / "server.json").exists())
        self.assertFalse(host.health(self.llama, 0.5)[0])

    def test_stopping_during_load_leaves_no_process(self):
        self.host.kill_server()
        os.environ["FAKE_LOAD_DELAY"] = "30"
        self.addCleanup(os.environ.pop, "FAKE_LOAD_DELAY", None)
        threading.Timer(1, self.host.stop_event.set).start()
        with self.assertRaises(RuntimeError):
            self.host.load(self.model_a, None)
        self.assertFalse((self.tmp / "server.json").exists())
        self.assertIsNone(self.host.process)

    def test_unload_survives_a_failing_stop(self):
        self.host.idle_minutes = 0
        with mock.patch.object(server, "stop", side_effect=RuntimeError("boom")):
            self.host.unload()
        self.assertEqual(self.host.state, "unloaded")
        self.assertIsNone(self.host.process)
        self.assertEqual(self.host.gate.hold, "loading model")
        self.assertFalse(host.health(self.llama, 0.5)[0])

    def test_models_and_other_posts_while_unloaded(self):
        self.host.unload()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("GET", "/v1/models")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["data"][0]["id"], "A.gguf")
        self.assertEqual(self.host.state, "unloaded")  # listing does not load the model
        status, body = post(self.port, "/v1/messages/count_tokens", {"n": 1}, timeout=60)  # not a generation: wakes and waits
        self.assertEqual(status, 200)
        self.assertEqual(self.host.state, "ready")

    def test_context_is_auto_unless_given_and_config_is_reread(self):
        self.assertIsNone(self.host.load_context)
        fresh = {"server": {"port": self.llama, "slots": 2, "marker": 1}}
        with mock.patch("hearthwork.onboard.load_config", lambda: fresh):
            self.host.unload()
            self.host.wake()
            self.assertTrue(wait_for(lambda: self.host.state == "ready", 30))
        self.assertEqual(self.host.config["server"].get("marker"), 1)
        self.assertIsNone(self.host.load_context)
        self.assertEqual(self.host.switch(str(self.model_b), 8192, False)[0], 202)
        self.assertTrue(wait_for(lambda: self.host.state == "ready" and self.host.model == "B", 30))
        self.assertEqual(self.host.load_context, 8192)
        self.assertEqual(self.host.switch(str(self.model_a), None, False)[0], 202)
        self.assertTrue(wait_for(lambda: self.host.state == "ready" and self.host.model == "A", 30))
        self.assertIsNone(self.host.load_context)

    def test_request_during_crash_is_served_after_restart(self):
        with self.assertRaises(Exception):
            post(self.llama, "/__crash", {}, timeout=5)
        status, body = post(self.port, "/v1/chat/completions", {"n": 1}, timeout=60)  # arrives while the server is dead
        self.assertEqual(status, 200)
        self.assertTrue(wait_for(lambda: self.host.recovered == 1))

    def test_not_retried_when_recovery_fails(self):
        self.fail_file.write_text("x")
        with self.assertRaises(Exception):
            post(self.llama, "/__crash", {}, timeout=5)
        status, body = post(self.port, "/v1/chat/completions", {"n": 1}, timeout=60)
        self.assertEqual(status, 503)
        self.assertIn("llama-server", body)

    def _interrupted(self, path):
        t, result = self.stream(path=path, count=100, delay=0.1)
        self.assertTrue(wait_for(lambda: self.host.gate.active))
        time.sleep(0.5)
        with self.assertRaises(Exception):
            post(self.llama, "/__crash", {}, timeout=5)
        t.join(20)
        return result["body"]

    def test_midstream_failure_anthropic(self):
        body = self._interrupted("/v1/messages")
        self.assertIn("tok0", body)
        tail = body[body.index("event: error"):]
        self.assertEqual(tail.count("event: error"), 1)
        self.assertEqual(json.loads(tail.split("data: ", 1)[1].strip())["error"]["type"], "api_error")
        self.assertTrue(tail.rstrip().endswith("}"))  # nothing spliced after the error
        self.assertTrue(wait_for(lambda: self.host.recovered == 1, 30))

    def test_midstream_failure_openai(self):
        body = self._interrupted("/v1/chat/completions")
        self.assertIn('"error"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))
        self.assertEqual(body.count("tok0"), 1)

    def test_midstream_failure_responses(self):
        body = self._interrupted("/v1/responses")
        self.assertIn("event: error", body)
        self.assertEqual(body.count("tok0"), 1)


class RetryTests(unittest.TestCase):
    """The relay retries once, and only when the connection was refused/reset before the request was accepted."""

    def relay(self, on_refused):
        self.dead = free_port()  # nothing listens here
        gate = host.Gate(lambda: 2)
        gate.on_refused = on_refused
        listener = harnesses.make_relay(self.dead, ("127.0.0.1", 0), None, gate=gate)
        threading.Thread(target=listener.serve_forever, daemon=True).start()
        self.addCleanup(listener.shutdown)
        return listener.server_address[1], gate

    def test_retried_once_after_a_restart(self):
        calls = []

        def restart():
            calls.append(1)
            fake = http_server(self.dead)
            self.addCleanup(fake.shutdown)
            return True

        port, gate = self.relay(restart)
        status, body = post(port, "/v1/chat/completions", {"n": 1}, timeout=30)
        self.assertEqual((status, calls), (200, [1]))
        self.assertEqual(gate.snapshot()["active"], 0)

    def test_no_second_retry_and_clear_error(self):
        calls = []
        port, gate = self.relay(lambda: calls.append(1) or False)
        status, body = post(port, "/v1/messages", {"n": 1}, timeout=30)
        self.assertEqual((status, calls), (503, [1]))
        self.assertEqual(json.loads(body)["error"]["type"], "api_error")


def http_server(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            data = b'{"ok": true}'
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class HelperTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(host.classify_exit(1, ["x", "ggml_cuda: failed to allocate 3 GiB"]), "out of memory")
        self.assertEqual(host.classify_exit(139, ["segfault"]), "crash")
        self.assertEqual(host.classify_error(ConnectionRefusedError()), "connection refused")
        self.assertEqual(host.classify_error(ConnectionResetError()), "connection reset")
        self.assertEqual(host.classify_error(socket.timeout("timed out")), "timeout")

    def test_status_lines(self):
        base = {"pid": 1, "port": 2, "uptime": 5, "active": 0, "queued": 0, "limit": 2, "clients": {}, "serverPort": 3,
                "slots": 2, "model": "m", "ready": False, "idleMinutes": 30}
        failed = dict(base, state="failed", recovery={"lastFailure": {"reason": "llama-server crashed (exit code 1)"}, "restarts": 3,
                                                       "limit": 3, "windowMinutes": 15, "recovered": 0})
        text = "\n".join(host.status_lines(failed))
        self.assertIn("failed: llama-server crashed", text)
        self.assertIn("idle unload: after 30 min", text)
        switching = dict(base, state="ready", transition={"kind": "switch", "target": "X", "phase": "draining", "draining": 2})
        self.assertIn("switching to X (draining 2)", "\n".join(host.status_lines(switching)))


if __name__ == "__main__":
    unittest.main()
