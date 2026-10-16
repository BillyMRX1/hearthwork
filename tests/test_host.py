"""The host: admission and queue, overload errors, disconnects, streaming, control token, accounting, one normalization."""
import http.client
import http.server
import json
import socket
import threading
import time
import unittest
from unittest import mock

from hearthwork import harnesses, host


class Upstream(http.server.ThreadingHTTPServer):
    """Stands in for llama-server: POSTs wait until `release` is set (or stream until the caller leaves)."""
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), self.Handler)
        self.release, self.go_on = threading.Event(), threading.Event()
        self.arrived, self.bodies, self.headers_seen = [], [], []
        self.write_failed, self.closed = threading.Event(), threading.Event()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_GET(self):
            data = b'{"data": [{"id": "m"}], "total_slots": 2}'
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            payload = json.loads(body or b"{}")
            self.server.arrived.append(payload.get("n"))
            self.server.bodies.append(payload)
            self.server.headers_seen.append(dict(self.headers.items()))
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: one\n\n")
                self.wfile.flush()
                self.server.go_on.wait(5)
                for _ in range(40):  # a caller that left shows as a failing write
                    try:
                        self.wfile.write(b"data: more\n\n")
                        self.wfile.flush()
                    except OSError:
                        self.server.write_failed.set()
                        return
                    time.sleep(0.1)
                return
            for _ in range(200):  # waiting for the model; a caller that shut the connection down shows as end of stream
                if self.server.release.is_set():
                    break
                if harnesses.client_gone(self.connection):
                    self.server.closed.set()
                    return
                time.sleep(0.05)
            data = json.dumps({"ok": payload.get("n")}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def serve(relay):
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    return relay.server_address[1]


def post(port, path, payload, headers=None, timeout=15):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(payload), headers={"content-type": "application/json", **(headers or {})})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def wait_for(condition, seconds=5):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class Base(unittest.TestCase):
    def setUp(self):
        self.upstream = Upstream()
        self.addCleanup(self.upstream.shutdown)
        self.addCleanup(self.upstream.server_close)

    def host(self, limit=1, **gate):
        self.gate = host.Gate(lambda: limit, **gate)
        relay = harnesses.make_relay(self.upstream.server_address[1], gate=self.gate)
        self.addCleanup(relay.shutdown)
        return serve(relay)

    def threads(self, port, count, path="/v1/chat/completions", headers=None):
        results = {}

        def run(n):
            results[n] = post(port, path, {"n": n}, headers)
        started = []
        for n in range(count):
            t = threading.Thread(target=run, args=(n,), daemon=True)
            t.start()
            started.append(t)
            # one at a time into the queue, so the order is the order of arrival
            wait_for(lambda: self.gate.snapshot()["active"] + self.gate.snapshot()["queued"] > n)
        return results, started


class QueueTests(Base):
    def test_limit_and_fifo_order(self):
        port = self.host(limit=1)
        results, started = self.threads(port, 3)
        snap = self.gate.snapshot()
        self.assertEqual((snap["active"], snap["queued"]), (1, 2))
        self.assertEqual(self.upstream.arrived, [0])
        self.upstream.release.set()
        for t in started:
            t.join(10)
        self.assertEqual(self.upstream.arrived, [0, 1, 2])  # first come, first served
        self.assertTrue(all(r[0] == 200 for r in results.values()))
        self.assertEqual(self.gate.snapshot()["active"], 0)

    def test_two_slots_run_two_at_once(self):
        port = self.host(limit=2)
        _, started = self.threads(port, 3)
        snap = self.gate.snapshot()
        self.assertEqual((snap["active"], snap["queued"]), (2, 1))
        self.upstream.release.set()
        for t in started:
            t.join(10)

    def test_overload_in_anthropic_format(self):
        port = self.host(limit=1, max_queue=1)
        _, started = self.threads(port, 2, "/v1/messages")
        status, headers, body = post(port, "/v1/messages", {"n": 9})
        self.assertEqual(status, 529)
        self.assertEqual(json.loads(body)["error"]["type"], "overloaded_error")
        self.assertEqual(json.loads(body)["type"], "error")
        self.upstream.release.set()
        for t in started:
            t.join(10)

    def test_overload_in_openai_format(self):
        port = self.host(limit=1, max_queue=1)
        _, started = self.threads(port, 2, "/v1/chat/completions")
        status, headers, body = post(port, "/v1/chat/completions", {"n": 9})
        self.assertEqual(status, 503)
        self.assertIn("Retry-After", headers)
        self.assertIn("message", json.loads(body)["error"])
        self.assertEqual(self.gate.snapshot()["rejected"], 1)
        self.upstream.release.set()
        for t in started:
            t.join(10)

    def test_queue_timeout(self):
        port = self.host(limit=1, max_wait=0.6)
        _, started = self.threads(port, 1)
        status, _, body = post(port, "/v1/messages", {"n": 5})
        self.assertEqual(status, 529)
        self.assertIn("waited", json.loads(body)["error"]["message"])
        self.assertEqual(self.gate.snapshot()["queued"], 0)
        self.upstream.release.set()
        started[0].join(10)

    def test_other_requests_skip_the_queue(self):
        port = self.host(limit=1)
        _, started = self.threads(port, 1)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/props")
        self.assertEqual(conn.getresponse().status, 200)
        self.upstream.release.set()
        self.assertEqual(post(port, "/v1/messages/count_tokens", {"n": 7})[0], 200)  # not generation: no slot needed
        started[0].join(10)


class DisconnectTests(Base):
    def raw(self, port, path, payload, headers=""):
        sock = socket.create_connection(("127.0.0.1", port))
        body = json.dumps(payload).encode()
        sock.sendall((f"POST {path} HTTP/1.1\r\nHost: x\r\ncontent-type: application/json\r\n{headers}"
                      f"content-length: {len(body)}\r\n\r\n").encode() + body)
        return sock

    def test_caller_leaves_while_queued(self):
        port = self.host(limit=1)
        _, started = self.threads(port, 1)
        sock = self.raw(port, "/v1/chat/completions", {"n": 1})
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["queued"] == 1))
        sock.close()
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["queued"] == 0))
        self.upstream.release.set()
        started[0].join(10)
        self.assertEqual(self.upstream.arrived, [0])  # the one that left never reached the model

    def test_streaming_passes_through_and_closing_closes_upstream(self):
        port = self.host(limit=1)
        sock = self.raw(port, "/v1/chat/completions", {"n": 1, "stream": True})
        sock.settimeout(5)
        got = b""
        while b"data: one" not in got:  # arrives while the upstream is still going: not buffered
            chunk = sock.recv(4096)
            self.assertTrue(chunk)
            got += chunk
        self.assertEqual(self.gate.snapshot()["active"], 1)  # a stream counts until it ends
        sock.close()
        self.assertTrue(self.upstream.write_failed.wait(8))
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["active"] == 0))

    def test_caller_leaves_while_waiting_for_the_model(self):
        port = self.host(limit=1)
        sock = self.raw(port, "/v1/chat/completions", {"n": 1})
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["active"] == 1))
        sock.close()
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["active"] == 0))  # the slot is free at once
        self.assertTrue(self.upstream.closed.wait(8))  # and the model server saw the connection close


class AccountingTests(Base):
    def test_clients_are_told_apart(self):
        port = self.host(limit=1)
        _, started = self.threads(port, 1, headers={"Authorization": "Bearer local-opencode-ab12"})
        t = threading.Thread(target=post, args=(port, "/v1/messages", {"n": 2}, {"x-api-key": "sk-secret"}), daemon=True)
        t.start()
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["queued"] == 1))
        clients = self.gate.snapshot()["clients"]
        self.assertEqual(clients["local-opencode-ab12"]["active"], 1)
        other = [k for k in clients if k.startswith("key-")]
        self.assertEqual(len(other), 1)  # a foreign token shows as a hash, never itself
        self.assertNotIn("sk-secret", json.dumps(clients))
        self.assertEqual(host.display_name("local-opencode-ab12"), "opencode (ab12)")
        self.upstream.release.set()
        t.join(10)
        started[0].join(10)

    def test_client_header_wins(self):
        self.assertEqual(host.client_id({"x-hearthwork-client": "share:laptop", "authorization": "Bearer local-x-1"}), "share:laptop")


class ControlTests(Base):
    def make(self):
        self.h = host.Host({"server": {"port": self.upstream.server_address[1], "slots": 2}}, lambda m: None)
        relay = harnesses.make_relay(self.upstream.server_address[1], intercept=self.h.control, gate=self.h.gate)
        self.addCleanup(relay.shutdown)
        return serve(relay)

    def get(self, port, path, token=None, method="GET"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(method, path, headers={"Authorization": f"Bearer {token}"} if token else {})
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"{}")

    def test_token_required(self):
        port = self.make()
        self.assertEqual(self.get(port, "/hearthwork/control/status")[0], 401)
        self.assertEqual(self.get(port, "/hearthwork/control/status", "wrong")[0], 401)
        self.assertEqual(self.get(port, "/hearthwork/control/stop", self.h.token, "POST")[0], 200)
        self.assertTrue(self.h.stop_event.is_set())

    def test_status_snapshot(self):
        port = self.make()
        status, data = self.get(port, "/hearthwork/control/status", self.h.token)
        self.assertEqual(status, 200)
        self.assertEqual((data["active"], data["queued"], data["limit"]), (0, 0, 2))
        self.assertNotIn("token", data)

    def test_other_hearthwork_paths_are_not_forwarded(self):
        port = self.make()
        self.assertEqual(self.get(port, "/hearthwork/info")[0], 404)

    def test_record_is_private_and_holds_the_token(self):
        import tempfile
        from pathlib import Path
        path = Path(tempfile.mkdtemp()) / "host.json"
        with mock.patch.object(host, "HOST_FILE", path):
            h = host.Host({"server": {"port": 1, "slots": 2}}, lambda m: None)
            h.record()
        data = json.loads(path.read_text())
        self.assertEqual(data["token"], h.token)
        if hasattr(__import__("os"), "getuid"):
            self.assertEqual(path.stat().st_mode & 0o077, 0)


class ForwardingTests(Base):
    def test_proxies_do_not_normalize_a_second_time(self):
        calls = []
        real = harnesses.normalize_chat

        def counting(body):
            calls.append(1)
            return real(body)
        self.gate = host.Gate(lambda: 2)
        with mock.patch.object(harnesses, "normalize_chat", counting):
            hostport = serve(harnesses.make_relay(self.upstream.server_address[1], gate=self.gate))
            proxy = serve(harnesses.make_relay(hostport, normalize=False,
                                               forward_headers=lambda h: {"X-Hearthwork-Client": "share:laptop"}))
            self.upstream.release.set()
            status, _, _ = post(proxy, "/v1/chat/completions", {"n": 1, "messages": [{"role": "user", "content": "hi"}]},
                                {"X-Hearthwork-Client": "spoofed"})
        self.assertEqual(status, 200)
        self.assertEqual(len(calls), 1)  # only the host normalizes
        self.assertIn("share:laptop", self.gate.snapshot()["clients"])
        self.assertNotIn("spoofed", self.gate.snapshot()["clients"])


class LoggerTests(unittest.TestCase):
    def test_rotates(self):
        import tempfile
        from pathlib import Path
        folder = Path(tempfile.mkdtemp())
        log = host.Logger(folder / "host.log", limit=200)
        for i in range(40):
            print(f"line {i} " + "x" * 20, file=log)
        self.assertTrue((folder / "host.log.1").exists())
        self.assertLess((folder / "host.log").stat().st_size, 600)


if __name__ == "__main__":
    unittest.main()
