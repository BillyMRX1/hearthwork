"""Host hardening: one host per computer, bounded waits, slot release, disconnect detection, stop with requests."""
import contextlib
import http.client
import io
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import harnesses, host

from test_host import Base, post, serve, wait_for


class SlotLeakTests(Base):
    def test_failure_before_the_upstream_request_releases_the_slot(self):
        self.gate = host.Gate(lambda: 1)

        def boom(handler):
            raise RuntimeError("header building failed")
        port = serve(harnesses.make_relay(self.upstream.server_address[1], gate=self.gate, forward_headers=boom))
        try:
            post(port, "/v1/chat/completions", {"n": 1}, timeout=5)
        except (OSError, http.client.HTTPException):
            pass
        self.assertTrue(wait_for(lambda: self.gate.snapshot()["active"] == 0))


class ClientGoneTests(unittest.TestCase):
    def pair(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        client = socket.create_connection(server.getsockname())
        accepted, _ = server.accept()
        self.addCleanup(server.close)
        self.addCleanup(accepted.close)
        return client, accepted

    def test_open_connection_is_not_gone(self):
        client, accepted = self.pair()
        self.addCleanup(client.close)
        self.assertFalse(harnesses.client_gone(accepted))
        client.sendall(b"x")  # readable with data: still there
        time.sleep(0.1)
        self.assertFalse(harnesses.client_gone(accepted))

    def test_close_and_reset_are_gone(self):
        client, accepted = self.pair()
        client.close()
        self.assertTrue(wait_for(lambda: harnesses.client_gone(accepted)))
        client, accepted = self.pair()
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()  # RST
        self.assertTrue(wait_for(lambda: harnesses.client_gone(accepted)))

    def test_selector_errors_mean_unknown_not_gone(self):
        client, accepted = self.pair()
        self.addCleanup(client.close)
        with mock.patch("selectors.DefaultSelector", side_effect=ValueError("filedescriptor out of range in select()")):
            self.assertFalse(harnesses.client_gone(accepted))
        with mock.patch("selectors.DefaultSelector", side_effect=OSError("boom")):
            self.assertFalse(harnesses.client_gone(accepted))


class OneHostTests(unittest.TestCase):
    def test_second_host_cannot_take_the_lock(self):
        lock = Path(tempfile.mkdtemp()) / "host.lock"
        self.assertTrue(host.acquire_lock(lock))
        code = "import sys; from hearthwork import host; sys.exit(0 if host.acquire_lock(sys.argv[1]) else 7)"
        other = subprocess.run([sys.executable, "-c", code, str(lock)], env=dict(os.environ))
        self.assertEqual(other.returncode, 7)  # exactly one host; the second leaves at once

    def test_second_host_exits_without_touching_the_model(self):
        with mock.patch.object(host, "acquire_lock", return_value=False), \
                mock.patch("hearthwork.server.stop") as stop, mock.patch("hearthwork.server.spawn_server") as spawn:
            self.assertEqual(host.serve({"server": {"port": 1, "slots": 1}}, None), 1)
        stop.assert_not_called()
        spawn.assert_not_called()

    def test_port_cannot_be_bound_twice(self):
        first = harnesses.make_relay(1, ("127.0.0.1", 0), exclusive=True)
        self.addCleanup(first.server_close)
        with self.assertRaises(OSError):
            harnesses.make_relay(1, ("127.0.0.1", first.server_address[1]), exclusive=True)


class WaitReadyTests(unittest.TestCase):
    INFO = {"pid": 1, "port": 1, "token": "t"}

    def run_wait(self, infos, calls, timeout=host.LOAD_TIMEOUT, clock=None):
        infos, calls = iter(infos), iter(calls)
        patches = [mock.patch.object(host, "read_info", side_effect=lambda: next(infos, None)),
                   mock.patch.object(host, "call", side_effect=lambda *a, **k: next(calls, None)),
                   mock.patch.object(host.time, "sleep")]
        if clock:
            patches.append(mock.patch.object(host.time, "time", side_effect=clock))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return host._wait_ready("x", None, timeout)

    def test_host_disappearing_stops_the_wait(self):
        self.assertFalse(self.run_wait([self.INFO, None], [{"ready": False}]))  # `hearthwork stop` in another terminal

    def test_never_ready_times_out(self):
        ticks = iter(range(0, 100000, 100))
        self.assertFalse(self.run_wait([self.INFO] * 100, [{"ready": False}] * 100, timeout=500, clock=lambda: next(ticks)))

    def test_pid_that_is_not_a_host_is_given_up_on(self):
        ticks = iter(range(0, 100000, 30))
        self.assertFalse(self.run_wait([self.INFO] * 100, [], clock=lambda: next(ticks)))

    def test_ready_returns_true(self):
        self.assertTrue(self.run_wait([self.INFO], [{"ready": True, "restored": 0}]))


class StopTests(unittest.TestCase):
    def setUp(self):
        self.status = {"pid": 5, "port": 1, "model": "m", "context": 1, "ready": True, "error": None, "adopted": False,
                       "restored": 0, "serverPort": 2, "uptime": 5, "slots": 2, "limit": 2, "active": 1, "queued": 1,
                       "rejected": 0, "clients": {"local-opencode-ab12": {"active": 1, "queued": 1, "total": 2}}}
        self.calls = []
        file = Path(tempfile.mkdtemp()) / "host.json"
        for p in (mock.patch.object(host, "HOST_FILE", file), mock.patch.object(host, "STATE", file),
                  mock.patch.object(host, "read_info", return_value={"pid": 5, "port": 1, "token": "t"}),
                  mock.patch.object(host, "call", side_effect=lambda info, path, method="GET", timeout=5: (self.calls.append(path) or self.status)),
                  mock.patch.object(host, "_alive", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def test_refuses_while_requests_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(host.stop({}))
        self.assertEqual(self.calls, ["status"])  # never told the host to stop
        self.assertIn("opencode (ab12)", out.getvalue())
        self.assertIn("--force", out.getvalue())

    def test_force_stops(self):
        self.assertTrue(host.stop({}, quiet=True, force=True))
        self.assertIn("stop", self.calls)

    def test_idle_stops_without_force(self):
        self.status.update(active=0, queued=0)
        self.assertTrue(host.stop({}, quiet=True))
        self.assertIn("stop", self.calls)


class DrainTests(unittest.TestCase):
    def test_close_refuses_new_and_waits_for_running(self):
        gate = host.Gate(lambda: 1)
        ticket = gate.admit(socket.socket(), "a")
        gate.close()
        with self.assertRaises(harnesses.Overloaded):
            gate.admit(socket.socket(), "b")
        self.assertFalse(gate.wait_idle(0.3))
        threading.Timer(0.2, ticket.release).start()
        self.assertTrue(gate.wait_idle(5))


if __name__ == "__main__":
    unittest.main()
