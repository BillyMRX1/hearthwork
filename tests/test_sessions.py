"""The session guard (#fix 1): clients register with the host, a switch refuses while sessions are open, `--force`
closes them, unregistering/stale pids clear them, and the control API carries it end to end with a real client process."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import harnesses, host

SRC = str(Path(host.__file__).resolve().parent.parent)  # so a child process can import hearthwork

CLIENT = """
import json, os, sys, time
from hearthwork import host
info = json.loads(sys.argv[1])
record = host.call(info, "session", "POST", payload={"action": "register", "client": sys.argv[2], "kind": sys.argv[3],
                                                     "pid": os.getpid()})
print(json.dumps(record), flush=True)
time.sleep(120)
"""


def wait_for(condition, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


class SessionsRegistryTests(unittest.TestCase):
    def test_register_open_unregister(self):
        s = host.Sessions(alive=lambda pid: True)
        record = s.register("local-claude-ab12", "claude", 4242)
        self.assertEqual((record["kind"], record["pid"]), ("claude", 4242))
        self.assertEqual([x["client"] for x in s.open()], ["local-claude-ab12"])
        self.assertTrue(s.unregister("local-claude-ab12"))
        self.assertEqual(s.open(), [])
        self.assertFalse(s.unregister("local-claude-ab12"))  # already gone

    def test_empty_client_is_ignored(self):
        s = host.Sessions(alive=lambda pid: True)
        self.assertIsNone(s.register("", "claude", 1))
        self.assertEqual(s.open(), [])

    def test_stale_pid_is_dropped(self):
        s = host.Sessions(alive=lambda pid: pid == 1)
        s.register("local-claude-live", "claude", 1)
        s.register("local-claude-dead", "claude", 2)
        self.assertEqual([x["client"] for x in s.open()], ["local-claude-live"])
        self.assertNotIn("local-claude-dead", s.items)

    def test_clear(self):
        s = host.Sessions(alive=lambda pid: True)
        s.register("local-claude-ab12", "claude", 1)
        s.register("local-task-cdef", "task", 2)
        s.clear()
        self.assertEqual(s.open(), [])

    def test_line_and_message_format(self):
        s = host.Sessions(alive=lambda pid: True)
        s.register("local-claude-ab12", "claude", 4242)
        self.assertRegex(host.session_line(s.open()[0]), r"^claude session since \d\d:\d\d:\d\d \(pid 4242\)$")
        self.assertIn("1 agent session is still using the model", host.sessions_message(s.open()))
        s.register("local-task-cdef", "task", 4243)
        message = host.sessions_message(s.open())
        self.assertIn("2 agent sessions are still using the model", message)
        self.assertIn("claude session since", message)
        self.assertIn("--force", message)


class SwitchGuardTests(unittest.TestCase):
    """switch() itself, without a model: the refusal happens before any work starts."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.model_a, self.model_b = self.tmp / "A.gguf", self.tmp / "B.gguf"
        for model in (self.model_a, self.model_b):
            model.write_text("x")

    def ready_host(self):
        h = host.Host({"server": {"port": 8001, "slots": 2}}, lambda message: None)
        h.ready, h.state, h.model_path = True, "ready", self.model_a
        return h

    def test_switch_refused_while_a_session_is_open(self):
        h = self.ready_host()
        h.sessions.register("local-claude-ab12", "claude", os.getpid())
        code, payload = h.switch(str(self.model_b), None, False)
        self.assertEqual(code, 409)
        message = payload["error"]["message"]
        self.assertIn("claude session since", message)
        self.assertIn(f"(pid {os.getpid()})", message)
        self.assertIn("--force", message)
        self.assertIsNone(h.transition)  # refused before a transition started
        self.assertEqual(h.state, "ready")

    def test_switch_proceeds_without_sessions(self):
        h = self.ready_host()
        with mock.patch.object(host.Host, "_switch"):  # the refusal is what we test here, not the reload
            self.assertEqual(h.switch(str(self.model_b), None, False)[0], 202)

    def test_stale_session_does_not_block(self):
        h = self.ready_host()
        h.sessions = host.Sessions(alive=lambda pid: False)
        h.sessions.register("local-claude-dead", "claude", 999999)
        with mock.patch.object(host.Host, "_switch"):
            self.assertEqual(h.switch(str(self.model_b), None, False)[0], 202)

    def test_force_switch_closes_the_sessions(self):
        h = self.ready_host()
        h.sessions.register("local-claude-ab12", "claude", os.getpid())
        self.assertTrue(h.begin("switch", "B"))
        with mock.patch.object(host.Host, "release_server"), mock.patch.object(host.Host, "load"), \
                mock.patch.object(host.Host, "set_ready"):
            h._switch(self.model_b, None, True)
        self.assertEqual(h.sessions.open(), [])


class MemoryReleaseTests(unittest.TestCase):
    """A restart waits for the stopped server's RAM to come back, but never longer than it must."""

    def host_with(self, frees):
        h = host.Host({"server": {"port": 8001, "slots": 2}}, lambda message: None)
        self.values = list(frees)
        return h

    def test_returns_once_the_target_is_reached(self):
        h = self.host_with([8.0, 20.0])
        with mock.patch("hearthwork.onboard.memory_gb", side_effect=lambda: (32.0, self.values.pop(0))):
            started = time.monotonic()
            h.wait_memory_released(previous_gb=12.0, free_before=7.0, timeout=30)
        self.assertLess(time.monotonic() - started, host.MEMORY_POLL * 3)
        self.assertEqual(self.values, [])  # the first check (20.0 >= 7 + 12*0.9) already sufficed

    def test_settles_early_when_nothing_more_comes_back(self):
        h = self.host_with([8.0, 8.0, 8.0, 8.0])
        with mock.patch("hearthwork.onboard.memory_gb", side_effect=lambda: (32.0, self.values.pop(0))):
            started = time.monotonic()
            h.wait_memory_released(previous_gb=12.0, free_before=8.0, timeout=60)
        self.assertLess(time.monotonic() - started, 15)  # two settles, not the full 60 s

    def test_nothing_to_wait_for_when_no_previous_model(self):
        h = host.Host({"server": {"port": 8001, "slots": 2}}, lambda message: None)
        with mock.patch("hearthwork.onboard.memory_gb", side_effect=AssertionError("must not measure")):
            h.wait_memory_released(previous_gb=None, free_before=None, timeout=60)
            h.wait_memory_released(previous_gb=0.5, free_before=8.0, timeout=60)


class ControlApiTests(unittest.TestCase):
    """The guard through the real control API, with a child process registering like prepare() does."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.model_a, self.model_b = self.tmp / "A.gguf", self.tmp / "B.gguf"
        for model in (self.model_a, self.model_b):
            model.write_text("x")
        self.h = host.Host({"server": {"port": 8001, "slots": 2}}, lambda message: None)
        self.h.ready, self.h.state, self.h.model_path = True, "ready", self.model_a
        self.listener = harnesses.make_relay(self.h.llama_port, ("127.0.0.1", 0), self.h.control, gate=self.h.gate)
        self.addCleanup(self.listener.shutdown)
        threading.Thread(target=self.listener.serve_forever, daemon=True).start()
        self.info = {"port": self.listener.server_address[1], "token": self.h.token}

    def client(self, client_id, kind):
        env = {**os.environ, "PYTHONPATH": SRC, "PYTHONIOENCODING": "utf-8"}
        return subprocess.Popen([sys.executable, "-c", CLIENT, json.dumps(self.info), client_id, kind],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)

    def test_register_refused_unregister_and_stale_cleanup(self):
        child = self.client("local-claude-live", "claude")
        self.addCleanup(child.wait)
        try:
            line = child.stdout.readline()  # the child prints the registration record before it sleeps
            self.assertIn("local-claude-live", line)
            self.assertTrue(wait_for(lambda: self.h.sessions.open(), 10))
            self.assertEqual(self.h.sessions.open()[0]["kind"], "claude")
            answer = host.call(self.info, "switch", "POST", payload={"model": str(self.model_b)}, errors=True)
            message = answer["error"]["message"]
            self.assertIn("claude session since", message)
            self.assertIn("(pid ", message)  # the child's own pid, not this test process
            self.assertNotIn(f"(pid {os.getpid()})", message)
        finally:
            child.kill()
            child.wait()
        self.assertTrue(wait_for(lambda: not self.h.sessions.open(), 10))  # the dead pid is dropped as stale

    def test_explicit_unregister_clears_it(self):
        answer = host.call(self.info, "session", "POST", payload={"action": "register", "client": "local-task-x", "kind": "task"})
        self.assertIsNotNone(answer["session"])
        self.assertEqual(len(self.h.sessions.open()), 1)
        gone = host.call(self.info, "session", "POST", payload={"action": "unregister", "client": "local-task-x"})
        self.assertIsNone(gone["session"])
        self.assertEqual(self.h.sessions.open(), [])
        self.assertEqual(host.call(self.info, "session", "POST", payload={"action": "bogus"}, errors=True)
                         .get("error", {}).get("message"), "action must be register or unregister")


class ClientHelperTests(unittest.TestCase):
    """host.session() / Session: registration is best effort and a no-op without a host, and close() is idempotent."""

    def test_no_host_registers_nothing(self):
        with mock.patch.object(host, "read_info", return_value=None):
            session = host.session("local-claude-ab12", "claude")
            self.assertFalse(session.registered)
            self.assertFalse(session.closed)  # nothing was opened, so nothing is closed either
            session.close()
            session.close()  # idempotent

    def test_session_deregisters_on_close(self):
        calls = []
        with mock.patch.object(host, "read_info", return_value={"port": 1, "token": "t"}), \
                mock.patch.object(host, "call", side_effect=lambda *a, **k: calls.append((a, k)) or {"session": {"client": "c"}}), \
                mock.patch.object(host, "_track_session"):
            session = host.session("local-claude-ab12", "claude")
            self.assertTrue(session.registered)
            session.close()
        self.assertTrue(session.closed)
        seen = " ".join(str(call) for call in calls)
        self.assertIn("local-claude-ab12", seen)
        self.assertIn("register", seen)
        self.assertIn("unregister", seen)


if __name__ == "__main__":
    unittest.main()