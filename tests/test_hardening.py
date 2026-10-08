import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import api, harnesses, mcp, procs, server, task
from hearthwork.harnesses import HARNESSES

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def sleeping_tasks(slots):
    tasks = mcp.Tasks(os.getcwd(), slots)
    tasks.command = lambda args: (SLEEP, "x")
    return tasks


def wait_for(condition, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


class QueueTests(unittest.TestCase):
    def test_queue_cancel_and_shutdown(self):
        tasks = sleeping_tasks(2)
        self.addCleanup(tasks.shutdown)
        a, b, c = (tasks.start({}) for _ in range(3))
        self.assertTrue(wait_for(lambda: all(tasks.items[i]["process"] for i in (a, b))))
        self.assertEqual(tasks.result(a, 0)["status"], "running")
        queued = tasks.result(c, 0)
        self.assertEqual((queued["status"], queued["position"]), ("queued", 1))
        self.assertIsNone(tasks.items[c]["process"])  # never more than 2 at once
        self.assertEqual(tasks.cancel(a)["status"], "cancelled")
        self.assertEqual(tasks.result(a, 0)["status"], "cancelled")
        self.assertTrue(wait_for(lambda: tasks.items[c]["process"] is not None))  # a slot freed: the third starts
        self.assertEqual(len(tasks.running()), 2)
        process = tasks.items[b]["process"]
        tasks.shutdown()
        self.assertTrue(wait_for(lambda: process.poll() is not None))
        self.assertTrue(wait_for(lambda: tasks.items[c]["process"].poll() is not None))
        with self.assertRaises(ValueError):
            tasks.start({})

    def test_cancel_queued_and_unknown(self):
        tasks = sleeping_tasks(1)
        self.addCleanup(tasks.shutdown)
        first, second = tasks.start({}), tasks.start({})
        self.assertEqual(tasks.cancel(second), {"id": second, "status": "cancelled", "was": "queued"})
        self.assertEqual(tasks.queue, [])
        with self.assertRaises(ValueError):
            tasks.cancel("nope")
        tasks.cancel(first)

    def test_cancel_tool(self):
        srv = mcp.Server(slots=1, cwd=os.getcwd())
        srv.tasks = sleeping_tasks(1)
        self.addCleanup(srv.tasks.shutdown)
        task_id = srv.tasks.start({})
        reply = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "local_task_cancel", "arguments": {"id": task_id}}})
        self.assertFalse(reply["result"]["isError"])
        self.assertIn("cancelled", reply["result"]["content"][0]["text"])

    def test_slots_for_local_and_remote(self):
        self.assertEqual(procs.parallel_slots({"server": {"slots": 4}}), 4)
        self.assertEqual(procs.parallel_slots({}), 2)
        self.assertEqual(procs.parallel_slots({"remote": {"host": "h"}}), 2)
        self.assertEqual(procs.parallel_slots({"remote": {"slots": 3}, "server": {"slots": 9}}), 3)
        self.assertEqual(procs.parallel_slots({"remote": {"info": {"slots": 5}}}), 5)
        self.assertEqual(procs.parallel_slots({"server": {"slots": "x"}}), 2)


class RegistryTests(unittest.TestCase):
    def test_dead_pids_are_cleaned_and_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for pid in ("111", "222", "junk"):
                (directory / pid).write_text("")
            self.assertEqual(procs.running_tasks(directory, alive=lambda pid: pid == 111), [111])
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ["111"])
        self.assertIsNone(procs.overload_warning(1, 2))
        self.assertIn("2 other", procs.overload_warning(2, 2))
        self.assertTrue(procs.pid_alive(os.getpid()))
        self.assertFalse(procs.pid_alive(0))


class CorsTests(unittest.TestCase):
    def test_origin_decisions(self):
        self.assertTrue(api.origin_allowed(None, None))
        self.assertTrue(api.origin_allowed("http://localhost:3000", None))
        self.assertTrue(api.origin_allowed("http://127.0.0.1:5173", None))
        self.assertFalse(api.origin_allowed("https://evil.example", None))
        self.assertFalse(api.origin_allowed("http://localhost.evil.example", None))
        self.assertFalse(api.origin_allowed("null", None))
        self.assertTrue(api.origin_allowed("https://evil.example", "key"))

    def test_headers(self):
        self.assertEqual(dict(api.cors_headers("http://localhost:3000", None))["Access-Control-Allow-Origin"], "http://localhost:3000")
        self.assertEqual(dict(api.cors_headers("https://app.example", "k"))["Access-Control-Allow-Origin"], "*")
        self.assertEqual(api.cors_headers("https://evil.example", None), [])
        self.assertEqual(api.cors_headers(None, "k"), [])
        pre = dict(api.preflight_headers("http://localhost:1", None, "x-custom"))
        self.assertEqual(pre["Access-Control-Allow-Headers"], "x-custom")
        self.assertIn("OPTIONS", pre["Access-Control-Allow-Methods"])

    def intercept(self, command, path, headers, key=None):
        class Handler:
            pass
        handler = Handler()
        handler.command, handler.path, handler.headers = command, path, api_headers(headers)
        handler.sent, handler.response = [], []
        handler._send_json = lambda status, body: handler.sent.append((status, body))
        handler.send_response = lambda status: handler.response.append(("status", status))
        handler.send_header = lambda n, v: handler.response.append((n, v))
        handler.end_headers = lambda: None
        instance = api.Api({"server": {"port": 1}, "modelsDir": "."}, 0, key)
        return instance.intercept(handler), handler

    def test_preflight_and_embeddings(self):
        done, handler = self.intercept("OPTIONS", "/v1/chat/completions", {"Origin": "http://localhost:3000"})
        self.assertTrue(done)
        self.assertIn(("status", 204), handler.response)
        done, handler = self.intercept("OPTIONS", "/v1/chat/completions", {"Origin": "https://evil.example"})
        self.assertEqual(handler.sent[0][0], 403)
        done, handler = self.intercept("OPTIONS", "/v1/chat/completions", {"Origin": "https://evil.example"}, key="k")
        self.assertIn(("status", 204), handler.response)  # no key on a preflight: browsers never send one
        done, handler = self.intercept("POST", "/v1/embeddings", {"authorization": "Bearer k"}, key="k")
        self.assertEqual(handler.sent[0][0], 501)
        self.assertIn("embeddings not enabled", handler.sent[0][1]["error"]["message"])
        done, handler = self.intercept("POST", "/v1/embeddings", {"Origin": "https://evil.example"})
        self.assertEqual(handler.sent[0][0], 403)  # a web page cannot use a key-less API


def api_headers(values):
    lowered = {k.lower(): v for k, v in values.items()}

    class H(dict):
        def get(self, key, default=None):
            return lowered.get(key.lower(), default)
    return H(lowered)


class ServerCwdTests(unittest.TestCase):
    def test_background_server_cwd_is_the_data_dir(self):
        with mock.patch.object(server.subprocess, "Popen") as popen:
            server.popen_server(["llama-server"])
        self.assertEqual(popen.call_args.kwargs["cwd"], str(server.HOME))

    def test_breakaway_falls_back(self):
        calls = []

        def fake(cmd, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise PermissionError("no breakaway")
            return mock.Mock()
        with mock.patch.object(server.subprocess, "Popen", side_effect=fake):
            server.popen_server(["llama-server"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["cwd"], str(server.HOME))

    @unittest.skipIf(os.name == "nt", "POSIX branch")
    def test_posix_branch(self):
        config = {"server": {"port": 1, "slots": 1}}
        with mock.patch.object(server, "stop"), mock.patch.object(server.ctx, "choose", return_value=(1, "s", None)), \
                mock.patch.object(server.ctx, "describe", return_value=""), mock.patch.object(server, "command", return_value=["x"]), \
                mock.patch.object(server, "WINDOWS", False), mock.patch.object(server, "ready", return_value=True), \
                mock.patch.object(server, "restore_slots", return_value=0), mock.patch.object(server, "STATE"), \
                mock.patch.object(server, "LOG", Path(tempfile.gettempdir()) / "hw-test.log"), \
                mock.patch.object(server.subprocess, "Popen") as popen:
            server.start_background(config, Path("m.gguf"))
        self.assertEqual(popen.call_args.kwargs["cwd"], str(server.HOME))


class ResultTests(unittest.TestCase):
    def test_every_agent_survives_empty_and_garbled_output(self):
        for agent, entry in HARNESSES.items():
            for out, err in (("", ""), (None, None), ("\x00�{{{ [{", "boom"), ("[]", "")):
                message, error = entry["result"](out, err, 0, "/does/not/exist")
                self.assertIsInstance(message, str, agent)
                self.assertEqual(message, message.strip(), agent)
                if agent != "aider" or not (out or "").strip("\x00� "):
                    self.assertTrue(error or not message, agent)

    def test_claude_takes_the_last_result_event(self):
        out = '{"type":"system"}\n{"type":"result","result":"  done \\u00d7 "}\nnoise\n'
        self.assertEqual(harnesses.result_claude(out, "", 0, None), ("done ×", None))

    def test_no_result_status(self):
        config = {"server": {"port": 1, "slots": 2}}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(task, "installed", return_value=True), \
                mock.patch.object(task, "model_for_task", return_value=(1, "m", 1000, None)), \
                mock.patch.object(task, "prepare", return_value=(["x"], {})), \
                mock.patch.object(task, "run_process", return_value=(0, '{"result": ""}', "", False)):
            report = task.run_task(config, "t", "claude", cwd=folder)
        self.assertEqual(report["status"], "no_result")
        self.assertEqual(task.exit_code(report), 1)
        self.assertIn("no result", report["error"])


class EncodingTests(unittest.TestCase):
    def test_utf8_stream_and_json(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        self.assertFalse(procs.utf8_stream(stream))
        self.assertTrue(procs.utf8_stream(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")))
        text = json.dumps({"m": "9×9 é 日本"}, ensure_ascii=True)
        self.assertEqual(json.loads(text)["m"], "9×9 é 日本")

    def test_setup_stdio_makes_pipes_utf8(self):
        buffer = io.BytesIO()
        fake = io.TextIOWrapper(buffer, encoding="cp1252")
        with mock.patch.object(sys, "stdout", fake), mock.patch.object(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252")), \
                mock.patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(), encoding="cp1252")):
            procs.setup_stdio()
            print("9×9", flush=True)
            self.assertEqual(buffer.getvalue().decode("utf-8").strip(), "9×9")


if __name__ == "__main__":
    unittest.main()
