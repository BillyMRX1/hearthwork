import os
import tempfile
import unittest
from unittest import mock

from hearthwork import mcp, task
from hearthwork.harnesses import clean_env, codex_command


class TaskTests(unittest.TestCase):
    def test_snapshot_diff(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in (("keep.txt", "a"), ("edit.txt", "b"), ("gone.txt", "c"), (".git/x", "z"), ("node_modules/y.js", "z")):
                path = os.path.join(folder, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").write(text)
            before = task.snapshot(folder)
            self.assertEqual(sorted(before), ["edit.txt", "gone.txt", "keep.txt"])
            open(os.path.join(folder, "edit.txt"), "w").write("longer")
            os.remove(os.path.join(folder, "gone.txt"))
            open(os.path.join(folder, "new.py"), "w").write("x = 1")
            created, modified, deleted = task.diff_snapshots(before, task.snapshot(folder))
            self.assertEqual(created, [{"path": "new.py", "size": 5}])
            self.assertEqual(modified, [{"path": "edit.txt", "size": 6}])
            self.assertEqual(deleted, [{"path": "gone.txt"}])

    def test_snapshot_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(5):
                open(os.path.join(folder, f"f{i}"), "w").close()
            self.assertEqual(len(task.snapshot(folder, limit=3)), 3)

    def test_claude_args(self):
        args = task.task_args("claude", "t", ["pytest *", "python *"])
        self.assertIn("-p", args)
        self.assertEqual(args[args.index("--permission-mode") + 1], "acceptEdits")
        tools = args[args.index("--allowedTools") + 1:]
        self.assertEqual(tools, ["Read", "Grep", "Glob", "Edit", "Write", "MultiEdit", "Bash(pytest *)", "Bash(python *)"])
        self.assertNotIn("t", args)  # the task goes in through stdin

    def test_codex_args_survive_command_building(self):
        args = task.task_args("codex", "t", [], "/tmp/last.txt")
        self.assertEqual(args[:2], ["exec", "--skip-git-repo-check"])
        self.assertEqual(args[-1], "-")
        with mock.patch("hearthwork.harnesses.codex_catalog", return_value="cat.json"):
            command, _ = codex_command("codex", "http://x:1", None, "m", 1000, 100, args)
        self.assertIn('sandbox_mode="workspace-write"', command)
        self.assertIn("model_provider=llamacpp", command)  # root -c settings kept next to the sandbox one

    def test_parse_claude(self):
        self.assertEqual(task.parse_result("claude", '{"result": "ok", "is_error": false}\n', "", 0), ("ok", None))
        message, error = task.parse_result("claude", '{"result": "bad", "is_error": true}', "", 1)
        self.assertIn("bad", error)
        self.assertIn("exited with 3", task.parse_result("claude", "garbage", "", 3)[1])

    def test_clean_env(self):
        env = clean_env({"CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CODE_SSE_PORT": "1", "CLAUDE_PID": "2",
                         "AI_AGENT": "x", "CODEX_SANDBOX": "seatbelt", "CODEX_THREAD_ID": "t", "CLAUDE_CONFIG_DIR": "/c",
                         "CODEX_HOME": "/h", "PATH": "/bin", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"})
        self.assertEqual(env, {"CLAUDE_CONFIG_DIR": "/c", "CODEX_HOME": "/h", "PATH": "/bin", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"})

    def test_exit_code(self):
        self.assertEqual(task.exit_code({"status": "ok"}), 0)
        self.assertEqual(task.exit_code({"status": "timeout"}), 124)
        self.assertEqual(task.exit_code({"status": "error"}), 1)


class McpTests(unittest.TestCase):
    def setUp(self):
        self.server = mcp.Server(slots=2, cwd=os.getcwd())

    def rpc(self, method, params=None, msg_id=1):
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        return self.server.handle(message)

    def test_initialize_negotiates(self):
        for version in mcp.VERSIONS:
            result = self.rpc("initialize", {"protocolVersion": version})["result"]
            self.assertEqual(result["protocolVersion"], version)
        self.assertEqual(self.rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"], mcp.VERSIONS[0])

    def test_notifications_get_no_answer(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/whatever", "params": {}}))

    def test_tools_list_and_ping(self):
        names = [t["name"] for t in self.rpc("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["local_task_start", "local_task_result", "local_task", "local_model_status"])
        self.assertEqual(self.rpc("ping")["result"], {})

    def test_unknown_method_and_bad_params(self):
        self.assertEqual(self.rpc("nope")["error"]["code"], -32601)
        self.assertEqual(self.rpc("tools/call", {"name": "nope"})["error"]["code"], -32602)
        self.assertEqual(self.server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": [1]})["error"]["code"], -32602)
        self.assertEqual(self.rpc("tools/call", {"name": "local_task_start", "arguments": "x"})["error"]["code"], -32602)

    def test_tool_argument_errors_are_results(self):
        for name, args in (("local_task_start", {}), ("local_task_start", {"task": "x", "agent": "gpt"}),
                           ("local_task_start", {"task": "x", "allow_commands": "pytest"}),
                           ("local_task_result", {"id": "nope"}), ("local_task_result", {})):
            result = self.rpc("tools/call", {"name": name, "arguments": args})["result"]
            self.assertTrue(result["isError"], name)

    def test_task_command(self):
        command, task_text = mcp.Tasks("/w").command({"task": "do it", "agent": "codex", "allow_commands": ["pytest *"], "timeout_seconds": 60})
        self.assertEqual(task_text, "do it")
        self.assertEqual(command[-1], "-")
        self.assertEqual(command[command.index("--agent") + 1], "codex")
        self.assertEqual(command[command.index("--allow") + 1], "pytest *")
        self.assertEqual(command[command.index("--cwd") + 1], "/w")

    def test_snippet(self):
        entry = mcp.snippet()["mcpServers"]["hearthwork"]
        self.assertTrue(os.path.isabs(entry["command"]))
        self.assertEqual(entry["args"][-1], "mcp")


if __name__ == "__main__":
    unittest.main()
