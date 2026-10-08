import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from hearthwork import harnesses, mcp, task
from hearthwork.harnesses import HARNESSES

BASE = "http://127.0.0.1:5555"


class Home:
    """Points the generated files at a temporary folder."""

    def __enter__(self):
        self.folder = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(harnesses, "HOME", __import__("pathlib").Path(self.folder.name))
        self.patch.start()
        return self.folder.name

    def __exit__(self, *exc):
        self.patch.stop()
        self.folder.cleanup()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(key, token=None, args=()):
    return HARNESSES[key]["command"](key, BASE, token, "model-x", 65536, 4096, list(args))


class RegistryTests(unittest.TestCase):
    def test_every_entry_is_complete(self):
        for key, entry in HARNESSES.items():
            for field in ("title", "binary", "version", "install", "api", "connects", "command", "task", "prompt", "result", "headless"):
                self.assertIn(field, entry, f"{key}.{field}")
            self.assertIn(entry["api"], harnesses.API_NAMES)
            self.assertIn(entry["prompt"], ("stdin", "file"))

    def test_mcp_and_task_know_every_agent(self):
        self.assertEqual(mcp.TASK_PROPERTIES["agent"]["enum"], list(HARNESSES))
        mcp.Tasks("/w").command({"task": "x", "agent": "opencode"})
        with self.assertRaises(ValueError):
            mcp.Tasks("/w").command({"task": "x", "agent": "nope"})

    def test_opencode(self):
        command, env = build("opencode", token="secret")
        self.assertEqual(command, ["opencode", "-m", "hearthwork/model-x"])
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["hearthwork"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"]["baseURL"], BASE + "/v1")
        self.assertEqual(provider["models"]["model-x"]["limit"], {"context": 65536, "output": 4096})
        self.assertEqual(env["HEARTHWORK_KEY"], "secret")
        self.assertNotIn("secret", env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(build("opencode", args=["run", "hi"])[0], ["opencode", "run", "-m", "hearthwork/model-x", "hi"])

    def test_aider(self):
        with Home() as home:
            command, env = build("aider", args=["--message", "hi"])
            self.assertEqual(env["OPENAI_API_BASE"], BASE + "/v1")
            self.assertEqual(command[command.index("--model") + 1], "openai/model-x")
            self.assertEqual(command[-2:], ["--message", "hi"])
            for flag in ("--no-show-model-warnings", "--no-check-update", "--no-analytics"):
                self.assertIn(flag, command)
            self.assertNotIn("--analytics-disable", command)  # that one writes into ~/.aider
            metadata = load(command[command.index("--model-metadata-file") + 1])
            self.assertEqual(metadata["openai/model-x"]["max_input_tokens"], 65536)
            settings = load(command[command.index("--model-settings-file") + 1])
            self.assertEqual(settings[0]["extra_params"], {"max_tokens": 4096})
            self.assertTrue(command[command.index("--model-metadata-file") + 1].startswith(home))

    def test_qwen(self):
        with Home():
            command, env = build("qwen", token="k", args=["-p", "hi"])
            self.assertEqual(command, ["qwen", "--model", "model-x", "-p", "hi"])
            self.assertEqual((env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], env["OPENAI_MODEL"]), (BASE + "/v1", "k", "model-x"))
            settings = load(env["QWEN_CODE_SYSTEM_SETTINGS_PATH"])
            self.assertEqual(settings["model"]["generationConfig"]["contextWindowSize"], 65536)
            self.assertEqual(settings["model"]["generationConfig"]["samplingParams"]["max_tokens"], 4096)

    def test_task_arguments(self):
        args = task.task_args("opencode", "t", ["python *"])
        self.assertEqual(args[:3], ["run", "--format", "json"])
        permission = json.loads(HARNESSES["opencode"]["task"](["python *"], None, None)[1]["OPENCODE_PERMISSION"])
        self.assertEqual(permission["bash"], {"*": "deny", "python *": "allow"})
        self.assertIn("--message-file", task.task_args("aider", "t", ["python *"], None, "p.txt"))
        self.assertIn("--no-suggest-shell-commands", task.task_args("aider", "t", ["python *"], None, "p.txt"))
        qwen = task.task_args("qwen", "t", ["python *", "pytest"])
        self.assertEqual(qwen[qwen.index("--allowed-tools") + 1:], ["run_shell_command(python)", "run_shell_command(pytest)"])
        self.assertIn("--continue", HARNESSES["qwen"]["task"]([], None, None, resume=True)[0])
        self.assertNotIn("--allowed-tools", task.task_args("qwen", "t", []))

    def test_opencode_resumes_its_own_session(self):
        events = json.dumps({"type": "step_start", "sessionID": "ses_abc"}) + "\nnot json\n"
        self.assertEqual(HARNESSES["opencode"]["session"](events), "ses_abc")
        task_args = HARNESSES["opencode"]["task"]
        self.assertEqual(task_args([], None, None, resume="ses_abc")[0][-2:], ["--session", "ses_abc"])
        self.assertNotIn("--session", task_args([], None, None)[0])  # --continue would take another folder's session

    def test_parse_results(self):
        events = "\n".join(json.dumps(e) for e in [
            {"type": "text", "part": {"text": "I will create it."}},
            {"type": "tool_use", "part": {"tool": "write"}},
            {"type": "text", "part": {"text": "Done."}}])
        self.assertEqual(task.parse_result("opencode", events, "", 0), ("Done.", None))
        self.assertIn("failed", task.parse_result("opencode", '{"type": "error", "error": {"data": {"message": "boom"}}}', "", 1)[1])
        qwen = json.dumps([{"type": "assistant"}, {"type": "result", "is_error": False, "result": "ok"}])
        self.assertEqual(task.parse_result("qwen", "warning\n" + qwen, "", 0), ("ok", None))
        self.assertIn("exited with 2", task.parse_result("qwen", "garbage", "", 2)[1])
        self.assertEqual(task.parse_result("aider", " reply ", "", 0), ("reply", None))
        self.assertIn("exited with 1", task.parse_result("aider", "x", "", 1)[1])


class DryRunTests(unittest.TestCase):
    def preview(self, key, remote=None, **kwargs):
        out = io.StringIO()
        with Home() as home, mock.patch.object(harnesses, "installed", return_value=None), contextlib.redirect_stdout(out):
            harnesses.preview(key, "model-x", 65536, remote=remote, **kwargs)
            self.assertEqual(os.listdir(home), [])  # a preview writes nothing
        return out.getvalue()

    def test_preview_has_command_env_and_files_without_secrets(self):
        remote = {"host": "10.0.0.5", "port": 8484, "key": "super-secret-key"}
        for key in HARNESSES:
            text = self.preview(key, remote)
            self.assertIn("command:", text)
            self.assertIn("NOT installed", text)
            self.assertNotIn("super-secret-key", text, key)
        text = self.preview("codex", remote)
        self.assertIn("HEARTHWORK_KEY=********", text)
        self.assertIn("http://10.0.0.5:8484/v1", text)

    def test_preview_uses_placeholder_relay(self):
        text = self.preview("qwen")
        self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:<relay-port>/v1", text)
        self.assertIn("OPENAI_API_KEY=********", text)
        self.assertIn("qwen-settings-model-x.json", text)
        self.assertIn('"contextWindowSize": 65536', text)
        self.assertIn("MAX_OUTPUT_TOKENS=4096", self.preview("claude"))  # a count, not a secret

    def test_preview_does_not_start_anything(self):
        with mock.patch.object(harnesses, "start_relay") as relay, mock.patch.object(harnesses.subprocess, "run") as run:
            self.preview("codex")
            relay.assert_not_called()
            run.assert_not_called()

    def test_mask_rule(self):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "HEARTHWORK_KEY"):
            self.assertTrue(harnesses.SECRET.search(name), name)
        self.assertFalse(harnesses.SECRET.search("CLAUDE_CODE_MAX_OUTPUT_TOKENS"))

    def test_agents_table(self):
        with mock.patch.object(harnesses, "installed", side_effect=lambda k: "/bin/x" if k == "aider" else None), \
                mock.patch.object(harnesses, "agent_version", return_value="9.9.9"):
            text = harnesses.agents_table()
        aider = next(line for line in text.splitlines() if line.startswith("aider"))
        self.assertIn("9.9.9", aider)
        self.assertIn("OpenAI Chat Completions", aider)
        self.assertIn("Not installed:", text)
        self.assertIn("npm i -g opencode-ai", text)
        self.assertNotIn("uv tool install --python 3.12 aider-chat", text)  # installed ones get no hint


if __name__ == "__main__":
    unittest.main()
