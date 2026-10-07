import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import bench


class SummaryTable(unittest.TestCase):
    def test_rows(self):
        table = bench.summary_table("Qwen", [("Claude Code", {"score": 7.5, "seconds": 150}), ("Codex", None)])
        lines = table.splitlines()
        self.assertEqual(lines[0], "| Model | Agent | Score | Time |")
        self.assertEqual(lines[2], "| Qwen | Claude Code | 7.5/8 | 2.5 min |")
        self.assertIn("failed", lines[3])


class IsolatedEnv(unittest.TestCase):
    def test_uses_given_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bench.isolated_env("codex", Path(tmp) / "cfg")
            self.assertEqual(env, {"CODEX_HOME": str(Path(tmp) / "cfg")})
            self.assertTrue((Path(tmp) / "cfg").is_dir())

    def test_refuses_real_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(Path, "home", return_value=Path(tmp)):
                with self.assertRaises(SystemExit):
                    bench.isolated_env("claude", Path(tmp) / ".claude")
                with self.assertRaises(SystemExit):
                    bench.isolated_env("claude", Path(tmp) / ".claude" / "sub")

    def test_refuses_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"CODEX_HOME": tmp}):
                with self.assertRaises(SystemExit):
                    bench.isolated_env("codex", Path(tmp))


class WindowsSandboxConfig(unittest.TestCase):
    def test_only_windows_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('model = "x"\n[tui]\na = 1\n[windows]\nsandbox = "elevated"\n[desktop]\nb = 2\n', encoding="utf-8")
            self.assertEqual(bench.windows_sandbox_config(path), '[windows]\nsandbox = "elevated"\n')
            self.assertEqual(bench.windows_sandbox_config(Path(tmp) / "missing.toml"), "")


class Retry(unittest.TestCase):
    def run_agent(self, outcomes):
        saved = []
        results = iter(outcomes)

        def fake(*args):
            outcome = next(results)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch.object(bench, "run_once", fake), mock.patch.object(bench, "save", saved.append):
            return bench.run_agent("codex", {}, "m", Path("x")), saved

    def test_retry_then_success(self):
        record, saved = self.run_agent([bench.AgentError("boom"), {"score": 6, "seconds": 1}])
        self.assertEqual(record["score"], 6)
        self.assertEqual(len(saved), 2)
        self.assertIn("error", saved[0])
        self.assertNotIn("score", saved[0])

    def test_two_failures(self):
        record, saved = self.run_agent([bench.AgentError("a"), bench.AgentError("b")])
        self.assertIsNone(record)
        self.assertEqual([s["attempt"] for s in saved], [1, 2])


if __name__ == "__main__":
    unittest.main()
