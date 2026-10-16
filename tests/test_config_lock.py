import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import budgets, onboard, setups

SRC = str(Path(__file__).resolve().parent.parent / "src")
WORKER = """
import sys
from hearthwork import onboard
key = sys.argv[1]
for i in range(200):
    onboard.update_config(lambda c: c.__setitem__(key, c.get(key, 0) + 1))
"""


class ConfigWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.patch = mock.patch.object(onboard, "CONFIG", self.home / "config.json")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_two_processes_lose_no_updates(self):
        (self.home / "config.json").write_text(json.dumps({"keep": 1}))
        env = dict(os.environ, HEARTHWORK_HOME=str(self.home), PYTHONPATH=SRC)
        procs = [subprocess.Popen([sys.executable, "-c", WORKER, key], env=env) for key in ("a", "b")]
        for p in procs:
            self.assertEqual(p.wait(timeout=120), 0)
        data = json.loads((self.home / "config.json").read_text())
        self.assertEqual((data["a"], data["b"], data["keep"]), (200, 200, 1))

    def test_save_config_writes_only_the_callers_changes(self):
        onboard.update_config(lambda c: c.update({"server": {"slots": 2, "port": 1}, "share": {"devices": []}}))
        mine = onboard.load_config()
        onboard.update_config(lambda c: c["share"].update({"devices": [{"name": "x"}]}))  # another process
        mine["server"]["slots"] = 1
        onboard.save_config(mine)
        data = json.loads((self.home / "config.json").read_text())
        self.assertEqual(data["server"], {"slots": 1, "port": 1})
        self.assertEqual(data["share"]["devices"], [{"name": "x"}])

    def test_unreadable_config_raises_and_is_not_overwritten(self):
        (self.home / "config.json").write_text("{half")
        with self.assertRaises(onboard.ConfigError):
            onboard.load_config()
        with self.assertRaises(onboard.ConfigError):
            onboard.update_config(lambda c: c.update(x=1))
        self.assertEqual((self.home / "config.json").read_text(), "{half")

    def test_missing_config_is_empty(self):
        self.assertEqual(onboard.load_config(), {})


class Restore(unittest.TestCase):
    def test_restore_removes_keys_that_were_absent_and_restores_the_budget_unit(self):
        c = {"hardware": {"vramGB": 16, "ramGB": 32, "backend": "cuda", "cpuThreads": 12},
             "server": {"fitTargetMiB": 1536, "slots": 2, "port": 1, "context": "auto"}, "host": {}}
        snap = setups.snapshot(c)
        budgets.set_preset(c, "work")
        self.assertEqual(c["server"]["threads"], 6)
        setups.restore_snapshot(c, snap)
        self.assertNotIn("threads", c["server"])
        self.assertEqual((c["server"]["slots"], c["server"]["fitTargetMiB"]), (2, 1536))
        self.assertNotIn("budget", c)
        self.assertNotIn("idleUnloadMinutes", c["host"])


class Guard(unittest.TestCase):
    def cfg(self, **custom):
        c = {"hardware": {"vramGB": 15.9, "ramGB": 22.6, "backend": "cuda", "cpuThreads": 12},
             "server": {"fitTargetMiB": 1536, "slots": 2}, "host": {}, "llamaServer": "llama-server"}
        if custom:
            budgets.set_preset(c, "balanced", custom)
        return c

    def found(self, ram):
        return ({"gpuGB": 10.0, "ramGB": ram, "totalGB": 10.0 + ram}, 131072, {"weightsGB": 20.0, "kvGB": 1.0, "perToken": 11264})

    def test_balanced_only_warns_but_an_explicit_ram_limit_refuses(self):
        for custom, expect in (({}, True), ({"ramGB": 5.0}, False)):
            c = self.cfg(**custom)
            with mock.patch.object(budgets, "model_need", return_value=self.found(18.0)), \
                    mock.patch.object(budgets, "memory_gb", return_value=(22.6, 30.0)), mock.patch.object(budgets, "log_refusal"):
                with mock.patch("builtins.print"):
                    self.assertEqual(budgets.guard(c, Path("m.gguf")), expect)

    def test_corrupt_model_header_warns_and_allows(self):
        with mock.patch.object(budgets, "model_need", side_effect=ValueError("bad gguf")), mock.patch("builtins.print") as out:
            self.assertTrue(budgets.guard(self.cfg(), Path("m.gguf")))
        self.assertIn("estimate unavailable", str(out.call_args_list).lower())

    def test_pending_compares_running_to_config(self):
        c = self.cfg()
        got = {"ready": True, "settings": {"fitTargetMiB": 1536, "slots": 2, "threads": None, "kvCacheType": None, "batch": None,
                                           "context": 4096, "server": "llama-server", "modelPath": "m.gguf", "requested": None}}
        with mock.patch("hearthwork.context.choose", return_value=(4096, "x", None)):
            self.assertEqual(budgets.pending(c, got), [])
            c["server"]["slots"] = 1
            self.assertEqual(budgets.pending(c, got), ["slots: 2 -> 1"])
            self.assertEqual(budgets.pending(c, dict(got, transition={"kind": "switch"})), [])


if __name__ == "__main__":
    unittest.main()
