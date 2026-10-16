import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from hearthwork import budgets, server

PC = {"vramGB": 15.9, "ramGB": 22.6, "backend": "cuda", "cpuThreads": 12}


def cfg(**hw):
    return {"hardware": dict(PC, **hw), "server": {"fitTargetMiB": 1536, "slots": 2, "batch": 4096, "kvCacheType": "q8_0",
            "port": 8001, "context": "auto"}, "host": {"idleUnloadMinutes": 0}, "llamaServer": "llama-server"}


class Presets(unittest.TestCase):
    def test_balanced_is_todays_settings(self):
        c = cfg()
        e = budgets.effective(c)
        self.assertEqual((e["fitTargetMiB"], e["slots"], e["threads"], e["idleMinutes"]), (1536, 2, None, 0))

    def test_work_and_back_restores_the_setup_values(self):
        c = cfg()
        e = budgets.set_preset(c, "work")
        self.assertEqual((e["fitTargetMiB"], e["slots"], e["threads"], e["idleMinutes"]), (5120, 1, 6, 15))
        self.assertEqual((c["server"]["fitTargetMiB"], c["server"]["slots"], c["server"]["threads"]), (5120, 1, 6))
        self.assertEqual(c["host"]["idleUnloadMinutes"], 15)
        budgets.set_preset(c, "balanced")
        self.assertEqual((c["server"]["fitTargetMiB"], c["server"]["slots"], c["host"]["idleUnloadMinutes"]), (1536, 2, 0))
        self.assertNotIn("threads", c["server"])
        self.assertNotIn("base", c["budget"])

    def test_dedicated_keeps_little_headroom(self):
        self.assertEqual(budgets.set_preset(cfg(), "dedicated")["fitTargetMiB"], 512)

    def test_custom_vram_becomes_fit_target(self):
        c = cfg()
        e = budgets.set_preset(c, "balanced", {"vramGB": 10.0, "ramGB": 12.0, "slots": 1, "threads": 8, "idle": 20})
        self.assertEqual(e["fitTargetMiB"], round((15.9 - 10) * 1024))
        self.assertEqual((e["ramGB"], e["slots"], e["threads"], e["idleMinutes"]), (12.0, 1, 8, 20))
        self.assertEqual(c["server"]["threads"], 8)

    def test_parse_set_units(self):
        self.assertEqual(budgets.parse_set(["vram=10GB", "ram=12", "idle=20m", "slots=1"])[0],
                         {"vramGB": 10.0, "ramGB": 12.0, "idle": 20, "slots": 1})
        self.assertIsNotNone(budgets.parse_set(["vram=lots"])[1])
        self.assertIsNotNone(budgets.parse_set(["gpu=1"])[1])

    def test_threads_reach_the_command(self):
        c = cfg()
        budgets.set_preset(c, "balanced", {"threads": 6})
        cmd = server.command(c, Path("m.gguf"), 4096, slot_path="x")
        self.assertEqual(cmd[cmd.index("-t") + 1], "6")
        budgets.set_preset(c, "balanced")
        self.assertNotIn("-t", server.command(c, Path("m.gguf"), 4096, slot_path="x"))


class Estimates(unittest.TestCase):
    def test_discrete_gpu_takes_the_vram_budget_rest_is_ram(self):
        e = budgets.estimate(20.5, 1.4, PC, {"vramGB": 8.0})
        self.assertEqual((e["gpuGB"], e["ramGB"], e["totalGB"]), (8.0, 15.4, 23.4))

    def test_unified_memory_is_counted_once(self):
        mac = {"backend": "metal", "ramGB": 32.0, "vramGB": 24.0}
        e = budgets.estimate(20.5, 1.4, mac, {"vramGB": 24.0})
        self.assertEqual((e["gpuGB"], e["ramGB"]), (0.0, 23.4))
        self.assertEqual(budgets.total_vram(mac), 0.0)

    def test_impossible_budget_explained_with_units_and_no_context_shrink(self):
        c = cfg(vramGB=8.0)
        e = budgets.set_preset(c, "balanced", {"vramGB": 8.0, "ramGB": 12.0})
        need = budgets.estimate(20.5, 1.4, c["hardware"], e)
        details = {"weightsGB": 20.5, "kvGB": 1.4, "perToken": 11264}
        problems = budgets.check(c, need, 131072, details, eff=e)
        self.assertEqual(len(problems), 1)
        self.assertIn("RAM budget: 12.0 GB", problems[0])
        self.assertIn("3.6 GB too much", problems[0])
        text = "\n".join(budgets.explain(c, Path("Qwen3.5-35B-A3B-Q4_K_M.gguf"), problems, need, 131072, details, e))
        self.assertIn("smaller quantization", text)
        self.assertIn("another preset", text)
        self.assertNotIn("--context", text)  # 1.4 GB of KV cannot cover 3.6 GB: no pointless context advice

    def test_viable_budget_has_no_problems(self):
        c = cfg()
        e = budgets.effective(c)
        need = budgets.estimate(20.5, 1.4, c["hardware"], e)
        self.assertEqual(budgets.check(c, need, 131072, {}, eff=e), [])

    def test_low_free_ram_only_warns_and_reclaim_counts(self):
        need = {"ramGB": 12.4, "totalGB": 23.3, "gpuGB": 10.9}
        self.assertIn("Close other apps", budgets.warnings(need, 3.0)[0])
        self.assertEqual(budgets.warnings(need, 3.0, 10.0), [])

    def test_more_than_the_machine_has_is_refused(self):
        c = cfg()
        e = budgets.effective(c)
        need = budgets.estimate(60.0, 1.4, c["hardware"], e)
        self.assertIn("this computer's memory", budgets.check(c, need, 1, {}, eff=e)[-1])

    def test_guard_refuses_but_force_goes_on_and_context_is_untouched(self):
        c = cfg()
        c["budget"] = {"preset": "balanced", "custom": {"ramGB": 4.0}}
        found = (budgets.estimate(20.5, 1.4, c["hardware"], budgets.effective(c)), 131072,
                 {"weightsGB": 20.5, "kvGB": 1.4, "perToken": 11264})
        with mock.patch.object(budgets, "model_need", return_value=found), mock.patch.object(budgets, "memory_gb", return_value=(22.6, 18.0)), mock.patch.object(budgets, "log_refusal"):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertFalse(budgets.guard(c, Path("m.gguf"), 131072))
                self.assertTrue(budgets.guard(c, Path("m.gguf"), 131072, over_budget=True))
        self.assertIn("Not starting m.gguf", out.getvalue())
        self.assertIn("128K", out.getvalue())

    def test_table_has_three_columns_and_enforcement_note(self):
        text = "\n".join(budgets.table(cfg(), None, {"ramGB": None, "gpu": None}))
        for word in ("configured budget", "estimated need", "measured now", "best effort", "unavailable"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
