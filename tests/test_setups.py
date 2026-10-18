import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import setups


def profile(**kw):
    p = {"schema": 1, "name": "t", "created": "2026-01-01T00:00:00",
         "model": {"file": "m.gguf", "sizeBytes": 10, "fingerprint": None, "quantization": "Q4_K_M",
                   "architecture": "qwen", "chatTemplate": "abc"},
         "runtime": {"name": "cuda13", "llamaCppBuild": 100}, "agents": {"claude": "1.0"},
         "settings": {"context": "auto", "kvCacheType": "q8_0", "slots": 2, "batch": 4096, "fitTargetMiB": 1536},
         "hardware": {"os": "windows", "backend": "cuda", "vramGB": 16, "ramGB": 32, "gpus": ["X"]},
         "evidence": {"doctor": {"date": "d", "suite": "s", "conditions": {"quick": True},
                                 "agents": {"claude": {"verdict": "works", "version": "1.0"}}}, "bench": None},
         "status": "verified"}
    p.update(kw)
    return p


class SetupsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        patcher = mock.patch.object(setups, "DIR", self.tmp / "profiles")
        patcher.start()
        self.addCleanup(patcher.stop)
        p2 = mock.patch.object(setups, "still_pending", lambda config: [])  # never ask a real host
        p2.start()
        self.addCleanup(p2.stop)
        self.models = self.tmp / "models"
        (self.models / "sub").mkdir(parents=True)
        self.config = {"modelsDir": str(self.models), "server": {"port": 1}}

    def test_validate(self):
        setups.validate(profile())
        with self.assertRaises(ValueError):
            setups.validate({"schema": 2})
        bad = profile()
        del bad["settings"]
        with self.assertRaises(ValueError):
            setups.validate(bad)
        with self.assertRaises(ValueError):
            setups.validate(profile(name="../x"))

    def test_verified_requires_all_works(self):
        ok = {"agents": {"a": {"verdict": "works"}, "b": {"verdict": "works"}}}
        self.assertTrue(setups.is_verified(ok))
        ok["agents"]["b"]["verdict"] = "fails"
        self.assertFalse(setups.is_verified(ok))
        self.assertFalse(setups.is_verified(None))
        self.assertFalse(setups.is_verified({"agents": {}}))

    def test_export_sanitizes(self):
        p = profile(extra=f"{self.models}\\sub\\m.gguf", remote={"key": "secret", "apiKey": "x"})
        p["evidence"]["doctor"]["note"] = f"{Path.home()}/proj Bearer abcdefghijkl"
        p["_restore"] = {"x": 1}
        out = json.dumps(setups.export_profile(self.config, p))
        for forbidden in (str(self.models), "secret", "apiKey", "abcdefghijkl", str(Path.home()).replace("\\", "\\\\"), "_restore"):
            self.assertNotIn(forbidden, out)
        self.assertIn("<MODELS_DIR>", out)

    def test_import_resolves_model(self):
        f = self.models / "sub" / "m.gguf"
        f.write_bytes(b"x" * 10)
        data = profile()
        data["model"]["fingerprint"] = setups.fingerprint(f)
        hw = {"os": "windows", "backend": "cuda", "vramGB": 16, "ramGB": 32}
        with mock.patch.object(setups, "runtime_of", return_value={"name": "cuda13", "llamaCppBuild": 100}):
            p, problems = setups.import_profile(self.config, data, hw=hw, agents={"claude": "1.0"})
        self.assertFalse([x for x in problems if "model" in x])
        self.assertEqual(p["status"], "unverified")
        self.assertEqual(p["recordedStatus"], "verified")

    def test_import_reports_differences(self):
        hw = {"os": "linux", "backend": "cpu", "vramGB": 8, "ramGB": 16}
        with mock.patch.object(setups, "runtime_of", return_value={"name": "cpu", "llamaCppBuild": 5}):
            _, problems = setups.import_profile(self.config, profile(), hw=hw, agents={"claude": "2.0"})
            text = "\n".join(problems)
            for needle in ("model not found", "llama.cpp build", "agent claude version differs", "VRAM", "RAM", "backend", "OS differs"):
                self.assertIn(needle, text)
            _, problems = setups.import_profile(self.config, profile(), hw=hw, agents={})
        self.assertIn("agent claude is not installed", "\n".join(problems))

    def test_find_model_wrong_size(self):
        (self.models / "m.gguf").write_bytes(b"x" * 3)
        self.assertTrue(setups.find_model(self.config, {"file": "m.gguf", "sizeBytes": 10})[1])

    def test_change_detection(self):
        p = profile()
        now = {"model": dict(p["model"]), "runtime": dict(p["runtime"]), "agents": {"claude": "1.0"}}
        self.assertEqual(setups.differences(p, now), [])
        now["runtime"]["llamaCppBuild"] = 101
        now["agents"]["claude"] = "1.1"
        now["model"]["chatTemplate"] = "zzz"
        text = "\n".join(setups.differences(p, now))
        for needle in ("llama.cpp build", "agent claude version", "chat template"):
            self.assertIn(needle, text)
        status, changes = setups.check(self.config, p, now=now)
        self.assertEqual(status, "needs recheck")
        self.assertEqual(len(changes), 3)

    def test_check_run(self):
        now = {"model": profile()["model"], "runtime": {"name": "cuda13", "llamaCppBuild": 101}, "agents": {"claude": "1.1"}}
        good = {"time": "t", "agents": {"claude": {"verdict": "works", "version": "1.1", "stages": []}}}
        p = profile()
        status, _ = setups.check(self.config, p, run=True, run_doctor=lambda c, a: good, now=now)
        self.assertEqual(status, "verified")
        self.assertEqual(p["runtime"]["llamaCppBuild"], 101)
        bad = {"agents": {"claude": {"verdict": "fails", "stages": []}}}
        status, _ = setups.check(self.config, profile(), run=True, run_doctor=lambda c, a: bad, now=now)
        self.assertEqual(status, "needs recheck")

    def test_doctor_evidence_quick_flag(self):
        report = {"time": "t", "agents": {"a": {"verdict": "works", "stages": [{"name": "coding task"}]}}}
        self.assertFalse(setups.doctor_evidence(report)["conditions"]["quick"])

    def test_apply_failure_restores_then_previous(self):
        f = self.models / "m.gguf"
        f.write_bytes(b"x" * 10)
        old = self.models / "old.gguf"
        old.write_bytes(b"y")
        config = {"modelsDir": str(self.models), "lastModel": str(old),
                  "server": {"context": 8192, "slots": 1, "batch": 512, "kvCacheType": "f16", "fitTargetMiB": 100, "port": 1}}
        data = profile(agents={})
        data["model"]["fingerprint"] = setups.fingerprint(f)
        ensured = []
        stub = {"name": "previous", "settings": {}}
        with mock.patch.object(setups, "capture", return_value=dict(stub)), \
                mock.patch("hearthwork.context.choose", return_value=(4096, "x", None)):
            ok = setups.apply(config, data, lambda c: None, lambda c, m, x: ensured.append(m.name) or True,
                              validator=lambda c: False, versions={}, running=lambda c: None)
            self.assertFalse(ok)
            self.assertEqual(config["server"]["slots"], 1)
            self.assertEqual(config["lastModel"], str(old))
            self.assertEqual(ensured, ["m.gguf", "old.gguf"])
            self.assertNotIn("activeProfile", config)
            ok = setups.apply(config, data, lambda c: None, lambda c, m, x: True, versions={}, running=lambda c: None)
            self.assertTrue(ok)
            self.assertEqual((config["server"]["slots"], config["activeProfile"]), (2, "t"))
            # the second apply saved the settings from before it: back to the old model's setup
            config["server"]["slots"] = 1
            config["lastModel"] = str(old)
            self.assertTrue(setups.apply(config, data, lambda c: None, lambda c, m, x: True, versions={}, running=lambda c: None))
            self.assertTrue(setups.apply_previous(config, lambda c: None, lambda c, m, x: True))
            self.assertEqual((config["server"]["slots"], config["lastModel"]), (1, str(old)))

    def test_apply_already_active(self):
        f = self.models / "m.gguf"
        f.write_bytes(b"x" * 10)
        config = {"modelsDir": str(self.models), "lastModel": str(f),
                  "server": {"context": "auto", "slots": 2, "batch": 4096, "kvCacheType": "q8_0", "fitTargetMiB": 1536, "port": 1}}
        data = profile(agents={}, runtime={"name": None, "llamaCppBuild": None})
        data["model"]["fingerprint"] = setups.fingerprint(f)
        msgs, ensured = [], []
        live = {"model": "m", "context": 4096, "slots": 2}
        with mock.patch.object(setups, "capture", return_value={"name": "previous", "settings": {}}),                 mock.patch("hearthwork.context.choose", return_value=(4096, "x", None)):
            ok = setups.apply(config, data, lambda c: None, lambda *a: ensured.append(1) or True, say=msgs.append,
                              versions={}, running=lambda c: live)
            self.assertTrue(ok)
            self.assertEqual(ensured, [])
            self.assertEqual(config["activeProfile"], "t")
            self.assertIn("already active", msgs[-1])
            data["settings"]["batch"] = 512  # a real difference restarts, and says which
            msgs.clear()
            setups.apply(config, data, lambda c: None, lambda *a: ensured.append(1) or True, say=msgs.append,
                         versions={}, running=lambda c: live)
        self.assertEqual(ensured, [1])
        self.assertIn("changed: batch", chr(10).join(msgs))

    def test_unknown_template_is_not_a_change(self):
        p = profile()
        now = {"model": dict(p["model"], chatTemplate=None), "runtime": dict(p["runtime"]), "agents": {"claude": "1.0"}}
        self.assertEqual(setups.differences(p, now), [])
        self.assertEqual(setups.check(self.config, p, now=now)[0], "verified")
        self.assertIn("not compared", setups.template_note(p, now))
        p["model"]["chatTemplate"] = None
        now["model"]["chatTemplate"] = "zzz"
        self.assertEqual(setups.differences(p, now), [])

    def test_template_retries(self):
        replies = iter([None, None, {"chat_template": "T"}])
        with mock.patch("hearthwork.doctor.get_json", side_effect=lambda *a, **k: next(replies)),                 mock.patch.object(setups.time, "sleep"):
            self.assertIsNotNone(setups.template_fingerprint({"server": {"port": 1}}))
        with mock.patch("hearthwork.doctor.get_json", return_value=None), mock.patch.object(setups.time, "sleep"):
            self.assertIsNone(setups.template_fingerprint({}))

    def test_apply_missing_model_refuses(self):
        msgs = []
        ok = setups.apply({"modelsDir": str(self.models), "server": {}}, profile(), lambda c: None, lambda *a: True,
                          say=msgs.append, versions={}, running=lambda c: None)
        self.assertFalse(ok)
        self.assertIn("model not found", msgs[0])

    def test_status_line(self):
        setups.store(profile())
        cfg = {"activeProfile": "t"}
        with mock.patch.object(setups, "quick_changes", return_value=[]):
            self.assertEqual(setups.status_line(cfg), "profile: t (verified)")
        with mock.patch.object(setups, "quick_changes", return_value=["llama.cpp build"]):
            self.assertEqual(setups.status_line(cfg), "profile: t (needs recheck)")
        self.assertIsNone(setups.status_line({}))


class DoctorProgress(unittest.TestCase):
    """`profile save` runs Doctor one agent at a time and shows each result as it lands: fake reports, exact lines."""

    def fake(self, key, title, verdict, seconds):
        return {"time": "2026-01-01T00:00:00", "mode": "local",
                "server": {"name": "server", "status": "pass", "seconds": 0.1, "detail": "m on port 8001"},
                "environment": {"hearthwork": "0.10", "model": "m"}, "protocol": {},
                "agents": {key: {"title": title, "api": "openai-chat", "version": "1.0", "stages": [],
                                 "verdict": verdict, "failedStage": None, "toolsApplicable": True}},
                "seconds": seconds, "suggestions": [], "ok": verdict == "works"}

    def test_one_line_per_agent_as_it_finishes(self):
        lines = []
        reports = {"claude": self.fake("claude", "Claude Code", "works", 12.3),
                   "aider": self.fake("aider", "Aider", "works", 54.0)}
        with mock.patch("hearthwork.doctor.save"):
            merged = setups.save_doctor({}, ["claude", "aider"], say=lines.append, run_one=lambda keys: reports[keys[0]])
        self.assertEqual(lines, ["Doctor: Claude Code works (12.3 s)", "Doctor: Aider works (54.0 s)",
                                 "verified: 2 of 2 agents work"])
        self.assertEqual(sorted(merged["agents"]), ["aider", "claude"])
        self.assertEqual(merged["seconds"], 66.3)
        self.assertTrue(merged["ok"])
        self.assertTrue(setups.is_verified(setups.doctor_evidence(merged)))

    def test_quick_agents_checks_only_one_fast_agent(self):
        seen, lines = [], []

        def run_one(keys):
            seen.extend(keys)
            return self.fake(keys[0], "Claude Code", "works", 5.0)

        with mock.patch("hearthwork.doctor.save"):
            merged = setups.save_doctor({}, ["aider", "claude", "codex"], say=lines.append, quick_agents=True,
                                        run_one=run_one)
        self.assertEqual(seen, ["claude"])  # claude is the fast one, even when not first
        self.assertEqual(lines, ["Doctor: Claude Code works (5.0 s)", "verified: 1 of 1 agents work"])
        self.assertTrue(setups.is_verified(setups.doctor_evidence(merged)))

    def test_quick_agents_without_claude_takes_the_first(self):
        seen = []
        with mock.patch("hearthwork.doctor.save"):
            setups.save_doctor({}, ["aider", "codex"], say=lambda s: None, quick_agents=True,
                               run_one=lambda keys: (seen.extend(keys), self.fake(keys[0], "Aider", "works", 1.0))[1])
        self.assertEqual(seen, ["aider"])

    def test_failing_agent_shows_in_the_summary(self):
        lines = []
        reports = {"claude": self.fake("claude", "Claude Code", "works", 2.0),
                   "aider": self.fake("aider", "Aider", "protocol problem at streaming", 3.0)}
        with mock.patch("hearthwork.doctor.save"):
            merged = setups.save_doctor({}, ["claude", "aider"], say=lines.append, run_one=lambda keys: reports[keys[0]])
        self.assertEqual(lines, ["Doctor: Claude Code works (2.0 s)",
                                 "Doctor: Aider protocol problem at streaming (3.0 s)",
                                 "verified: 1 of 2 agents work"])
        self.assertFalse(merged["ok"])
        self.assertFalse(setups.is_verified(setups.doctor_evidence(merged)))


if __name__ == "__main__":
    unittest.main()
