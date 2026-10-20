import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from hearthwork import tune


def touch(folder, name, size):
    path = Path(folder) / name
    path.write_bytes(b"x" * size)
    return path


class Helpers(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(tune.parse_time("30m"), 1800)
        self.assertEqual(tune.parse_time("90s"), 90)
        self.assertEqual(tune.parse_time("1h"), 3600)
        self.assertEqual(tune.parse_time("30"), 1800)  # a bare number is minutes
        with self.assertRaises(ValueError):
            tune.parse_time("soon")
        with self.assertRaises(ValueError):
            tune.parse_time("0m")

    def test_family_strips_quantization(self):
        self.assertEqual(tune.family("Qwen3-14B-Q4_K_M"), "qwen3-14b")
        self.assertEqual(tune.family("Qwen3-14B-UD-Q3_K_XL"), "qwen3-14b")
        self.assertEqual(tune.family("Muse-30B-Q8_0"), "muse-30b")
        self.assertNotEqual(tune.family("Qwen3-14B-Q4_K_M"), tune.family("Qwen3-32B-Q4_K_M"))

    def test_estimate_cost(self):
        self.assertEqual(tune.estimate_cost(60, 100, 3), 360)
        self.assertEqual(tune.estimate_cost(None, None, 3), tune.LOAD_ESTIMATE * 4)


class Ranking(unittest.TestCase):
    def test_rank_score_then_seconds_then_latency(self):
        a = {"label": "a", "score_fraction": 1.0, "seconds": 200, "ttfb_ms": 50}
        b = {"label": "b", "score_fraction": 1.0, "seconds": 100, "ttfb_ms": 90}
        c = {"label": "c", "score_fraction": 0.67, "seconds": 10, "ttfb_ms": 10}
        d = {"label": "d", "score_fraction": 1.0, "seconds": 100, "ttfb_ms": 20}
        self.assertEqual([r["label"] for r in tune.rank([a, b, c, d])], ["d", "b", "a", "c"])

    def test_close_pair_on_score_seconds(self):
        best = {"score_fraction": 1.0, "seconds": 100}
        near = {"score_fraction": 1.0, "seconds": 108}     # 8% slower: a tie
        far = {"score_fraction": 1.0, "seconds": 130}      # 30% slower: not a tie
        self.assertTrue(tune.close_pair(best, near))
        self.assertFalse(tune.close_pair(best, far))
        # a lower score at a matching rate is not a tie
        self.assertFalse(tune.close_pair({"score_fraction": 0.5, "seconds": 30}, best))

    def test_score_seconds_without_score(self):
        self.assertEqual(tune.score_seconds({"score_fraction": 0, "seconds": 10}), float("inf"))


class Candidates(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.rng = {"min": 65536, "recommended": 65536, "max": 262144, "modelMax": 262144, "kvGB": 3.0, "reason": "x"}

    def build(self, paths, current=None, fits=None, choose=None, rng=None, min_context=None, limit=tune.MAX_CANDIDATES,
              ranked=None):
        choose = choose or (lambda config, path: (65536, "auto", None))
        rng = rng or (lambda config, path: self.rng)
        fits = fits or (lambda config, path, context: ({"totalGB": 10.0, "gpuGB": 8.0, "ramGB": 2.0}, None))
        ranked = ranked if ranked is not None else [(p, i + 1, f"rank {i + 1} for this computer") for i, p in enumerate(paths)]
        with mock.patch.object(tune.ctx, "choose", side_effect=choose), \
                mock.patch.object(tune.ctx, "range_for", side_effect=rng), \
                mock.patch.object(tune, "fits", side_effect=fits):
            return tune.build_candidates({}, min_context, current=current, models=paths, limit=limit, ranked=ranked)

    def test_installed_models_capped(self):
        paths = [touch(self.dir, f"Model-{i}-Q4_K_M.gguf", 100 + i) for i in range(8)]
        candidates, _ = self.build(paths)
        self.assertEqual(len(candidates), tune.MAX_CANDIDATES)
        self.assertTrue(all(c["kind"] == "model" for c in candidates))

    def test_ranked_order_and_current_first(self):
        a = touch(self.dir, "Alpha-8B-Q4_K_M.gguf", 100)
        b = touch(self.dir, "Bravo-8B-Q4_K_M.gguf", 200)
        c = touch(self.dir, "Charlie-8B-Q4_K_M.gguf", 300)
        ranked = [(c, 1, "rank 1"), (b, 2, "rank 2"), (a, 3, "rank 3")]
        candidates, _ = self.build([a, b, c], current=b, ranked=ranked)
        self.assertEqual([x["model"].name for x in candidates],
                         ["Bravo-8B-Q4_K_M.gguf", "Charlie-8B-Q4_K_M.gguf", "Alpha-8B-Q4_K_M.gguf"])
        self.assertTrue(candidates[0]["current"])
        self.assertEqual(candidates[0]["reason"], "the model that is running now")
        self.assertEqual(candidates[1]["reason"], "rank 1")

    def test_extras_are_always_kept_within_the_cap(self):
        paths = [touch(self.dir, f"Model-{i}-Q4_K_M.gguf", 100 + i) for i in range(8)]
        cur = touch(self.dir, "Current-14B-Q4_K_M.gguf", 500)
        smaller = touch(self.dir, "Current-14B-Q3_K_M.gguf", 300)
        candidates, _ = self.build(paths + [cur, smaller], current=cur, choose=lambda c, p: (131072, "auto", None))
        kinds = [c["kind"] for c in candidates]
        self.assertEqual(len(candidates), tune.MAX_CANDIDATES)
        self.assertEqual(candidates[0]["model"].name, cur.name)  # current first, over the cap
        self.assertIn("context", kinds)  # the two extras survive the cap
        self.assertIn("quant", kinds)

    def test_over_budget_model_is_skipped_with_a_note(self):
        good = touch(self.dir, "Good-8B-Q4_K_M.gguf", 100)
        big = touch(self.dir, "Huge-70B-Q4_K_M.gguf", 900)

        def fits(config, path, context):
            return (None, "RAM budget: 8.0 GB, the model needs about 40.0 GB in RAM") \
                if path.name.startswith("Huge") else ({"totalGB": 9.0}, None)

        candidates, notes = self.build([good, big], fits=fits)
        self.assertEqual([c["model"].name for c in candidates], ["Good-8B-Q4_K_M.gguf"])
        self.assertTrue(any("Huge" in n and "RAM budget" in n for n in notes))

    def test_extras_context_step_and_smaller_quant(self):
        cur = touch(self.dir, "Qwen3-14B-Q4_K_M.gguf", 500)
        smaller = touch(self.dir, "Qwen3-14B-Q3_K_M.gguf", 300)
        other = touch(self.dir, "Llama-8B-Q4_K_M.gguf", 100)
        candidates, _ = self.build([cur, smaller, other], current=cur,
                                   choose=lambda c, p: (131072 if Path(p).name == cur.name else 65536, "auto", None))
        labels = [c["label"] for c in candidates]
        step = next(c for c in candidates if c["kind"] == "context")
        self.assertEqual(step["context"], 131072 - tune.ctx.STEP)
        self.assertTrue(any(c["kind"] == "quant" and c["model"].name == smaller.name for c in candidates))
        self.assertEqual(len(labels), len(set(labels)))

    def test_no_extra_when_already_at_the_minimum_context(self):
        cur = touch(self.dir, "Solo-9B-Q4_K_M.gguf", 100)
        candidates, _ = self.build([cur], current=cur, choose=lambda c, p: (65536, "auto", None))
        self.assertEqual([c["kind"] for c in candidates], ["model"])

    def test_min_context_requirement(self):
        wide = touch(self.dir, "Wide-8B-Q4_K_M.gguf", 100)
        narrow = touch(self.dir, "Narrow-3B-Q4_K_M.gguf", 50)

        def rng(config, path):
            return dict(self.rng, max=32768) if path.name.startswith("Narrow") else self.rng

        candidates, notes = self.build([wide, narrow], choose=lambda c, p: (32768, "auto", None), rng=rng, min_context=65536)
        self.assertEqual([c["context"] for c in candidates], [65536])
        self.assertTrue(any("above this model's range" in n for n in notes))


class RunLoop(unittest.TestCase):
    def config(self):
        return {"server": {"port": 8001}, "modelsDir": "/x"}

    def candidate(self, label, model):
        return {"label": label, "model": Path(model), "context": 65536, "kind": "model", "need": {"totalGB": 8.0}}

    def status(self):
        return {"ready": True, "sessions": [], "settings": {"modelPath": "/models/Original-30B.gguf"},
                "context": 65536, "requestedContext": 65536}

    def result(self, label, seconds, load=60.0, score=3, ttfb=100):
        return {"label": label, "model": label, "context": 65536, "kind": "model", "score": score, "tasks": 3,
                "score_fraction": score / 3, "seconds": seconds, "load_seconds": load, "ttfb_ms": ttfb,
                "gen_tps": 20.0, "peak_ramGB": 5.0, "peak_gpuGB": None}

    def test_stops_before_a_candidate_that_would_overrun(self):
        clock = {"t": 0.0}
        candidates = [self.candidate("a", "/m/a.gguf"), self.candidate("b", "/m/b.gguf"), self.candidate("c", "/m/c.gguf")]
        calls = []

        def fake(config, cand, agent, tasks, root, say):
            calls.append(cand["label"])
            clock["t"] += 400  # each candidate takes 400 s
            return self.result(cand["label"], 400)

        with mock.patch.object(tune.time, "monotonic", side_effect=lambda: clock["t"] + 0.0):
            # re-run with a controllable clock: patch before run() so its loop reads it
            said = []
            profiles, logged = [], []
            with mock.patch.object(tune.host, "status", return_value=self.status()), \
                    mock.patch.object(tune, "build_candidates", return_value=(candidates, [])), \
                    mock.patch.object(tune, "run_candidate", side_effect=fake), \
                    mock.patch.object(tune, "restore") as restore, \
                    mock.patch.object(tune, "winner_profile", return_value={"name": "tune-x"}), \
                    mock.patch.object(tune, "append_log"):
                code = tune.run(self.config(), "claude", 3, 500, None, say=said.append)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["a"])  # only the first fit in the time; the rest were dropped
        self.assertTrue(any("Stopping" in s for s in said))
        restore.assert_called_once()

    def test_repeats_a_close_pair_when_time_allows(self):
        candidates = [self.candidate("a", "/m/a.gguf"), self.candidate("b", "/m/b.gguf")]
        seq = {"n": 0}

        def fake(config, cand, agent, tasks, root, say):
            seq["n"] += 1
            label = cand["label"] if seq["n"] <= 2 else cand["label"] + " (repeat)"
            r = self.result(label, 100)
            return r

        with mock.patch.object(tune.host, "status", return_value=self.status()), \
                mock.patch.object(tune, "build_candidates", return_value=(candidates, [])), \
                mock.patch.object(tune, "run_candidate", side_effect=fake), \
                mock.patch.object(tune, "restore"), \
                mock.patch.object(tune, "winner_profile", return_value={"name": "tune-x"}), \
                mock.patch.object(tune, "append_log"):
            code = tune.run(self.config(), "claude", 3, 1800, None, say=lambda s: None)
        self.assertEqual(code, 0)
        self.assertEqual(seq["n"], 3)  # two candidates + one repeat of the (tied) best

    def test_a_raising_candidate_restores_the_original(self):
        candidates = [self.candidate("a", "/m/a.gguf")]

        def boom(config, cand, agent, tasks, root, say):
            raise RuntimeError("mid-run failure")

        with mock.patch.object(tune.host, "status", return_value=self.status()), \
                mock.patch.object(tune, "build_candidates", return_value=(candidates, [])), \
                mock.patch.object(tune, "run_candidate", side_effect=boom), \
                mock.patch.object(tune, "switch_to") as switch_to:
            code = tune.run(self.config(), "claude", 3, 1800, None, say=lambda s: None)
        self.assertEqual(code, 1)  # nothing to recommend
        switch_to.assert_called_once()  # restore() switched back to the original model
        self.assertEqual(Path(switch_to.call_args[0][1]).name, "Original-30B.gguf")

    def test_open_sessions_abort(self):
        status = self.status()
        status["sessions"] = [{"kind": "claude", "pid": 1, "since": 0.0}]
        said = []
        with mock.patch.object(tune.host, "status", return_value=status):
            code = tune.run(self.config(), "claude", 3, 1800, None, say=said.append)
        self.assertEqual(code, 1)
        self.assertTrue(any("Close your agent sessions first" in s for s in said))

    def test_no_model_running(self):
        with mock.patch.object(tune.host, "status", return_value=None):
            self.assertEqual(tune.run(self.config(), "claude", 3, 1800, None, say=lambda s: None), 1)


class Saving(unittest.TestCase):
    def test_winner_profile_uses_the_tune_run_as_evidence(self):
        from hearthwork import setups
        captured = {}

        def fake_capture(config, name, evidence=True, **kw):
            captured["name"], captured["evidence"] = name, evidence
            captured["lastModel"] = config.get("lastModel")
            captured["context"] = (config.get("contexts") or {}).get(Path(config["lastModel"]).name)
            return {"name": name, "schema": 1, "model": {"file": "w.gguf"}, "settings": {}, "runtime": {}, "hardware": {}}

        winner = {"model": Path("/m/Winner-Q4_K_M.gguf"), "context": 131072, "label": "Winner"}
        run = {"results": [], "winner": {"label": "Winner"}}
        with mock.patch.object(setups, "capture", side_effect=fake_capture), \
                mock.patch.object(setups, "store") as store:
            profile = tune.winner_profile({}, "tune-2026-01-02", winner, run)
        self.assertEqual(captured["name"], "tune-2026-01-02")
        self.assertFalse(captured["evidence"])  # no doctor run: this run is the evidence
        self.assertEqual(Path(captured["lastModel"]).name, "Winner-Q4_K_M.gguf")
        self.assertEqual(captured["context"], 131072)
        self.assertEqual(profile["status"], setups.VERIFIED)
        self.assertEqual(profile["evidence"]["tune"], run)
        store.assert_called_once()

    def test_log_rotates_past_five_mb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tune.jsonl"
            path.write_text("x" * 40, encoding="utf-8")
            tune.append_log({"run": 1}, path=path, rotate_at=32)
            self.assertTrue((Path(tmp) / "tune.jsonl.1").exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"run": 1})
            tune.append_log({"run": 2}, path=path, rotate_at=32)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(l) for l in lines], [{"run": 1}, {"run": 2}])


class DryRun(unittest.TestCase):
    def run_main(self, argv, candidates):
        out = io.StringIO()
        with mock.patch.object(tune, "load_config", return_value={"modelsDir": "/x"}), \
                mock.patch.object(tune, "build_candidates", return_value=(candidates, ["Note-8B: skipped (too big)"])), \
                mock.patch.object(tune.host, "status", side_effect=AssertionError("--dry-run must not touch the host")):
            with redirect_stdout(out):
                code = tune.main(argv)
        return code, out.getvalue()

    def candidate(self):
        return {"label": "Model-8B-Q4_K_M.gguf @ 64K", "model": Path("/m/Model-8B-Q4_K_M.gguf"), "context": 65536,
                "kind": "model", "need": {"totalGB": 9.0, "gpuGB": 8.0, "ramGB": 1.0}}

    def test_dry_run_lists_candidates_without_switching(self):
        code, text = self.run_main(["--dry-run"], [self.candidate()])
        self.assertEqual(code, 0)
        self.assertIn("Model-8B-Q4_K_M.gguf @ 64K", text)
        self.assertIn("9.0 GB total", text)
        self.assertIn("skipped (too big)", text)

    def test_dry_run_json(self):
        code, text = self.run_main(["--dry-run", "--json"], [self.candidate()])
        self.assertEqual(code, 0)
        data = json.loads(text)
        self.assertEqual(data["candidates"][0]["context"], 65536)
        self.assertEqual(data["tasks"], tune.DEFAULT_TASKS)

    def test_bad_time_argument(self):
        with mock.patch.object(tune, "load_config", return_value={}):
            with self.assertRaises(SystemExit):
                tune.main(["--time", "soon"])


if __name__ == "__main__":
    unittest.main()