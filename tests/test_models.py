import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import context, models

SETTINGS = {"fitTargetMiB": 1536, "batch": 4096}


def pc(vram, ram, backend="cuda"):
    return {"vramGB": vram, "ramGB": ram, "backend": backend, "arch": "x64"}


def cand(name, moe, params, files, kv, active=None, tier=None):
    m = {"repo": name, "name": name, "moe": moe, "params": params, "note": "a model",
         "files": {q: {"gb": gb, "file": "f", "bytes": 1} for q, gb in files.items()},
         "info": {"arch": "x", "modelMax": 262144, "layers": 40, "kvPerToken": kv} if kv else None}
    if active:
        m["active"] = active
    if tier:
        m["tier"] = tier
    return m


DEVSTRAL = cand("devstral", False, 24, {"Q4_K_M": 14.3, "Q3_K_M": 10.7}, 87040, tier=4)
MOE = cand("moe30", True, 30, {"Q4_K_M": 17.3, "Q3_K_M": 13.7}, 26112, active=3, tier=3)
SMALL = cand("small9", False, 9, {"Q4_K_M": 5.5}, 40000, tier=2)
BIG = cand("big80", True, 80, {"Q4_K_M": 48.0, "Q3_K_M": 38.0, "Q2_K": 27.0}, 20000, active=3, tier=5)


def rank(cands, hw, free=20.0):
    return models.rank(cands, hw, free, SETTINGS)


def by(picks, name):
    return next(p for p in picks if p["name"] == name)


class FitAndRank(unittest.TestCase):
    def test_devstral_q3_does_not_fit_16gb_gpu_at_64k(self):
        p = by(rank([DEVSTRAL], pc(16, 22.6)), "devstral")
        self.assertEqual(p["fit"], "slow")
        self.assertAlmostEqual(p["kvGB"], 5.3, 1)  # 87 KB/token x 64K

    def test_devstral_fits_a_24gb_gpu(self):
        self.assertEqual(by(rank([DEVSTRAL], pc(24, 32)), "devstral")["fit"], "fast")

    def test_fit_tier_comes_first(self):
        picks = rank([DEVSTRAL, MOE, SMALL, BIG], pc(16, 22.6))
        order = ("fast", "good", "slow", "too big")
        tiers = [order.index(p["fit"]) for p in picks]
        self.assertEqual(tiers, sorted(tiers))
        self.assertEqual(picks[0]["name"], "small9")  # only one that fits the GPU, despite lowest capability

    def test_different_pcs_give_different_orders(self):
        small = [p["name"] for p in rank([DEVSTRAL, MOE, SMALL, BIG], pc(8, 16))]
        large = [p["name"] for p in rank([DEVSTRAL, MOE, SMALL, BIG], pc(32, 64))]
        self.assertNotEqual(small, large)
        self.assertEqual(large[0], "devstral")  # fits the GPU; the 80B MoE needs RAM
        self.assertEqual(small[0], "moe30")

    def test_capability_orders_within_a_tier(self):
        self.assertEqual([p["name"] for p in rank([SMALL, DEVSTRAL], pc(48, 64))], ["devstral", "small9"])

    def test_parameter_count_when_no_tier(self):
        a, b = dict(SMALL, name="a", repo="a", params=9), dict(SMALL, name="b", repo="b", params=30)
        a.pop("tier"), b.pop("tier")
        self.assertEqual(models.rank([a, b], pc(48, 64), 20, SETTINGS)[0]["name"], "b")

    def test_q2_never_recommended(self):
        p = by(rank([cand("q2", True, 80, {"Q2_K": 27.0}, 20000, active=3)], pc(16, 22.6)), "q2")
        self.assertTrue(p["lowQuality"])
        self.assertFalse(p["recommended"])
        self.assertIsNone(models.best([p]))
        self.assertIn("lower quality", models.describe(p)[0])

    def test_q2_only_when_nothing_better_fits(self):
        self.assertFalse(by(rank([BIG], pc(24, 32)), "big80")["lowQuality"])
        self.assertTrue(by(rank([BIG], pc(16, 22.6)), "big80")["lowQuality"])

    def test_q2_sorts_after_q3_in_same_tier(self):
        q2 = cand("q2", True, 80, {"Q2_K": 20.0}, 20000, active=3, tier=5)
        q3 = cand("q3", True, 20, {"Q3_K_M": 15.0}, 20000, active=3, tier=1)
        self.assertEqual([p["name"] for p in rank([q2, q3], pc(16, 22.6))], ["q3", "q2"])

    def test_q3_note(self):
        p = by(rank([DEVSTRAL], pc(18, 16)), "devstral")  # only Q3 fits this GPU
        self.assertEqual(p["quant"], "Q3_K_M")
        self.assertTrue(p["q3"])
        self.assertTrue(any("Q3" in line for line in models.describe(p)[1]))

    def test_too_big(self):
        p = by(rank([BIG], pc(8, 16)), "big80")
        self.assertEqual(p["fit"], "too big")
        self.assertIsNone(models.best([p]))

    def test_free_ram_warning(self):
        low = by(rank([MOE], pc(16, 22.6), free=3.0), "moe30")
        self.assertTrue(low["ramWarning"])
        self.assertIn("close other apps", " ".join(models.describe(low)[1]))
        self.assertFalse(by(rank([MOE], pc(16, 22.6), free=20.0), "moe30")["ramWarning"])

    def test_no_ram_warning_when_model_fits_gpu(self):
        self.assertFalse(by(rank([SMALL], pc(16, 22.6), free=0.5), "small9")["ramWarning"])

    def test_offline_fallback_is_conservative(self):
        p = by(rank([dict(DEVSTRAL, info=None)], pc(16, 22.6)), "devstral")
        self.assertTrue(p["estimated"])
        self.assertEqual(p["kvGB"], models.FALLBACK_KV_GB)
        self.assertIn("not available offline", " ".join(models.describe(p)[1]))

    def test_models_not_looked_up_are_skipped(self):
        self.assertEqual(rank([dict(MOE, files=None)], pc(16, 22)), [])

    def test_moe_estimated_faster_than_dense_in_ram(self):
        picks = rank([MOE, DEVSTRAL], pc(16, 22.6))
        self.assertGreater(by(picks, "moe30")["speedTokensPerSec"][1], by(picks, "devstral")["speedTokensPerSec"][1])

    def test_text_has_no_benchmark_claims(self):
        text = ""
        for p in rank([DEVSTRAL, MOE, SMALL, BIG], pc(16, 22.6)):
            label, lines = models.describe(p)
            text += label + " ".join(lines)
        text += " ".join(m["note"] for m in models.MODELS)
        for word in ("tested", "reference", "benchmark", "measured"):
            self.assertNotIn(word, text.lower())

    def test_curated_list_has_sizes_and_sources(self):
        for m in models.MODELS:
            self.assertTrue(m["params"] and m["source"].startswith("https://"))
            if m["moe"]:
                self.assertTrue(m["active"])


def gguf(arch="llama", layers=32, heads=32, kv_heads=8, emb=4096, ctx=131072):
    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b

    def num(key, value):
        return s(key) + struct.pack("<II", 4, value)
    items = [s("general.architecture") + struct.pack("<I", 8) + s(arch), num(f"{arch}.block_count", layers),
             num(f"{arch}.attention.head_count", heads), num(f"{arch}.attention.head_count_kv", kv_heads),
             num(f"{arch}.embedding_length", emb), num(f"{arch}.context_length", ctx)]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items)


class Header(unittest.TestCase):
    def test_info_from_partial_bytes(self):
        i = context.info_from_bytes(gguf() + b"\0" * 100)
        self.assertEqual(i["arch"], "llama")
        self.assertEqual(i["layers"], 32)
        self.assertEqual(i["kvPerToken"], int(32 * 8 * 128 * 2 * 34 / 32))

    def test_garbage_is_none(self):
        self.assertIsNone(context.info_from_bytes(b"not a gguf"))

    def test_header_info_uses_cache_and_range(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(models, "HOME", Path(d)):
            with mock.patch.object(models, "_get") as get:
                get.return_value.__enter__.return_value.read.return_value = gguf()
                first = models.header_info("a/b", "f.gguf", 5)
                second = models.header_info("a/b", "f.gguf", 5)
            self.assertEqual(first, second)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args[0][1]["Range"], "bytes=0-24999999")

    def test_offline_lookups_return_none(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(models, "HOME", Path(d)), \
                mock.patch.object(models, "_get", side_effect=OSError):
            self.assertIsNone(models.header_info("a/b", "f.gguf", 5))
            self.assertIsNone(models.repo_files("a/b"))


if __name__ == "__main__":
    unittest.main()
