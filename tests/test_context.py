import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import context as ctx

GB = 2**30
PC = {"vramGB": 15.9, "ramGB": 22.6, "backend": "cuda"}
SETTINGS = {"fitTargetMiB": 1536, "batch": 4096, "kvCacheType": "q8_0"}


def standard(layers=48, kv=4, key=128, heads=32, context=262144, arch="qwen3moe"):
    return {"general.architecture": arch, f"{arch}.block_count": layers, f"{arch}.context_length": context,
            f"{arch}.attention.head_count": heads, f"{arch}.attention.head_count_kv": kv,
            f"{arch}.attention.key_length": key, f"{arch}.attention.value_length": key}


def per_token(meta, kv_type="q8_0"):
    arch, layers, elements = ctx.kv_elements_per_token(meta)
    return elements * ctx.BYTES_PER_ELEMENT[kv_type]


class KvPerToken(unittest.TestCase):
    def test_qwen3_coder_is_52_kb_per_token(self):
        self.assertEqual(per_token(standard()), 48 * 4 * 256 * 34 / 32)
        self.assertAlmostEqual(per_token(standard()) * 65536 / GB, 3.19, places=1)

    def test_hybrid_counts_every_fourth_layer(self):
        meta = standard(layers=40, kv=2, key=256, arch="qwen35moe")
        meta["qwen35moe.full_attention_interval"] = 4
        self.assertAlmostEqual(per_token(meta), 10880)

    def test_mla_is_one_576_latent_per_layer(self):
        meta = standard(layers=47, kv=1, key=576, arch="deepseek2")
        meta.update({"deepseek2.attention.kv_lora_rank": 512, "deepseek2.rope.dimension_count": 64})
        self.assertEqual(per_token(meta), 47 * 576 * 34 / 32)

    def test_per_layer_kv_heads_skip_layers_without_attention(self):
        meta = standard(layers=4, kv=[0, 8, 0, 8], key=64)
        self.assertEqual(per_token(meta, "f16"), 16 * 128 * 2)

    def test_head_sizes_default_to_embedding_over_heads(self):
        meta = {"general.architecture": "llama", "llama.block_count": 2, "llama.embedding_length": 4096,
                "llama.attention.head_count": 32, "llama.context_length": 8192}
        self.assertEqual(per_token(meta, "f16"), 2 * 32 * 256 * 2)  # kv heads default to heads

    def test_unknown_details_give_none(self):
        self.assertIsNone(ctx.kv_elements_per_token({"general.architecture": "mystery"}))
        self.assertIsNone(ctx.kv_elements_per_token({}))

    def test_cache_type_sizes(self):
        meta = standard()
        self.assertEqual(per_token(meta, "f16") / per_token(meta, "q8_0"), 32 / 17)
        self.assertEqual(per_token(meta, "q4_0") / per_token(meta, "f16"), 18 / 64)


class Range(unittest.TestCase):
    def info(self, **kw):
        meta = standard(**kw)
        return {"modelMax": meta[f"{meta['general.architecture']}.context_length"],
                "kvPerToken": per_token(meta)}

    def test_cheap_context_gets_128k_on_a_big_model_partly_in_ram(self):
        info = {"modelMax": 262144, "kvPerToken": 10880}
        rng = ctx.context_range(info, 20.5 * GB, PC, SETTINGS)
        self.assertEqual((rng["min"], rng["recommended"]), (65536, 131072))
        self.assertGreater(rng["max"], 131072)
        self.assertLessEqual(rng["max"], 262144)

    def test_expensive_context_stays_at_the_minimum(self):
        rng = ctx.context_range(self.info(), 17.3 * GB, PC, SETTINGS)
        self.assertEqual(rng["recommended"], 65536)
        self.assertEqual(rng["kvGB"], 3.2)

    def test_small_model_on_the_gpu_uses_the_leftover_memory(self):
        rng = ctx.context_range(self.info(), 5 * GB, PC, SETTINGS)
        self.assertEqual(rng["recommended"], 131072)  # capped: longer prompts take minutes
        self.assertEqual(rng["max"], 262144)

    def test_model_limit_caps_everything(self):
        rng = ctx.context_range({"modelMax": 32768, "kvPerToken": 1000}, 5 * GB, PC, SETTINGS)
        self.assertEqual((rng["min"], rng["recommended"], rng["max"]), (32768, 32768, 32768))

    def test_range_is_ordered_and_in_steps(self):
        for size in (3, 8, 17, 20, 40):
            rng = ctx.context_range(self.info(), size * GB, PC, SETTINGS)
            self.assertLessEqual(rng["min"], rng["recommended"])
            self.assertLessEqual(rng["recommended"], rng["max"])
            self.assertEqual(rng["recommended"] % ctx.STEP, 0)

    def test_cpu_only_and_metal(self):
        cpu = ctx.context_range({"modelMax": 262144, "kvPerToken": 10880}, 5 * GB, {"vramGB": 0, "ramGB": 32}, SETTINGS)
        self.assertEqual(cpu["recommended"], 65536)
        mac = ctx.context_range({"modelMax": 262144, "kvPerToken": 10880}, 20 * GB,
                                {"vramGB": 36, "ramGB": 48, "backend": "metal"}, SETTINGS)
        self.assertEqual(mac["recommended"], 131072)

    def test_no_details_no_range(self):
        self.assertIsNone(ctx.context_range(None, GB, PC, SETTINGS))
        self.assertIsNone(ctx.context_range({"modelMax": 1, "kvPerToken": None}, GB, PC, SETTINGS))

    def test_format(self):
        self.assertEqual((ctx.format_k(65536), ctx.format_k(202752)), ("64K", "198K"))
        self.assertEqual((ctx.parse_k("96K"), ctx.parse_k("98304"), ctx.parse_k("x")), (98304, 98304, None))


class Choice(unittest.TestCase):
    def config(self, **server):
        return {"server": {"context": "auto", **SETTINGS, **server}, "hardware": PC}

    def test_precedence_and_fallback(self):
        model = Path("nowhere/model.gguf")  # unreadable: no range
        config = self.config()
        self.assertEqual(ctx.choose(config, model)[0], 65536)  # old rule: 22.6 + 15.9 >= 24
        self.assertEqual(ctx.choose({**config, "hardware": {"ramGB": 8}}, model)[0], 32768)
        config["server"]["context"] = 49152
        self.assertEqual(ctx.choose(config, model)[:2], (49152, "config"))
        config["contexts"] = {"model.gguf": 81920}
        self.assertEqual(ctx.choose(config, model)[0], 81920)
        self.assertEqual(ctx.choose(config, model, 98304)[:2], (98304, "--context"))

    def test_migration_keeps_user_choices(self):
        for old, new in ((32768, "auto"), (65536, "auto"), (131072, 131072), ("auto", "auto")):
            config = {"server": {"context": old}}
            ctx.migrate(config)
            self.assertEqual(config["server"]["context"], new)
        config = {"server": {"context": 65536}}
        ctx.migrate(config)
        config["server"]["context"] = 65536  # chosen after the migration: stays
        self.assertFalse(ctx.migrate(config))
        self.assertEqual(config["server"]["context"], 65536)


def gguf(path, meta, tokens=3000, vocabulary_first=False):
    """A tiny GGUF header: the given metadata (ints, strings, int lists) and a big string array like a vocabulary."""
    def text(s):
        raw = s.encode()
        return struct.pack("<Q", len(raw)) + raw

    def pair(key, value):
        if isinstance(value, str):
            return text(key) + struct.pack("<I", 8) + text(value)
        if isinstance(value, list):
            return text(key) + struct.pack("<IIQ", 9, 4, len(value)) + struct.pack(f"<{len(value)}i", *value)
        return text(key) + struct.pack("<IQ", 10, value)

    body = b"".join(pair(k, v) for k, v in meta.items())
    vocabulary = text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, tokens) + b"".join(text("tok%d" % i) for i in range(tokens))
    body = vocabulary + body if vocabulary_first else body + vocabulary
    body += text("general.file_type") + struct.pack("<IQ", 10, 7)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta) + 2) + body + b"\0" * 64)


class Header(unittest.TestCase):
    def test_reads_a_file_and_skips_the_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.gguf"
            gguf(path, standard(layers=4, kv=[0, 2, 0, 2], key=64, context=131072))
            info = ctx.model_info(path)
            self.assertEqual((info["arch"], info["modelMax"], info["layers"]), ("qwen3moe", 131072, 4))
            self.assertEqual(info["kvPerToken"], 4 * 128 * 34 / 32)
            self.assertEqual(ctx.model_bytes(path), path.stat().st_size)

    def test_skips_a_vocabulary_that_comes_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.gguf"
            gguf(path, standard(layers=4, kv=2, key=64), vocabulary_first=True)
            self.assertEqual(ctx.read_metadata(path)["general.file_type"], 7)
            self.assertEqual(ctx.model_info(path)["layers"], 4)

    def test_split_model_reads_part_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "m-00001-of-00002.gguf"
            gguf(first, standard(layers=2, kv=1, key=32))
            (Path(tmp) / "m-00002-of-00002.gguf").write_bytes(b"tensors only")
            self.assertEqual(ctx.model_info(Path(tmp) / "m-00002-of-00002.gguf")["layers"], 2)
            self.assertEqual(ctx.model_bytes(first), first.stat().st_size + len(b"tensors only"))

    def test_broken_files_give_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (("a.gguf", b""), ("b.gguf", b"GGUF\x03\0\0\0"), ("c.gguf", b"not a model at all")):
                (Path(tmp) / name).write_bytes(data)
                self.assertIsNone(ctx.model_info(Path(tmp) / name))
            self.assertIsNone(ctx.model_info(Path(tmp) / "missing.gguf"))


class FindModels(unittest.TestCase):
    def test_skips_files_with_readable_metadata_but_no_context_length(self):
        from hearthwork import onboard
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chat = root / "Chat-Q4.gguf"
            speech = root / "Nemotron-3-Diarization.q8_0.gguf"
            unreadable = root / "Broken.gguf"
            for path in (chat, speech, unreadable):
                path.write_bytes(b"")
            infos = {chat: {"arch": "llama", "modelMax": 8192, "layers": 32, "kvPerToken": 1},
                     speech: {"arch": "speech", "modelMax": None, "layers": None, "kvPerToken": None}}

            def fake_info(path, kv_type="q8_0"):
                return infos.get(path)  # None for the unreadable file

            with mock.patch.object(onboard, "model_info", side_effect=fake_info):
                names = [p.name for p in onboard.find_models(root)]
        self.assertEqual(names, ["Broken.gguf", "Chat-Q4.gguf"])


if __name__ == "__main__":
    unittest.main()
