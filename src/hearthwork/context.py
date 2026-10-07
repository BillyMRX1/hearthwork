"""Context size per model and per computer.

The memory a context needs (the KV cache) depends on the model: Qwen3.5-35B-A3B has attention in 1 of 4 layers
(~11 KB per token at q8_0), Qwen3-Coder-30B in every layer (52 KB per token). This module reads the model's
GGUF header, computes that, and from the hardware gives a range:

  min          64K: Claude Code's own prompt is ~20K tokens; below 64K it compacts constantly
  recommended  the largest context whose KV cache keeps the model where it is (on the GPU, or the GPU share
               --fit chose), at most 128K
  max          the largest that fits GPU + RAM, capped at the model's own limit

config.json: server.context is "auto" (default) or a number for all models; contexts[<model file>] overrides it
for one model; `hearthwork start --context N` overrides both for that start.
"""
import json
import re
import struct
from pathlib import Path

STEP = 16384
MIN_CONTEXT = 65536
MAX_RECOMMENDED = 131072  # beyond 128K a long prompt takes minutes to read on a consumer GPU
OS_RESERVE_GB = 6         # RAM that stays with the OS and other programs
BYTES_PER_ELEMENT = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                     "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
HYBRID_ARCHS = {"qwen35", "qwen35moe", "qwen3next"}  # only every `full_attention_interval`-th layer has attention
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
KEEP_ARRAY = 1024  # per-layer numbers (head counts) are kept; token lists are skipped


# ---------- GGUF header ----------

def read_metadata(path):
    """The key/value pairs of a GGUF file's header, or {} if it cannot be read. Big arrays (the vocabulary)
    are skipped without being built. A split model (*-0000N-of-0000M.gguf) keeps its metadata in part 1."""
    path = Path(path)
    part = re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-00001-of-\1.gguf", path.name)
    path = path.with_name(part)
    try:
        with open(path, "rb") as f:
            return _read_header(f)
    except (OSError, struct.error, ValueError, UnicodeDecodeError, MemoryError):
        return {}


def _read_header(f):
    def u(fmt):
        size = struct.calcsize(fmt)
        data = f.read(size)
        if len(data) != size:
            raise ValueError("truncated")
        return struct.unpack("<" + fmt, data)[0]

    def text():
        n = u("Q")
        if n > 1 << 24:
            raise ValueError("bad string")
        return f.read(n).decode("utf-8", "replace")

    def skip_text():
        f.seek(u("Q"), 1)

    def value(kind, keep=True):
        if kind in _SCALARS:
            return u(_SCALARS[kind])
        if kind == 8:
            return text() if keep else skip_text()
        if kind == 9:
            inner, n = u("I"), u("Q")
            if inner in _SCALARS and (not keep or n > KEEP_ARRAY):
                f.seek(n * struct.calcsize("<" + _SCALARS[inner]), 1)
                return None
            items = [value(inner, keep and n <= KEEP_ARRAY) for _ in range(n)]
            return items if keep and n <= KEEP_ARRAY else None
        raise ValueError(f"unknown type {kind}")

    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    u("I")
    u("Q")
    meta = {}
    for _ in range(u("Q")):
        key = text()
        if key.startswith("tokenizer.") and _complete(meta):
            break  # the vocabulary is the bulk of the header; the model's own keys come before it
        meta[key] = value(u("I"))
    return meta


def _complete(meta):
    arch = meta.get("general.architecture")
    return bool(arch) and f"{arch}.block_count" in meta and f"{arch}.context_length" in meta


# ---------- KV cache per token ----------

def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def kv_elements_per_token(meta):
    """(arch, layers, number of values stored per token across all layers) or None if the header lacks what is needed.

    standard: for each layer with attention, kv_heads x (key_length + value_length).
    hybrid (Qwen3.5/3.6, Qwen3-Next): attention in every `full_attention_interval`-th layer only.
    MLA (deepseek2: GLM-4.7-Flash, DeepSeek): one latent per layer, kv_lora_rank + rope dims = 576 elements,
    which the value side shares, so it is counted once.
    Sliding-window models (gemma3, gpt-oss) are counted as full attention in every layer: an upper bound."""
    arch = meta.get("general.architecture")
    if not arch:
        return None
    get = lambda key: meta.get(f"{arch}.{key}")  # noqa: E731
    layers = _num(get("block_count"))
    if not layers:
        return None
    layers = int(layers)
    lora = _num(get("attention.kv_lora_rank"))
    if lora:
        rope = _num(get("rope.dimension_count")) or 0
        width = lora + rope if rope else _num(get("attention.key_length"))
        return (arch, layers, layers * int(width)) if width else None
    heads, kv = get("attention.head_count"), get("attention.head_count_kv")
    if kv is None:
        kv = heads
    embedding = _num(get("embedding_length"))
    per_layer = kv if isinstance(kv, list) else [kv] * layers
    if arch in HYBRID_ARCHS and not isinstance(kv, list):
        per_layer = [kv] * (layers // int(_num(get("full_attention_interval")) or 4))
    if not per_layer or not all(_num(k) is not None for k in per_layer):
        return None
    head = max((h for h in (heads if isinstance(heads, list) else [heads]) if _num(h)), default=None)
    key = _num(get("attention.key_length")) or (embedding / head if embedding and head else None)
    value = _num(get("attention.value_length")) or key
    if not key or not value:
        return None
    return arch, layers, int(sum(per_layer) * (key + value))  # zero entries (no attention in that layer) add nothing


def model_info(path, kv_type="q8_0"):
    """{arch, modelMax, layers, kvPerToken (bytes, None if unknown)} for a GGUF file, or None if unreadable."""
    meta = read_metadata(path)
    arch = meta.get("general.architecture")
    if not arch:
        return None
    found = kv_elements_per_token(meta)
    return {"arch": arch, "modelMax": int(_num(meta.get(f"{arch}.context_length")) or 0) or None,
            "layers": found[1] if found else None,
            "kvPerToken": found[2] * BYTES_PER_ELEMENT.get(kv_type, 2) if found else None}


# ---------- the range ----------

def format_k(tokens):
    return f"{round(tokens / 1024)}K"


def parse_k(text):
    """'96K' or '98304' -> 98304; None if it is neither."""
    match = re.fullmatch(r"(\d+)\s*(k?)", str(text).strip().lower())
    return int(match.group(1)) * (1024 if match.group(2) else 1) if match else None


def _fits(budget_bytes, per_token, lo, hi):
    """Largest multiple of STEP (or `hi` itself when that is smaller) with a KV cache within the budget."""
    steps = int(budget_bytes // per_token) // STEP * STEP
    return max(lo, min(steps, hi))


def context_range(info, model_bytes, hw, settings):
    """{min, recommended, max, modelMax, kvGB, reason} for a model on this computer; None without model details."""
    if not info or not info.get("kvPerToken"):
        return None
    per_token, gb = info["kvPerToken"], 2**30
    model_max = info.get("modelMax") or 131072
    vram, ram = hw.get("vramGB") or 0, hw.get("ramGB") or 0
    unified = hw.get("backend") == "metal"
    size = model_bytes / gb
    smallest = min(MIN_CONTEXT, model_max)
    cap = min(MAX_RECOMMENDED, model_max)
    fit_gb = settings.get("fitTargetMiB", 1024) / 1024
    if unified or vram < 1:  # unified memory (Metal) or CPU only: the KV cache sits in RAM next to the model
        budget = max(2.0, ram * 0.75 - size - 2) * gb
        if vram < 1:
            cap = min(cap, MIN_CONTEXT)  # on a CPU a longer prompt takes minutes to read
        reason = "unified memory" if unified else "CPU only"
    else:
        # llama.cpp's compute buffer grows with the prompt batch (~1.5 GB at 4096).
        usable = vram - fit_gb - max(0.4, 1.5 * settings.get("batch", 4096) / 4096)
        if size <= usable:
            budget, reason = max(2.0, usable - size) * gb, "the model fits in GPU memory; the rest holds the context"
        else:
            budget, reason = max(2.0, 0.12 * vram) * gb, "model partly in RAM; the context is kept small so it keeps its GPU share"
    recommended = _fits(budget, per_token, smallest, cap)
    room = (vram + ram - size - OS_RESERVE_GB - fit_gb) * gb if not unified else (ram * 0.75 - size) * gb
    maximum = max(recommended, _fits(room, per_token, smallest, model_max)) if room > 0 else recommended
    if recommended == smallest and per_token * smallest > budget:
        reason += f"; {format_k(smallest)} is the least Claude Code works well with"
    return {"min": smallest, "recommended": recommended, "max": maximum, "modelMax": model_max,
            "kvGB": round(per_token * recommended / gb, 1), "reason": reason}


# ---------- what a model starts with ----------

def model_bytes(path):
    """All parts of a multi-part model count."""
    path = Path(path)
    if "-00001-of-" in path.name:
        return sum(p.stat().st_size for p in path.parent.glob(path.name.replace("-00001-of-", "-*-of-")))
    return path.stat().st_size


def range_for(config, model):
    """The context range of `model` on this computer, or None if its header does not say enough."""
    try:
        s = config["server"]
        return context_range(model_info(model, s.get("kvCacheType", "q8_0")), model_bytes(model),
                             config.get("hardware") or {}, s)
    except (OSError, KeyError):
        return None


def fallback(config):
    """The old rule, when the model's details are unknown: 64K with 24 GB or more of GPU + RAM, else 32K."""
    hw = config.get("hardware") or {}
    total = (hw.get("ramGB") or 0) + (hw.get("vramGB") or 0 if hw.get("backend") != "metal" else 0)
    return 65536 if total >= 24 else 32768


def choose(config, model, override=None):
    """(context, source, range): --context, the model's saved value, a number in server.context, or auto."""
    saved = (config.get("contexts") or {}).get(Path(model).name)
    fixed = config["server"].get("context")
    fixed = fixed if isinstance(fixed, int) else None
    rng = range_for(config, model)
    for value, source in ((override, "--context"), (saved, "saved for this model"), (fixed, "config")):
        if value:
            return int(value), source, rng
    if rng:
        return rng["recommended"], "auto", rng
    return fallback(config), "auto; model details unknown", None


def describe(context, source, rng):
    line = f"context: {format_k(context)} ({source}"
    if rng:
        line += f"; range {format_k(rng['min'])}-{format_k(rng['max'])}, model max {format_k(rng['modelMax'])}"
    return line + ")"


def migrate(config):
    """Once: 32768 / 65536 in server.context came from the old automatic setup, so they become "auto".
    Any other number was the user's choice and stays. True if config changed."""
    s = config.get("server")
    if not s or config.get("contextMigrated"):
        return False
    config["contextMigrated"] = True
    if s.get("context") in (32768, 65536):
        s["context"] = "auto"
    return True


def main(argv, config):
    """`hearthwork context [model] [N|auto]`: the ranges, or save / clear a model's context."""
    from .onboard import find_models, save_config
    models = find_models(config["modelsDir"])
    hint = argv[0] if argv else None
    if hint:
        models = [m for m in models if hint.lower() in m.name.lower()]
        if len(models) != 1:
            print(f"'{hint}' matches {len(models)} models; use a more specific part of the file name.")
            return 1
    if len(argv) > 1:
        value, name = argv[1].lower(), models[0].name
        contexts = config.setdefault("contexts", {})
        if value == "auto":
            contexts.pop(name, None)
        elif parse_k(value):
            contexts[name] = parse_k(value)
        else:
            print("Give a number of tokens (98304 or 96K), or auto.")
            return 1
        save_config(config)
    for model in models:
        context, source, rng = choose(config, model)
        print(f"{model.name}\n  {describe(context, source, rng)}" + (f"\n  {rng['kvGB']} GB of KV cache; {rng['reason']}" if rng else ""))
    if not hint:
        print("\nSet one with `hearthwork context <model> <N|auto>`; `hearthwork start --context N` is for one start only.")
    return 0
