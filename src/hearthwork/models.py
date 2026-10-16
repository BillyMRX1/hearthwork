"""Which models suit this computer, ranked from its hardware and the models' published properties.

Shared by `hearthwork check` and setup's suggestions. The ranking (`rank`) is a pure function of the hardware, free
RAM, each model's file sizes and its KV-cache size per token (read from the GGUF header on Hugging Face with a
range request, cached on disk). Order of importance:
  1. usable first: fits the GPU, or a MoE partly in RAM estimated at >= 15 tokens/s; then the slower ones (dense
     partly in RAM, slow MoE); then too big. Memory includes that model's own KV cache at the context Hearthwork
     would use, see context.py
  2. quantization: Q3 is allowed (with a note); Q2 is never recommended, only listed last in its class
  3. capability: the curated "tier" (from the maker's published coding/agent benchmarks), else parameter count
  4. estimated speed
"""
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .context import context_range, info_from_bytes
from .paths import HOME

# Coding models with tool-calling chat templates (checked on Hugging Face, 2026-10).
# "moe": only a few experts run per token, so the model stays usable when part of it sits in system RAM.
# "params"/"active": billions of parameters in total / used per token (MoE). "tier": capability class 1-5 for
# coding agents: Hearthwork's judgement from size, active parameters and the maker's model card ("source");
# not measured by Hearthwork. A model without a tier is ranked by parameter count.
MODELS = [
    {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "name": "Qwen3.5-35B-A3B", "moe": True, "params": 35, "active": 3, "tier": 4,
     "source": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B", "note": "MoE general model with strong coding and tool use"},
    {"repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "name": "Qwen3-Coder-30B-A3B", "moe": True, "params": 30,
     "active": 3, "tier": 3, "source": "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct",
     "note": "MoE coding model made for agents"},
    {"repo": "unsloth/GLM-4.7-Flash-GGUF", "name": "GLM-4.7-Flash (30B MoE)", "moe": True, "params": 30, "active": 3, "tier": 3,
     "source": "https://huggingface.co/zai-org/GLM-4.7-Flash", "note": "MoE model for coding and agents"},
    {"repo": "ggml-org/gpt-oss-20b-GGUF", "name": "gpt-oss-20b", "moe": True, "params": 21, "active": 3.6, "tier": 3,
     "source": "https://huggingface.co/openai/gpt-oss-20b", "note": "small MoE; good at tool use"},
    {"repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "name": "Devstral-Small-2-24B", "moe": False,
     "params": 24, "tier": 4, "source": "https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512",
     "note": "dense coding agent; needs to fit the GPU to be fast"},
    {"repo": "unsloth/Qwen3.5-27B-GGUF", "name": "Qwen3.5-27B", "moe": False, "params": 27, "tier": 4,
     "source": "https://huggingface.co/Qwen/Qwen3.5-27B", "note": "dense general model; needs to fit the GPU to be fast"},
    {"repo": "unsloth/Qwen3-Coder-Next-GGUF", "name": "Qwen3-Coder-Next (80B MoE)", "moe": True, "params": 80, "active": 3,
     "tier": 5, "source": "https://huggingface.co/Qwen/Qwen3-Coder-Next", "note": "largest coding MoE here; needs a big machine"},
    {"repo": "lmstudio-community/Qwen3.5-9B-GGUF", "name": "Qwen3.5-9B", "moe": False, "params": 9, "tier": 2,
     "source": "https://huggingface.co/Qwen/Qwen3.5-9B", "note": "small machines only; weaker with the agents' tools"},
]
# Best quality first. Below Q3, answers get noticeably worse.
QUANTS = ["Q4_K_M", "UD-Q4_K_XL", "MXFP4", "Q4_K_S", "Q3_K_M", "UD-Q3_K_XL", "Q3_K_S", "UD-Q2_K_XL", "Q2_K"]
FALLBACK_KV_GB = 6.0  # KV cache at 64K when a header can't be read: the old 3.4 GB was one model's, so be cautious
FALLBACK_CONTEXT = 65536
HEADER_BYTES = 25_000_000
RAM_SPEED_GBS, EFFICIENCY = 40.0, 0.5  # dual-channel system RAM; real speed is about half the memory-bandwidth bound
GPU_SPEED_GBS = {"cuda": 300.0, "vulkan": 250.0, "metal": 150.0}  # typical class of that kind of GPU; no card lookup
USABLE_TOKENS_PER_SEC = 15  # a MoE partly in RAM counts as usable at this estimated speed (low end)
FITS = {0: "fast", 1: "good", 2: "slow", 3: "too big"}


def quant_group(quant):
    """0 = Q4 or better, 1 = Q3, 2 = Q2 (never recommended)."""
    return 2 if "Q2" in quant else 1 if "Q3" in quant else 0


# ---------- Hugging Face ----------

def _get(url, headers=None, timeout=20):
    request = urllib.request.Request(url, headers={"User-Agent": "hearthwork-check", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def repo_files(repo):
    """{quant: {"gb": total size, "file": first part's name, "bytes": first part's size}} from Hugging Face; None offline."""
    try:
        with _get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true") as r:
            files = json.load(r)
    except Exception:
        return None
    found = {}
    for f in sorted(files, key=lambda f: f.get("path", "")):
        path = f.get("path", "")
        if not path.endswith(".gguf") or "mmproj" in path.lower():
            continue
        size = (f.get("lfs") or {}).get("size", f.get("size", 0))
        name = re.sub(r"-\d{5}-of-\d{5}", "", Path(path).name).upper()
        for quant in QUANTS:
            if name.endswith(f"-{quant.upper()}.GGUF") or name.endswith(f"_{quant.upper()}.GGUF"):
                entry = found.setdefault(quant, {"gb": 0.0, "file": path, "bytes": size})
                entry["gb"] += size / 2**30
    return found


def header_info(repo, file, size, kv_type="q8_0"):
    """{arch, modelMax, layers, kvPerToken} from the first 25 MB of a model file; cached by repo+file+size."""
    cache = HOME / "cache" / "headers" / (re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}__{file}__{size}") + f".{kv_type}.json")
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        with _get(f"https://huggingface.co/{repo}/resolve/main/{file}", {"Range": f"bytes=0-{HEADER_BYTES - 1}"}, 60) as r:
            info = info_from_bytes(r.read(HEADER_BYTES), kv_type)
    except Exception:
        return None
    if info:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(info), encoding="utf-8")
        except OSError:
            pass
    return info


def gather(models=MODELS):
    """Each model with its "files" (None offline) and "info" (None when the header couldn't be read), looked up
    concurrently. One header per model: the KV cache per token is the same for every quantization of it."""
    def look(model):
        files = repo_files(model["repo"])
        smallest = min(files.values(), key=lambda e: e["gb"]) if files else None
        info = header_info(model["repo"], smallest["file"], smallest["bytes"]) if smallest else None
        return {**model, "files": files, "info": info}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(look, models))


# ---------- the ranking (pure) ----------

def speed_estimate(model, size_gb, gpu_gb, hw):
    """(low, high) tokens/s, a rough estimate: each token reads the active weights once, so speed is bounded by memory
    bandwidth. Uses a typical bandwidth for the GPU kind and for system RAM, and half of that bound."""
    gpu_speed = GPU_SPEED_GBS.get(hw.get("backend"), RAM_SPEED_GBS)
    active_gb = size_gb * (model["active"] / model["params"] if model.get("moe") and model.get("active") else 1)
    on_gpu = min(1.0, gpu_gb / size_gb) if size_gb else 1.0
    seconds = active_gb * (on_gpu / gpu_speed + (1 - on_gpu) / RAM_SPEED_GBS)
    mid = EFFICIENCY / seconds if seconds else 0
    return max(1, round(mid * 0.7)), max(2, round(mid * 1.3))


def evaluate(model, gb, info, hw, ram_free, settings):
    """How one quantization (of `gb` GB) of a model sits on this computer."""
    ctx = context_range(info, gb * 2**30, hw, settings)
    estimated = ctx is None
    kv = FALLBACK_KV_GB if estimated else info["kvPerToken"] * ctx["recommended"] / 2**30
    context = FALLBACK_CONTEXT if estimated else ctx["recommended"]
    unified = hw.get("backend") == "metal"
    vram = (hw.get("vramGB") or 0) if hw.get("backend") in ("cuda", "vulkan", "metal") else 0
    gpu_total = max(0.0, vram - settings.get("fitTargetMiB", 1536) / 1024)  # GPU memory left for the model + its context
    ram_room = 0.0 if unified else max(0.0, (hw.get("ramGB") or 0) - 6)
    gpu_room = max(0.0, gpu_total - kv)
    if gb <= gpu_room:
        tier = 0
    elif gb <= gpu_room + ram_room:
        tier = 1 if model["moe"] else 2
    else:
        tier = 3
    ram_need = 0.0 if unified else max(0.0, gb + kv - gpu_total)
    return {"tier": tier, "kvGB": round(kv, 1), "context": context, "estimated": estimated,
            "ramNeedGB": round(ram_need, 1), "ramWarning": ram_need > ram_free,
            "speed": speed_estimate(model, gb, min(gb, gpu_room), hw) if tier < 3 else None}


def place(model, files, info, hw, ram_free, settings):
    """Best quantization for this machine -> evaluation dict with "quant" and "sizeGB", or None if none fits.
    Q4+/Q3 are tried first (MoE: best quality that fits at all; dense: fitting the GPU matters most); Q2 only as a
    last resort."""
    order = [q for q in QUANTS if q in files]
    for allowed in ([q for q in order if quant_group(q) < 2], [q for q in order if quant_group(q) == 2]):
        results = [{**evaluate(model, files[q]["gb"], info, hw, ram_free, settings), "quant": q, "sizeGB": round(files[q]["gb"], 1)}
                   for q in allowed]
        fitting = [r for r in results if r["tier"] < 3]
        if not model["moe"]:
            fitting = [r for r in fitting if r["tier"] == 0] or fitting
        if fitting:
            return fitting[0]
    return None


def capability(model):
    if model.get("tier"):
        return model["tier"]
    p = model.get("params") or 0
    return 1 if p < 7 else 2 if p < 14 else 3 if p < 28 else 4


def rank(candidates, hw, ram_free, settings):
    """Pure ranking. candidates: MODELS entries plus "files" ({quant: {"gb"...}} or None) and "info" (a
    context.model_info result or None). Returns picks, best first, each with quant, sizeGB, fit, flags and a speed
    estimate. Models without "files" (not looked up) are left out."""
    keyed = []
    for index, c in enumerate(candidates):
        if c.get("files") is None:
            continue
        placed = place(c, c["files"], c.get("info"), hw, ram_free, settings)
        base = {k: v for k, v in c.items() if k not in ("files", "info")}
        if not placed:
            keyed.append(((2, 0, 0, 0, 0, index), {**base, "quant": None, "sizeGB": None, "fit": "too big", "recommended": False}))
            continue
        group = quant_group(placed["quant"])
        usable = placed["tier"] == 0 or (placed["tier"] == 1 and placed["speed"][0] >= USABLE_TOKENS_PER_SEC)
        key = (0 if usable else 1, 1 if group == 2 else 0, -capability(c), -(c.get("params") or 0), -placed["speed"][1], index)
        keyed.append((key, {**base, **{k: placed[k] for k in ("quant", "sizeGB", "kvGB", "context", "estimated", "ramNeedGB", "ramWarning")},
                            "fit": FITS[placed["tier"]], "lowQuality": group == 2, "q3": group == 1,
                            "speedTokensPerSec": list(placed["speed"]), "recommended": group < 2}))
    return [pick for _, pick in sorted(keyed, key=lambda kp: kp[0])]


def best(picks):
    """The top pick worth recommending: it fits (GPU, or GPU + RAM) and is not a Q2."""
    return next((p for p in picks if p["fit"] != "too big" and p.get("recommended")), None)


def describe(pick):
    """(fit label, detail lines) as plain text for a pick."""
    label = {"fast": "fits the GPU: fast", "good": "GPU + RAM: good (MoE)", "slow": "GPU + RAM: slow (dense)",
             "too big": "too big"}[pick["fit"]]
    if pick.get("lowQuality"):
        label += "; possible, lower quality"
    lines = [pick["note"]]
    if pick.get("speedTokensPerSec"):
        lo, hi = pick["speedTokensPerSec"]
        lines.append(f"about {lo}-{hi} tokens/s (rough estimate from model size and typical memory speed)")
    if pick.get("kvGB") is not None:
        lines.append(f"context memory {pick['kvGB']} GB at {pick['context'] // 1024}K"
                     + (" (estimated, model details not available offline)" if pick.get("estimated") else ""))
    if pick.get("q3"):
        lines.append("Q3: smaller, somewhat lower quality; more GPU memory or RAM would allow Q4")
    if pick.get("ramWarning"):
        lines.append(f"needs about {pick['ramNeedGB']} GB of RAM and less is free now: close other apps first")
    return label, lines
