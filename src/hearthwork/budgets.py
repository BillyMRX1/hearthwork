"""Resource budgets: `hearthwork budget [show | use dedicated|balanced|work | set vram=10GB ram=12GB slots=1 threads=8 idle=20m | reset]`.

A preset (or custom numbers) maps onto settings Hearthwork already passes to llama-server:
  VRAM budget  -> --fit-target (headroom llama.cpp keeps free; best effort, the engine places layers by it)
  slots        -> -np (also the host's request concurrency)
  threads      -> -t
  idle         -> the host's idle unload
  RAM budget   -> not a flag: llama.cpp cannot cap itself; Hearthwork estimates the need and refuses a start that would exceed it
Before a model starts or switches, the need (weights + KV cache for the context + runtime overhead) is estimated and compared
with the budget and with the memory that is free now. Nothing is shrunk silently: an impossible start is refused with options.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys

from . import context as ctx
from .onboard import GREEN, RESET, YELLOW, find_models, load_config, memory_gb, update_config

GB = 2**30
OVERHEAD_GB = 1.5   # llama.cpp compute buffers + CUDA/Metal runtime
PRESETS = {
    # fit: VRAM headroom kept free (MiB, None = the computer's usual setting); ram: share of RAM the AI may use
    "dedicated": {"fit": 512, "slots": 2, "threads": None, "idle": None, "ram": 0.85,
                  "text": "most for the AI: little VRAM kept free, 2 slots"},
    "balanced": {"fit": None, "slots": None, "threads": None, "idle": None, "ram": 0.85,
                 "text": "the default: the setup's own settings"},
    "work": {"fit": 5120, "slots": 1, "threads": "half", "idle": 15, "ram": 0.65,
             "text": "gaming / work: ~5 GB VRAM kept free, 1 slot, half the CPU threads, unload after 15 idle minutes"},
}
ENFORCED = ("Enforced by flags: slots (-np), threads (-t), idle unload. VRAM: best effort (llama.cpp --fit-target keeps the "
            "headroom free). RAM: checked before a start (not capped by the OS).")


# ---------- settings ----------

def state(config):
    return config.get("budget") or {}


def hw_of(config):
    return config.get("hardware") or {}


def unified(hw):
    return hw.get("backend") == "metal"


def total_vram(hw):
    return 0.0 if unified(hw) or hw.get("backend") not in ("cuda", "vulkan") else float(hw.get("vramGB") or 0)


def base_values(config):
    b = state(config).get("base")
    if b:
        return b
    return {"fitTargetMiB": config["server"].get("fitTargetMiB", 1536), "slots": config["server"].get("slots", 2),
            "idle": (config.get("host") or {}).get("idleUnloadMinutes") or 0}


def effective(config):
    """What the budget means: preset values over the setup's own, then custom numbers."""
    hw, st = hw_of(config), state(config)
    preset, custom = st.get("preset", "balanced"), st.get("custom") or {}
    p, base = PRESETS[preset], base_values(config)
    threads = p["threads"]
    threads = max(2, (hw.get("cpuThreads") or 4) // 2) if threads == "half" else threads
    fit = p["fit"] or base["fitTargetMiB"]
    vram_total = total_vram(hw)
    if "vramGB" in custom and vram_total:
        fit = max(256, round((vram_total - custom["vramGB"]) * 1024))
    ram_total = hw.get("ramGB") or 0
    ram = custom.get("ramGB", round(p["ram"] * ram_total, 1))
    idle = custom["idle"] if "idle" in custom else (p["idle"] if p["idle"] is not None else base["idle"])
    return {"preset": preset, "custom": bool(custom), "fitTargetMiB": fit,
            "vramGB": round(max(0.0, vram_total - fit / 1024), 1) if vram_total else 0.0,
            "ramGB": ram, "ramEnforced": preset != "balanced" or "ramGB" in custom, "slots": custom.get("slots", p["slots"] or base["slots"]),
            "threads": custom.get("threads", threads), "idleMinutes": idle}


def apply(config):
    """Write the effective budget into the settings the rest of Hearthwork reads. Returns the effective dict."""
    eff = effective(config)
    s = config["server"]
    s["fitTargetMiB"], s["slots"] = eff["fitTargetMiB"], eff["slots"]
    if eff["threads"]:
        s["threads"] = eff["threads"]
    else:
        s.pop("threads", None)
    config.setdefault("host", {})["idleUnloadMinutes"] = eff["idleMinutes"]
    return eff


def set_preset(config, preset, custom=None):
    st = config.get("budget") or {}
    if "base" not in st and (preset != "balanced" or custom):
        st["base"] = base_values(config)
    st["preset"], st["custom"] = preset, custom or {}
    if preset == "balanced" and not custom:
        base = st.pop("base", None)
        if base:  # back to the setup's own values
            config["server"]["fitTargetMiB"], config["server"]["slots"] = base["fitTargetMiB"], base["slots"]
            config.setdefault("host", {})["idleUnloadMinutes"] = base["idle"]
            config["server"].pop("threads", None)
        config["budget"] = st
        return effective(config)
    config["budget"] = st
    return apply(config)


# ---------- estimates ----------

def estimate(weights_gb, kv_gb, hw, eff):
    """Memory a model needs under a budget: {gpuGB, ramGB, totalGB}. The GPU takes what the VRAM budget allows, the rest is RAM.
    Unified memory (Apple): one pool, counted once, as RAM."""
    total = weights_gb + kv_gb + OVERHEAD_GB
    vram = total_vram(hw)
    if not vram:
        return {"gpuGB": 0.0, "ramGB": round(total, 1), "totalGB": round(total, 1)}
    gpu = min(total, eff["vramGB"])
    return {"gpuGB": round(gpu, 1), "ramGB": round(total - gpu, 1), "totalGB": round(total, 1)}


def model_need(config, model, context=None, eff=None):
    """(estimate, context, details) for a model file; None when the file cannot be read."""
    eff = eff or effective(config)
    try:
        chosen = ctx.choose(config, model, context)[0]
        info = ctx.model_info(model, config["server"].get("kvCacheType", "q8_0"))
        weights = ctx.model_bytes(model) / GB
    except (OSError, KeyError):
        return None
    per_token = (info or {}).get("kvPerToken")
    kv = per_token * chosen / GB if per_token else 3.0 * chosen / 65536
    return estimate(weights, kv, hw_of(config), eff), chosen, {"weightsGB": weights, "kvGB": kv, "perToken": per_token}


def check(config, need, context, details, eff=None):
    """Problems (list of strings, each with units) that stop the model from starting under the budget. Empty = viable."""
    eff, hw = eff or effective(config), hw_of(config)
    problems = []
    over = need["ramGB"] - eff["ramGB"]
    if over > 0.05 and eff.get("ramEnforced", True):
        label = "RAM budget" if not unified(hw) else "memory budget (unified memory)"
        problems.append(f"{label}: {eff['ramGB']:.1f} GB, the model needs about {need['ramGB']:.1f} GB in RAM "
                        f"({over:.1f} GB too much)")
    capacity = (hw.get("ramGB") or 0) + total_vram(hw)
    if need["totalGB"] > capacity + 0.05:
        problems.append(f"this computer's memory: {capacity:.1f} GB (RAM + VRAM), the model needs about "
                        f"{need['totalGB']:.1f} GB ({need['totalGB'] - capacity:.1f} GB too much)")
    return problems


def warnings(need, free_ram, reclaim_ram=0.0):
    """Not a reason to refuse (free memory moves): a shortfall against the RAM free right now."""
    if free_ram is None:
        return []
    available = free_ram + reclaim_ram
    if need["ramGB"] > available + 0.05:
        return [f"Warning: free RAM now is {available:.1f} GB and the model needs about {need['ramGB']:.1f} GB "
                f"({need['ramGB'] - available:.1f} GB short). Close other apps if loading is slow or fails."]
    return []


def soft_warnings(need, eff):
    """The default (balanced) budget never refuses on RAM, only says so; presets and ram= are enforced."""
    over = need["ramGB"] - eff["ramGB"]
    if over > 0.05 and not eff.get("ramEnforced", True):
        return [f"Warning: the model needs about {need['ramGB']:.1f} GB of RAM, {over:.1f} GB more than the default budget "
                f"({eff['ramGB']:.1f} GB). Starting anyway; `hearthwork budget set ram=<GB>` makes a limit binding."]
    return []


def log_refusal(text):
    from .paths import HOST_LOG
    import time
    try:
        with open(HOST_LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " budget: refused: " + text.replace(chr(10), " | ") + chr(10))
    except OSError:
        pass


def options(config, problems, need, context, details, eff):
    over = max(need["ramGB"] - eff["ramGB"], 0.0)
    lines = ["Options:", "  - a smaller quantization of this model (about "
             f"{max(over, 0.1):.1f} GB smaller weights at least)"]
    per_token = details.get("perToken")
    if per_token and details["kvGB"] > over:
        smaller = int((details["kvGB"] - over) * GB / per_token) // 4096 * 4096
        if smaller >= 4096:
            lines.append(f"  - a smaller context on purpose: `hearthwork start --context {smaller}` ({ctx.format_k(smaller)}; "
                         f"now {ctx.format_k(context)}). Hearthwork never shrinks it for you")
    lines.append("  - another preset (`hearthwork budget use dedicated` gives the AI the most) or raise a number "
                 "(`hearthwork budget set ram=<GB>`)")
    return lines


def explain(config, model, problems, need, context, details, eff):
    lines = [f"{YELLOW}Not starting {model.name} with the '{budget_name(eff)}' budget{RESET} "
             f"(context {ctx.format_k(context)}):"]
    lines += [f"  - {p}" for p in problems]
    lines.append(f"  estimate: weights {details['weightsGB']:.1f} GB + KV cache {details['kvGB']:.1f} GB + runtime "
                 f"{OVERHEAD_GB:.1f} GB = {need['totalGB']:.1f} GB ({need['gpuGB']:.1f} GB GPU + {need['ramGB']:.1f} GB RAM)")
    return lines + options(config, problems, need, context, details, eff)


def guard(config, model, context=None, reclaim=None, over_budget=False):
    """Before a start or switch: True when `model` can run under the budget. Prints the explanation when not.
    An estimate that cannot be made (unreadable model header) only warns."""
    try:
        eff = effective(config)
        got = model_need(config, model, context, eff)
        if not got:
            return True  # unreadable file: llama-server will say what is wrong
        need, chosen, details = got
        free = memory_gb()[1]
        warns = warnings(need, free, (reclaim or {}).get("ramGB", 0.0)) + soft_warnings(need, eff)
        problems = check(config, need, chosen, details, eff=eff)
        lines = explain(config, model, problems, need, chosen, details, eff) if problems else []
    except Exception as error:  # corrupt GGUF (struct.error, ValueError...), odd config: never block a start on the estimate
        print(f"{YELLOW}Memory estimate unavailable ({type(error).__name__}: {error}); starting without the budget check.{RESET}")
        return True
    for w in warns:
        print(f"{YELLOW}{w}{RESET}")
    if not problems:
        return True
    for line in lines:
        print(line)
    log_refusal(f"{model.name}: " + "; ".join(problems))
    if over_budget:
        print(f"{YELLOW}--over-budget: starting anyway.{RESET}")
        return True
    print("(`--over-budget` starts anyway.)")
    return False


# ---------- measuring ----------

def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def process_rss_gb(pid):
    """Resident memory of a process in GB (tasklist on Windows, /proc or ps elsewhere); None if unknown."""
    if not pid:
        return None
    try:
        if sys.platform == "win32":
            rows = list(csv.reader(run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"]).splitlines()))
            digits = "".join(c for c in rows[0][-1] if c.isdigit()) if rows and len(rows[0]) >= 5 else ""
            return int(digits) * 1024 / GB if digits else None
        if sys.platform.startswith("linux"):
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / GB
        out = run(["ps", "-o", "rss=", "-p", str(pid)]).strip()
        return int(out) * 1024 / GB if out.isdigit() else None
    except (OSError, ValueError, IndexError):
        return None


def gpu_memory_gb(pid=None):
    """{usedGB, freeGB, totalGB, processGB} from nvidia-smi, or None when it is not available."""
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"])
    try:
        used, free, total = [float(x) / 1024 for x in out.splitlines()[0].split(",")]
    except (ValueError, IndexError):
        return None
    process = None
    for line in run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if pid and len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            process = int(parts[1]) / 1024
    return {"usedGB": used, "freeGB": free, "totalGB": total, "processGB": process}


def server_pid():
    from .paths import STATE
    try:
        return json.loads(STATE.read_text()).get("pid")
    except (OSError, ValueError):
        return None


def measure():
    """What the model server uses right now: {pid, ramGB, gpu}; values None when unknown."""
    pid = server_pid()
    return {"pid": pid, "ramGB": process_rss_gb(pid), "gpu": gpu_memory_gb(pid)}


def current_model(config, got=None):
    """Path of the model the host serves (or the last one used), or None."""
    name = (got or {}).get("model")
    try:
        models = find_models(config["modelsDir"])
    except (OSError, KeyError):
        return None
    for m in models:
        if name and name in (m.stem, m.name):
            return m
    last = config.get("lastModel")
    return next((m for m in models if last and (str(m) == last or m.name == last)), None)


# ---------- display ----------

def gb(x):
    return "-" if x is None else f"{x:.1f} GB"


def budget_name(eff):
    """What to call the active budget: the preset, plus a marker when `budget set` changed numbers on top of it."""
    return eff["preset"] + (" + custom numbers" if eff["custom"] else "")


def slot_text(n):
    """Slots with correct English: '1 slot', '2 slots'."""
    return f"{n} slot" if n == 1 else f"{n} slots"


def table(config, got=None, measured=None):
    """Lines: configured budget | estimated need | measured now, per resource, and which limits are enforced."""
    eff, hw = effective(config), hw_of(config)
    model = current_model(config, got) if got is None or got.get("model") else None
    need = None
    if model:
        found = model_need(config, model, (got or {}).get("context"), eff)
        need = found[0] if found else None
    m = measured if measured is not None else (measure() if got else {"ramGB": None, "gpu": None})
    gpu = m.get("gpu")
    gpu_text = "unavailable (no nvidia-smi)"
    if gpu:
        gpu_text = (gb(gpu["processGB"]) + " by llama-server; " if gpu["processGB"] is not None else "") \
            + f"{gb(gpu['usedGB'])} used on the GPU by all programs"
    head = budget_name(eff)
    lines = [f"budget: {GREEN}{head}{RESET}" + (f"  ({PRESETS[eff['preset']]['text']})" if not eff["custom"] else ""),
             f"  {'':<8}{'configured budget':<28}{'estimated need':<20}measured now"]
    if total_vram(hw):
        lines.append(f"  {'VRAM':<8}{gb(eff['vramGB']) + ' (' + str(eff['fitTargetMiB']) + ' MiB free)':<28}"
                     f"{gb(need['gpuGB']) if need else '-':<20}{gpu_text}")
    ram_label = "memory" if unified(hw) else "RAM"
    rss = m.get("ramGB")
    lines.append(f"  {ram_label:<8}{gb(eff['ramGB']):<28}{gb(need['ramGB']) if need else '-':<20}"
                 f"{gb(rss) + ' by llama-server' if rss is not None else 'unavailable'}")
    lines.append(f"  {'slots':<8}{eff['slots']:<28}{'-':<20}{str((got or {}).get('slots', '-')) + ' (host)' if got else '-'}")
    lines.append(f"  {'threads':<8}{(eff['threads'] or 'default'):<28}{'-':<20}-")
    lines.append(f"  {'idle':<8}{(str(eff['idleMinutes']) + ' min') if eff['idleMinutes'] else 'off':<28}{'-':<20}"
                 f"{(str((got or {}).get('idleMinutes')) + ' min') if got and got.get('idleMinutes') else '-' }")
    if model and need:
        lines.append(f"  need for {model.name}: weights + KV cache for the context + {OVERHEAD_GB:.1f} GB runtime")
    lines.append("  " + ENFORCED)
    waiting = pending(config, got)
    if waiting:
        lines.append(f"  {YELLOW}pending (saved, not running yet): {'; '.join(waiting)}. `hearthwork start` applies it when idle.{RESET}")
    return lines


def summary(config, got=None):
    eff = effective(config)
    m = measure() if got else None
    return {"preset": eff["preset"], "custom": eff["custom"], "configured": {k: eff[k] for k in
            ("vramGB", "ramGB", "fitTargetMiB", "slots", "threads", "idleMinutes")},
            "measured": {"serverRamGB": m and m["ramGB"], "gpu": m and m["gpu"]}}


# ---------- command ----------

class Refused(Exception):
    """The memory check said no; nothing is written."""


def pending(config, got):
    """Settings the config wants that the running server was not started with: list of 'name: running -> wanted'.
    Only what Hearthwork itself passed to llama-server is compared (never what the engine reports)."""
    started = (got or {}).get("settings")
    if not got or got.get("adopted") or not got.get("ready") or got.get("transition") or not started:
        return []
    try:
        from .runtime import selected_server
        eff = effective(config)
        model = Path(started["modelPath"])
        wanted_context = ctx.choose(config, model, started.get("requested"))[0]
        pairs = (("VRAM headroom (MiB)", started.get("fitTargetMiB"), eff["fitTargetMiB"]), ("slots", started.get("slots"), eff["slots"]),
                 ("threads", started.get("threads"), eff["threads"]), ("KV cache", started.get("kvCacheType"), config["server"].get("kvCacheType")),
                 ("batch", started.get("batch"), config["server"].get("batch")), ("context", started.get("context"), wanted_context),
                 ("llama.cpp build", started.get("server"), selected_server(config) or config.get("llamaServer")))
    except Exception:
        return []
    return [f"{name}: {old if old is not None else 'default'} -> {new if new is not None else 'default'}" for name, old, new in pairs if old != new]


budget_pending = pending


def parse_gb(text):
    t = text.lower().replace("gib", "").replace("gb", "").replace("g", "").strip()
    value = float(t)
    if value <= 0:
        raise ValueError
    return value


def parse_minutes(text):
    t = text.lower().strip()
    if t in ("off", "0", "never"):
        return 0
    t = t[:-1] if t.endswith("m") else t
    if int(t) < 0:
        raise ValueError
    return int(t)


def parse_set(args):
    out = {}
    for arg in args:
        key, _, value = arg.partition("=")
        key = key.lower()
        try:
            if key == "vram":
                out["vramGB"] = parse_gb(value)
            elif key == "ram":
                out["ramGB"] = parse_gb(value)
            elif key in ("slots", "threads"):
                out[key] = int(value)
                if out[key] < 1:
                    raise ValueError
            elif key == "idle":
                out["idle"] = parse_minutes(value)
            else:
                return None, f"Unknown setting '{key}'. Use vram=, ram=, slots=, threads= or idle=."
        except ValueError:
            return None, f"Cannot read '{arg}'. Examples: vram=10GB ram=12GB slots=1 threads=8 idle=20m"
    return out, None


USAGE = ("Usage: hearthwork budget [show] | use dedicated|balanced|work | set vram=10GB ram=12GB slots=1 threads=8 idle=20m | reset"
         "  [--force] [--over-budget]")


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    from . import host
    force, over_budget = "--force" in argv, "--over-budget" in argv
    argv = [a for a in argv if a not in ("--force", "--over-budget")]
    config = load_config()
    if not config.get("server"):
        print("Run `hearthwork setup` first.")
        return 1
    action = argv[0].lower() if argv else "show"
    if action == "show":
        got = host.status()
        for line in table(config, got):
            print(line)
        return 0
    if action not in ("use", "set", "reset") or (action == "use" and not (len(argv) == 2 and argv[1].lower() in PRESETS))             or (action == "set" and len(argv) < 2):
        print(USAGE)
        return 2
    custom = None
    if action == "set":
        custom, error = parse_set(argv[1:])
        if error:
            print(error)
            return 2
    got = host.status()
    model = current_model(config, got) if got else None
    result = {}

    def change(fresh):
        """Read, decide, check and write under one lock: the memory check sees the merged state."""
        if action == "use":
            result["new"] = set_preset(fresh, argv[1].lower())
        elif action == "set":
            merged = dict(state(fresh).get("custom") or {}, **custom)
            result["new"] = set_preset(fresh, state(fresh).get("preset", "balanced"), merged)
        else:
            result["new"] = set_preset(fresh, "balanced")
        if got and model and got.get("ready") and not got.get("adopted"):
            if not guard(fresh, model, got.get("requestedContext"), {"ramGB": (measure().get("ramGB") or 0.0)}, over_budget):
                raise Refused()

    try:
        config = update_config(change)
    except Refused:
        print("Budget not changed.")
        return 1
    new = result["new"]
    print(f"Budget: {new['preset']}" + (" + custom numbers" if new["custom"] else "")
          + f": VRAM headroom {new['fitTargetMiB']} MiB, {slot_text(new['slots'])}, threads {new['threads'] or 'default'}, "
            f"idle unload {str(new['idleMinutes']) + ' min' if new['idleMinutes'] else 'off'}.")
    if got and model and got.get("ready") and not got.get("adopted"):
        before = len(budget_pending(config, got))
        ok = host.apply_pending(config, host.read_info(), got, force, True, True)
        if before and ok:
            after = host.status()
            left = budget_pending(config, after) if after else []
            if left:
                print(f"{YELLOW}Saved, still pending: {'; '.join(left)}.{RESET}")
                return 1
        return 0 if ok else 1
    if got:
        host.call(host.read_info(), "idle", "POST", payload={"minutes": new["idleMinutes"]})
    print("Applies at the next model start.")
    return 0
