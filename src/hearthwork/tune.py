#!/usr/bin/env python3
"""Automatic tuning: try a few configurations of what is installed and recommend the best tested one.

  hearthwork tune [--time 30m] [--min-context N] [--agent claude] [--quick-tasks N] [--dry-run] [--json]

Candidates come from the models already on this computer (never a download): every distinct model that fits the
budget, plus at most two extras - the current model at the next smaller context step, and a smaller quantization of
the same model family when one is installed. At most five candidates. Each one is switched to through the host
(sessions open abort the run), warmed up, then measured with a SHORT graded bench (the first N of bench.py's prompts,
default 3): task score, total seconds, first-token latency, median generation speed and peak memory. The original
model is restored at the end, always, even on Ctrl+C or an exception. The winner is saved as profile `tune-<date>`
(evidence: this run) and printed with the command that applies it - never applied for you.

The result is "best among the configurations tested" on this computer at this time: it is not a universal optimum,
and a machine, runtime or model change can change it. Every run is appended to tune.jsonl (rotated at 5 MB).
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import bench, budgets, host
from . import context as ctx
from .harnesses import HARNESSES, installed
from .onboard import CYAN, GREEN, RED, RESET, YELLOW, find_models, load_config
from .paths import HOME

LOG = HOME / "tune.jsonl"
LOG_ROTATE = 5 * 1024 * 1024
MAX_CANDIDATES = 5
DEFAULT_TASKS = 3
DEFAULT_TIME = 30 * 60
LOAD_ESTIMATE = 60.0            # assumed seconds to load a model until the first real switch is measured
SAMPLE_INTERVAL = 2.0           # how often peak memory is sampled while a candidate runs
CLOSE_TOLERANCE = 0.10          # best and second-best this close are re-run once

QUANT = re.compile(r"(?i)(ud-)?(?:q|iq)\d+(?:_[a-z0-9]+)*|mxfp4|f16|bf16|q8_0")


# ---------- small helpers ----------

def parse_time(text):
    """Seconds from '30m', '90s', '1h' or a bare number of minutes ('30'). Raises ValueError on nonsense."""
    t = str(text).strip().lower()
    factor = 1
    if t.endswith("m"):
        t, factor = t[:-1], 60
    elif t.endswith("h"):
        t, factor = t[:-1], 3600
    elif t.endswith("s"):
        t, factor = t[:-1], 1
    else:
        factor = 60  # a bare number is minutes, like --time 30
    value = float(t)
    if value <= 0:
        raise ValueError(text)
    return int(value * factor)


def family(stem):
    """The model family of a file stem, with its quantization tokens removed: 'Qwen3-14B-Q4_K_M' -> 'qwen3-14b'."""
    stripped = QUANT.sub("", stem).lower()
    return re.sub(r"[-_.]+$", "", re.sub(r"[-_.]{2,}", "-", stripped)) or stem.lower()


def estimate_cost(load_seconds, per_task_seconds, tasks):
    """Seconds a candidate is expected to need: a load plus `tasks` at the observed average."""
    return (load_seconds or LOAD_ESTIMATE) + (per_task_seconds or LOAD_ESTIMATE) * tasks


def score_seconds(result):
    """Seconds per point of score: a lower value is a better configuration. Inf without a score."""
    score = result.get("score_fraction") or 0.0
    return result["seconds"] / score if score > 0 else float("inf")


def close_pair(a, b, tolerance=CLOSE_TOLERANCE):
    """Are two results within `tolerance` of each other on score-seconds? (the repeat rule)"""
    sa, sb = score_seconds(a), score_seconds(b)
    if not (math.isfinite(sa) and math.isfinite(sb)):
        return a.get("score_fraction") == b.get("score_fraction")
    lo, hi = sorted((sa, sb))
    return lo > 0 and hi <= lo * (1 + tolerance)


def rank(results):
    """Best first: task score (fraction), then total seconds, then first-token latency."""
    return sorted(results, key=lambda r: (-(r.get("score_fraction") or 0.0), r["seconds"],
                                          r["ttfb_ms"] if r.get("ttfb_ms") is not None else float("inf")))


# ---------- candidates ----------

def fits(config, path, context):
    """(need, None) when the model fits the budget, else (None, a one-line reason). Never prints."""
    try:
        got = budgets.model_need(config, path, context)
    except Exception as error:  # unreadable header: llama-server decides, treat as fitting
        return None, f"estimate unavailable ({type(error).__name__})"
    if not got:
        return None, "the model header could not be read"
    need, chosen, details = got
    try:
        problems = budgets.check(config, need, chosen, details)
    except Exception:
        problems = []
    if problems:
        return need, problems[0]
    return need, None


def _context_for(config, path, min_context):
    """(context, None) for a model, raised to `min_context` when it can reach it, else (None, a note)."""
    context = ctx.choose(config, path)[0]
    if min_context and context < min_context:
        rng = ctx.range_for(config, path)
        if rng and rng["max"] < min_context:
            return None, f"{path.name}: {ctx.format_k(min_context)} is above this model's range here; skipped"
        return min_context, None
    return context, None


def _primary_candidates(config, paths, current, min_context, notes):
    out = []
    seen = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        context, note = _context_for(config, path, min_context)
        if note:
            notes.append(note)
            continue
        need, problem = fits(config, path, context)
        if problem:
            notes.append(f"{path.name}: skipped ({problem})")
            continue
        out.append({"model": path, "context": context, "need": need,
                    "label": f"{path.name} @ {ctx.format_k(context)}", "kind": "model",
                    "current": bool(current and path.name == current.name)})
    return out


def _extras(config, current, chosen, min_context, notes):
    """At most two extras: the current model one context step smaller, and a smaller quantization of its family."""
    out = []
    if not current:
        return out
    auto = ctx.choose(config, current)[0]
    rng = ctx.range_for(config, current)
    smaller = auto - ctx.STEP
    if rng:
        smaller = max(rng["min"], smaller)
    if smaller < auto and (not min_context or smaller >= min_context):
        need, problem = fits(config, current, smaller)
        if problem:
            notes.append(f"{current.name} at {ctx.format_k(smaller)}: skipped ({problem})")
        else:
            out.append({"model": current, "context": smaller, "need": need,
                        "label": f"{current.name} @ {ctx.format_k(smaller)}",
                        "kind": "context", "current": True})
    try:
        size = current.stat().st_size
    except OSError:
        size = None
    same = []
    for path in chosen:
        if path.name == current.name or family(path.stem) != family(current.stem):
            continue
        try:
            if size is None or path.stat().st_size < size:
                same.append((path.stat().st_size if size is None else size - path.stat().st_size, path))
        except OSError:
            continue
    for _, path in sorted(same)[:1]:  # the closest smaller file, i.e. the next quantization down
        context, note = _context_for(config, path, min_context)
        if note:
            notes.append(note)
            continue
        need, problem = fits(config, path, context)
        if problem:
            notes.append(f"{path.name}: skipped ({problem})")
            continue
        out.append({"model": path, "context": context, "need": need,
                    "label": f"{path.name} (smaller quant)", "kind": "quant", "current": False})
        break
    return out


def build_candidates(config, min_context=None, current=None, models=None, limit=MAX_CANDIDATES):
    """(candidates, notes): a bounded list of configurations to try, and why others were left out."""
    paths = models if models is not None else find_models(config.get("modelsDir") or ".")
    if current is None:
        current = budgets.current_model(config)
    notes = []
    extras = _extras(config, current, paths, min_context, notes)  # variations of what is running go first
    candidates = extras + _primary_candidates(config, paths, current, min_context, notes)
    seen, unique = set(), []
    for c in candidates:  # at most one candidate per (model file, context)
        key = str(c["model"].resolve()).lower() + ":" + str(c["context"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:limit], notes


# ---------- measuring one candidate ----------

def switch_to(config, model, context):
    """Point the running host at `model` (context), through the host's drain path. True when it is ready."""
    return host.ensure(config, model, context)


def measure():
    """What the model server uses right now (peak-memory sampling)."""
    return budgets.measure()


class MemorySampler(threading.Thread):
    """Samples the model server's memory every SAMPLE_INTERVAL seconds and keeps the peaks."""

    def __init__(self, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_ramGB = None
        self.peak_gpuGB = None

    def sample(self):
        try:
            now = measure()
        except Exception:
            return
        ram = now.get("ramGB")
        if ram is not None:
            self.peak_ramGB = ram if self.peak_ramGB is None else max(self.peak_ramGB, ram)
        gpu = now.get("gpu") or {}
        used = gpu.get("processGB") if gpu.get("processGB") is not None else gpu.get("usedGB")
        if used is not None:
            self.peak_gpuGB = used if self.peak_gpuGB is None else max(self.peak_gpuGB, used)

    def run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def stop(self):
        self.stop_event.set()
        self.sample()  # one last look, so a one-shot config still records something


def diagnostics_metrics(config, seconds):
    """First-token latency (median ms) and median generation speed from the host's diagnostics ring."""
    info = host.read_info()
    if not info:
        return {"ttfb_ms": None, "gen_tps": None}
    payload = host.call(info, f"diagnostics?seconds={max(30, int(seconds) + 30)}", timeout=10) or {}
    ttfb = payload.get("ttfb_ms") or {}
    return {"ttfb_ms": ttfb.get("p50"), "gen_tps": payload.get("gen_tps_median")}


def run_candidate(config, candidate, agent, tasks, root, say=print):
    """Switch to `candidate`, warm it up, run the short bench, and return a result, or None when it is skipped."""
    need, problem = fits(config, candidate["model"], candidate["context"])
    if problem:
        say(f"{YELLOW}  {candidate['label']}: skipped ({problem}){RESET}")
        return None
    say(f"{BOLD}Trying{RESET} {candidate['label']}  {DIM}({agent}, {tasks} tasks){RESET}", flush=True)
    started_switch = time.monotonic()
    if not switch_to(config, candidate["model"], candidate["context"]):
        say(f"{YELLOW}  could not switch to it; skipped{RESET}")
        return None
    load_seconds = time.monotonic() - started_switch
    try:
        bench.warmup(host.port(config), candidate["model"].stem)  # through the host, like bench
    except Exception as error:  # a warmup hiccup is not a reason to drop the candidate
        say(f"{DIM}  warmup skipped ({type(error).__name__}){RESET}")
    sampler = MemorySampler()
    sampler.start()
    started = time.monotonic()
    try:
        record = bench.run_agent(agent, config, candidate["model"].stem, root, tasks=tasks)
    finally:
        sampler.stop()
    elapsed = time.monotonic() - started
    if record is None:
        say(f"{YELLOW}  the agent itself failed: no score for this candidate{RESET}")
        return None
    diag = diagnostics_metrics(config, elapsed)
    score = record.get("score") or 0.0
    result = {"label": candidate["label"], "kind": candidate["kind"], "model": candidate["model"].stem,
              "context": candidate["context"], "score": score, "tasks": tasks,
              "score_fraction": round(score / tasks, 3), "seconds": round(record.get("seconds") or elapsed, 1),
              "load_seconds": round(load_seconds, 1), "ttfb_ms": diag["ttfb_ms"], "gen_tps": diag["gen_tps"],
              "peak_ramGB": sampler.peak_ramGB, "peak_gpuGB": sampler.peak_gpuGB, "agents": record.get("agent")}
    say(f"  score {score:g}/{tasks}  {result['seconds']:.0f} s"
        + (f"  first token {result['ttfb_ms']:.0f} ms" if result["ttfb_ms"] is not None else "")
        + (f"  {result['gen_tps']:.1f} tok/s" if result["gen_tps"] is not None else ""))
    return result


# ---------- reporting ----------

BOLD, DIM = "\033[1m", "\033[2m"


def report_table(results):
    """A markdown table of the tested configurations."""
    rows = ["| Configuration | Context | Score | Time | First token | Generation | Peak RAM |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['label']} | {ctx.format_k(r['context'])} | {r['score']:g}/{r['tasks']} "
                    f"({r['score_fraction']:.0%}) | {r['seconds']:.0f} s | "
                    + (f"{r['ttfb_ms']:.0f} ms | " if r["ttfb_ms"] is not None else "- | ")
                    + (f"{r['gen_tps']:.1f} tok/s | " if r["gen_tps"] is not None else "- | ")
                    + (f"{r['peak_ramGB']:.1f} GB |" if r["peak_ramGB"] is not None else "- |"))
    return "\n".join(rows)


def candidates_text(config, candidates, notes, tasks, min_context=None):
    """The --dry-run listing: each candidate and what it is estimated to need."""
    lines = [f"{BOLD}Candidates{RESET}  {DIM}({len(candidates)} of at most {MAX_CANDIDATES}; nothing is downloaded "
             f"or switched){RESET}"]
    if min_context:
        lines.append(f"  minimum context: {ctx.format_k(min_context)}")
    for c in candidates:
        need = c.get("need") or {}
        est = (f"weights+KV {need.get('totalGB', 0):.1f} GB total "
               f"({need.get('gpuGB', 0):.1f} GB GPU, {need.get('ramGB', 0):.1f} GB RAM)") if need else "estimate unavailable"
        lines.append(f"  {c['label']:<46} {ctx.format_k(c['context']):>6}  {est}"
                     + ("  (current)" if c.get("current") else ""))
    for note in notes:
        lines.append(f"  {DIM}- {note}{RESET}")
    lines.append(f"\nEach candidate is switched to, warmed up and measured with the first {tasks} bench tasks; "
                 "the original model is restored afterwards.")
    return "\n".join(lines)


# ---------- saving ----------

def winner_profile(config, name, winner, run_record):
    """A profile that records the winning configuration (model, context, settings) with this run as its evidence."""
    from . import setups
    c = copy.deepcopy(config)
    c.setdefault("server", {})["port"] = 0          # capture resolves the model from lastModel, not the running server
    c["lastModel"] = str(winner["model"])
    c.setdefault("contexts", {})[winner["model"].name] = winner["context"]
    profile = setups.capture(c, name, evidence=False)
    profile["evidence"] = {"doctor": None, "bench": None, "tune": run_record}
    profile["status"] = setups.VERIFIED
    setups.store(profile)
    return profile


def append_log(record, path=None, rotate_at=LOG_ROTATE):
    """Append the run to tune.jsonl, rotating to tune.jsonl.1 once the file passes 5 MB."""
    path = path or LOG
    try:
        if path.exists() and path.stat().st_size > rotate_at:
            os.replace(path, path.with_name(path.name + ".1"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ---------- the run ----------

def restore(config, original, say=print):
    """Put the original model and context back through the host. Never raises."""
    model, context = original.get("model"), original.get("context")
    if not model:
        return False
    say(f"{CYAN}Restoring the original model ({Path(model).name}).{RESET}")
    try:
        return bool(switch_to(config, Path(model), context))
    except Exception as error:
        say(f"{YELLOW}Could not restore automatically ({type(error).__name__}: {error}); "
            f"start it again with `hearthwork start`.{RESET}")
        return False


def run(config, agent, tasks, seconds_limit, min_context, say=print, as_json=False, now=None):
    """Try the candidates in order until the time is up, then rank, save and report. Returns an exit code."""
    status = host.status()
    if not status or not status.get("ready"):
        say("No model is running. Start one first (`hearthwork start`).")
        return 1
    sessions = status.get("sessions") or []
    if sessions:
        say(f"{YELLOW}Close your agent sessions first: {len(sessions)} agent session(s) are using the model:"
            f"{RESET}")
        for s in sessions:
            say(f"  {host.session_line(s)}")
        return 1
    model_path = (status.get("settings") or {}).get("modelPath")
    original = {"model": model_path, "context": status.get("context") or status.get("requestedContext")}
    current = budgets.current_model(config, status)
    candidates, notes = build_candidates(config, min_context, current=current)
    if not candidates:
        say("No installed model fits the budget. Nothing to test (see `hearthwork budget`).")
        return 1
    date = (now or datetime.now()).strftime("%Y-%m-%d")
    root = bench.BENCH / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-tune")
    deadline = time.monotonic() + seconds_limit
    say(f"{BOLD}Auto-tuning{RESET}: {len(candidates)} candidate(s), {tasks} task(s) each, "
        f"{seconds_limit / 60:.0f} min limit  {DIM}(best among the configurations tested){RESET}\n")
    results, per_task, load_est = [], None, LOAD_ESTIMATE
    try:
        for candidate in candidates:
            remaining = deadline - time.monotonic()
            cost = estimate_cost(load_est, per_task, tasks)
            if results and remaining < cost:
                say(f"{DIM}Stopping: {remaining / 60:.0f} min left would not cover another candidate "
                    f"(~{cost / 60:.0f} min).{RESET}")
                break
            try:
                result = run_candidate(config, candidate, agent, tasks, root, say)
            except Exception as error:  # a broken candidate never ends the run; the original is still restored below
                say(f"{YELLOW}  {candidate['label']} failed ({type(error).__name__}: {error}); skipping it{RESET}")
                result = None
            if result is None:
                continue
            results.append(result)
            per_task = (result["seconds"] / tasks) if tasks else None
            load_est = result["load_seconds"] or load_est
        if results:
            ranked = rank(results)
            if len(ranked) > 1 and close_pair(ranked[0], ranked[1]):  # too close to call: one repeat
                cost = estimate_cost(load_est, per_task, tasks)
                if deadline - time.monotonic() >= cost:
                    say(f"{DIM}The top two are within {CLOSE_TOLERANCE:.0%} on score-seconds: re-running the best "
                        f"once.{RESET}")
                    best = next(c for c in candidates if c["label"] == ranked[0]["label"])
                    try:
                        repeat = run_candidate(config, best, agent, tasks, root, say)
                    except Exception as error:
                        say(f"{YELLOW}  repeat failed ({type(error).__name__}: {error}){RESET}")
                        repeat = None
                    if repeat:
                        repeat["repeat"] = True
                        results.append(repeat)
    finally:
        restore(config, original, say)
    if not results:
        say("No candidate finished: nothing to recommend.")
        return 1
    ranked = rank(results)
    winner = ranked[0]
    winner_candidate = next(c for c in candidates if c["label"] == winner["label"])
    run_record = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "agent": agent, "tasks": tasks,
                  "timeLimitMinutes": round(seconds_limit / 60), "minContext": min_context,
                  "candidates": [{"label": c["label"], "context": c["context"], "kind": c["kind"],
                                  "model": c["model"].name} for c in candidates],
                  "notes": notes, "results": results,
                  "winner": {"label": winner["label"], "model": winner["model"], "context": winner["context"],
                             "score": winner["score"], "tasks": tasks}}
    say(f"\n{BOLD}Summary{RESET}  {DIM}({len(results)} run(s) of the configurations tested){RESET}\n"
        + report_table(rank(results)))
    claim = f"{GREEN}{BOLD}{winner['label']}{RESET} is the best among the configurations tested"
    say(f"\n{claim}  {DIM}({tasks} tasks, agent {HARNESSES[agent]['title']}, {date}; "
        f"{len(candidates)} candidate(s)){RESET}")
    say(f"{DIM}This is not a universal optimum: hardware, runtime or model changes can change it.{RESET}")
    if len(results) > 1 and close_pair(ranked[0], ranked[1]) and not any(r.get("repeat") for r in results):
        say(f"{DIM}The top two were close, but there was no time left to repeat the best.{RESET}")
    name = f"tune-{date}"
    try:
        profile = winner_profile(config, name, winner_candidate, run_record)
        say(f"\nSaved the winner as profile {BOLD}{profile['name']}{RESET}. Apply it with:\n"
            f"  {CYAN}hearthwork profile apply {profile['name']}{RESET}")
    except (ValueError, OSError) as error:
        say(f"{YELLOW}Could not save the profile: {error}{RESET}")
    run_record["profile"] = name
    append_log(run_record)
    if as_json:
        print(json.dumps({"winner": run_record["winner"], "results": results,
                          "candidates": run_record["candidates"], "profile": name}, indent=2))
    return 0


# ---------- CLI ----------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="hearthwork tune",
                                     description="Try a few installed configurations and save the best tested one "
                                                 "as a profile (nothing is downloaded; the original model is restored).")
    parser.add_argument("--time", default="30m", help="time limit for the whole run, e.g. 30m, 90s or 1h (default 30m)")
    parser.add_argument("--min-context", type=ctx.parse_k, metavar="N",
                        help="minimum context a candidate must offer, e.g. 65536 or 64K")
    parser.add_argument("--agent", choices=sorted(HARNESSES), default="claude", help="agent to drive the tasks")
    parser.add_argument("--quick-tasks", type=int, default=DEFAULT_TASKS, metavar="N",
                        help=f"how many of the graded tasks to run per candidate (default {DEFAULT_TASKS})")
    parser.add_argument("--dry-run", action="store_true", help="list the candidates and their estimated needs, then exit")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    try:
        seconds_limit = parse_time(args.time)
    except ValueError:
        sys.exit("--time takes e.g. 30m, 90s or 1h")
    tasks = max(1, min(args.quick_tasks, len(bench.PROMPTS)))
    config = load_config()
    if config.get("remote"):
        sys.exit("Automatic tuning switches local models; this computer uses a remote host.")
    if args.dry_run:
        candidates, notes = build_candidates(config, args.min_context)
        if args.json:
            print(json.dumps({"tasks": tasks, "minContext": args.min_context,
                              "candidates": [{"label": c["label"], "model": c["model"].name, "context": c["context"],
                                              "kind": c["kind"], "need": c.get("need")} for c in candidates],
                              "skipped": notes}, indent=2))
        else:
            print(candidates_text(config, candidates, notes, tasks, args.min_context))
        return 0
    if not installed(args.agent):
        sys.exit(f"{HARNESSES[args.agent]['title']} is not installed.")
    return run(config, args.agent, tasks, seconds_limit, args.min_context, as_json=args.json)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")