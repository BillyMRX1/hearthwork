"""Session diagnostics: what limited the agent sessions, from measurements only.

Recording is passive. The relay (harnesses.make_relay) hands each generation request a `Probe`; the probe sees the response
bytes as they pass (it keeps the first 4 KB and the last 16 KB, nothing is held back or delayed) and, when the request
ends, reads llama-server's `timings` and `usage` from them. One record per request goes into a ring of the last 500 and
into HOME/logs/requests.jsonl (rotated at ~5 MB). Records hold numbers only: no prompt, no response. A value the backend
did not report is None, shown as "unavailable", never guessed. HEARTHWORK_CAPTURE=1 is the opt-in exception: it also
writes request bodies and responses to HOME/logs/capture.jsonl (local file, never exported, may hold your code).

`hearthwork diagnose` reads requests.jsonl, so it works with the host stopped.
"""
import collections
import json
import math
import os
import re
import sys
import threading
import time

from .paths import LOGS

RING = 500
SOURCES = ("server", "relay-stream", "relay-total")
SOURCE_LABEL = {"server": "", "relay-stream": " (relay-measured)", "relay-total": " (relay-measured, includes prompt processing)"}
PROMPT_SOURCES = ("server", "relay-estimated")
PROMPT_LABEL = {"server": "", "relay-estimated": " (relay-estimated, includes overhead)"}
PROMPT_ESTIMATE_MIN_TTFB_MS = 200   # below this, first-byte time is too small to divide by
DEFAULT_SINCE = 24 * 3600
ROTATE_AT = 5_000_000
HEAD, TAIL = 4096, 16384
CAPTURE_LIMIT = 2_000_000
PRESSURE = 0.8          # a request above this share of the context is "under pressure"
REQUESTS_FILE = LOGS / "requests.jsonl"
CAPTURE_FILE = LOGS / "capture.jsonl"


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _scan(text, key):
    """Every JSON object that follows `"key":` in text, in order (partial ones at the cut ends are skipped)."""
    found, decoder = [], json.JSONDecoder()
    for match in re.finditer(r'"%s"\s*:\s*' % key, text):
        try:
            value, _ = decoder.raw_decode(text, match.end())
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def extract(text):
    """(prompt tokens, cached tokens, generated tokens, prompt tokens/s, generation tokens/s) from response text; None = unavailable.
    Handles llama.cpp's `timings` + OpenAI usage (chat), Responses usage, and Anthropic usage (input + cache_read)."""
    usage, timings = {}, {}
    for item in _scan(text, "usage"):
        usage.update(item)
    for item in _scan(text, "timings"):
        timings.update(item)
    prompt = cached = gen = None
    if "prompt_tokens" in usage:
        prompt, gen = _num(usage.get("prompt_tokens")), _num(usage.get("completion_tokens"))
        cached = _num((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    elif "input_tokens" in usage:
        prompt, gen = _num(usage.get("input_tokens")), _num(usage.get("output_tokens"))
        if "cache_read_input_tokens" in usage:  # Anthropic: input_tokens counts only the part that was not cached
            cached = _num(usage["cache_read_input_tokens"])
            prompt = prompt + cached if prompt is not None and cached is not None else None
        else:
            cached = _num((usage.get("input_tokens_details") or {}).get("cached_tokens"))
    if timings:
        if _num(timings.get("cache_n")) is not None:
            cached = timings["cache_n"]
        if prompt is None and _num(timings.get("prompt_n")) is not None:
            prompt = timings["prompt_n"] + (cached or 0)
        if gen is None:
            gen = _num(timings.get("predicted_n"))
    prompt_tps = _num(timings.get("prompt_per_second"))
    gen_tps = _num(timings.get("predicted_per_second"))
    return prompt, cached, gen, prompt_tps, gen_tps


def requested_output(body):
    """The request's max output tokens (any API's spelling), or None."""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, dict):
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens", "n_predict"):
            value = _num(payload.get(key))
            if value is not None and value > 0:
                return int(value)
    return None


def capacity_message(prompt_tokens, n_ctx, slots):
    share = f" (per session; {slots} sessions share the pool)" if slots and slots > 1 else " (per session)"
    head = f"{prompt_tokens:,} tokens don't fit" if prompt_tokens else "The request doesn't fit"
    return (f"{head} the model's {n_ctx:,} context{share}: raise it with `hearthwork context <model> <N>` "
            "or start a new conversation.") if n_ctx else (
        f"{head} the model's context: raise it with `hearthwork context <model> <N>` or start a new conversation.")


def rewrite_capacity(data, slots=None):
    """(new body, prompt tokens, n_ctx) when `data` is llama-server's context-exceeded error, else None. The error keeps
    its HTTP status and shape (OpenAI or Anthropic); only the message is replaced."""
    try:
        payload = json.loads(data)
        error = payload["error"]
        message = str(error.get("message", ""))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not (error.get("type") == "exceed_context_size_error" or "exceeds the available context size" in message):
        return None
    prompt, n_ctx = _num(error.get("n_prompt_tokens")), _num(error.get("n_ctx"))
    numbers = re.search(r"\((\d+) tokens\) exceeds .*?\((\d+) tokens\)", message)
    if numbers:
        prompt = prompt if prompt is not None else int(numbers.group(1))
        n_ctx = n_ctx if n_ctx is not None else int(numbers.group(2))
    error["message"] = capacity_message(prompt, n_ctx, slots)
    return json.dumps(payload).encode(), prompt, n_ctx


class Recorder:
    def __init__(self, path=None, size=RING, rotate_at=ROTATE_AT):
        self.path = path or REQUESTS_FILE
        self.ring = collections.deque(maxlen=size)
        self.rotate_at = rotate_at
        self.lock = threading.Lock()
        self.open = set()                            # probes whose request has not ended yet
        self.context = lambda: (None, None, None)   # the host sets: () -> (n_ctx, slots, model)

    def track(self, probe):
        with self.lock:
            self.open.add(probe)

    def untrack(self, probe):
        with self.lock:
            self.open.discard(probe)

    def running(self):
        """Requests that hold a slot right now (admitted, not yet finished)."""
        with self.lock:
            return sum(1 for p in self.open if p.started is not None)

    def add(self, record):
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self.lock:
            self.ring.append(record)
            try:
                if self.path.exists() and self.path.stat().st_size > self.rotate_at:
                    os.replace(self.path, str(self.path) + ".1")
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass  # diagnostics never get in the way of a request

    def recent(self, seconds=None):
        with self.lock:
            records = list(self.ring)
        if seconds is not None:
            records = [r for r in records if r["ts"] >= time.time() - seconds]
        return records


RECORDER = Recorder()


class NoProbe:
    def admitted(self, ticket): pass
    def response(self, status): pass
    def feed(self, chunk): pass
    def finish(self, outcome): pass
    def rewrite_error(self, data): return data


NOOP = NoProbe()


def begin(client, path, body, gate=None, recorder=None):
    return Probe(recorder or RECORDER, client, path, body, gate)


class Probe:
    def __init__(self, recorder, client, path, body, gate=None):
        self.recorder, self.done = recorder, False
        n_ctx, slots, model = recorder.context()
        self.record = {"ts": round(time.time(), 3), "client": client, "api": path, "model": os.path.basename(model.replace("\\", "/")) if model else None,
                       "outcome": None, "status": None, "wait_ms": None, "ttfb_ms": None, "total_ms": None, "prompt_tokens": None,
                       "cached_tokens": None, "gen_tokens": None, "prompt_tps": None, "gen_tps": None, "gen_tps_source": None, "n_ctx": n_ctx, "slots": slots,
                       "max_output": requested_output(body) if body else None, "active": None,
                       "queued_at_arrival": len(gate.queue) if gate else None, "capacity_error": False}
        self.arrived = time.monotonic()
        self.started = None
        self.head, self.tail = bytearray(), bytearray()
        self.seen = 0
        self.first_at = None
        self.streams = bool(re.search(rb'"stream"\s*:\s*true', body or b""))
        self.capture = os.environ.get("HEARTHWORK_CAPTURE") == "1"
        self.request_body = bytes(body or b"") if self.capture else None
        self.everything = bytearray() if self.capture else None
        recorder.track(self)

    def admitted(self, ticket):
        entry = ticket.entry
        self.started = entry["started"]
        self.record["wait_ms"] = round((entry["started"] - entry["queued"]) * 1000)
        self.record["active"] = len(ticket.gate.active)
        self.record["slots"] = ticket.gate.limit()

    def response(self, status):
        self.record["status"] = status

    def feed(self, chunk):
        if not chunk:
            return
        if self.first_at is None:
            self.first_at = time.monotonic()
        if self.record["ttfb_ms"] is None:
            self.record["ttfb_ms"] = round((time.monotonic() - (self.started or self.arrived)) * 1000)
        if self.seen < HEAD:
            self.head += chunk[:HEAD - self.seen]
        self.seen += len(chunk)
        self.tail += chunk
        if len(self.tail) > 2 * TAIL:
            del self.tail[:-TAIL]
        if self.everything is not None and len(self.everything) < CAPTURE_LIMIT:
            self.everything += chunk

    def rewrite_error(self, data):
        got = rewrite_capacity(data, self.record["slots"])
        if got is None:
            return data
        data, prompt, n_ctx = got
        self.record.update(capacity_error=True, prompt_tokens=prompt, n_ctx=n_ctx or self.record["n_ctx"])
        return data

    def finish(self, outcome):
        if self.done:
            return
        self.done = True
        self.recorder.untrack(self)
        record = self.record
        status = record["status"]
        if outcome == "ok" and status is not None and status >= 400:
            outcome = "error"
        record["outcome"] = outcome
        if self.started is not None:
            record["total_ms"] = round((time.monotonic() - self.started) * 1000)
        if self.seen and not record["capacity_error"] and (status is None or status < 400):
            tail = bytes(self.tail).decode("utf-8", "replace")
            text = tail if self.seen <= len(self.tail) else (bytes(self.head).decode("utf-8", "replace") + " "
                                                            + bytes(self.tail[-TAIL:]).decode("utf-8", "replace"))
            prompt, cached, gen, prompt_tps, gen_tps = extract(text)
            record.update(prompt_tokens=prompt, cached_tokens=cached, gen_tokens=gen, prompt_tps=prompt_tps, gen_tps=gen_tps)
            record["gen_tps_source"] = "server" if gen_tps is not None else None
            record["prompt_tps_source"] = "server" if prompt_tps is not None else None
            ttfb = record["ttfb_ms"]  # measured from admission, so queue wait is already excluded
            if prompt_tps is None and prompt and ttfb is not None and ttfb >= PROMPT_ESTIMATE_MIN_TTFB_MS and prompt - (cached or 0) > 0:
                # no server timings: uncached prompt tokens over the time to the first byte (includes overhead)
                record.update(prompt_tps=(prompt - (cached or 0)) / (ttfb / 1000), prompt_tps_source="relay-estimated")
            if gen_tps is None and gen and status == 200:  # no server timings: the relay's own clock and the usage token counts
                if self.streams and self.first_at is not None and time.monotonic() - self.first_at > 0.05:
                    record.update(gen_tps=gen / (time.monotonic() - self.first_at), gen_tps_source="relay-stream")
                elif not self.streams and record["total_ms"]:
                    record.update(gen_tps=gen / (record["total_ms"] / 1000), gen_tps_source="relay-total")
        self.recorder.add(record)
        if self.capture:
            self._write_capture()

    def _write_capture(self):
        try:
            entry = {"ts": self.record["ts"], "client": self.record["client"], "api": self.record["api"],
                     "request": (self.request_body or b"").decode("utf-8", "replace"),
                     "response": bytes(self.everything or b"").decode("utf-8", "replace")}
            with open(CAPTURE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass


# ---------- reading and summarizing ----------

def load_records(path=None):
    path = path or REQUESTS_FILE
    records = []
    for name in (str(path) + ".1", str(path)):
        try:
            with open(name, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict) and isinstance(item.get("ts"), (int, float)):
                        records.append(item)
        except OSError:
            pass
    return records


def parse_duration(text):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", text or "")
    if not match:
        raise ValueError(f"not a duration: {text!r} (examples: 90s, 30m, 2h, 1d)")
    return float(match.group(1)) * {"": 60, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]


def select(records, last=None, since=None, client=None, now=None):
    now = now or time.time()
    if since is not None:
        records = [r for r in records if r["ts"] >= now - since]
    if client:
        records = [r for r in records if client in str(r.get("client"))]
    records = sorted(records, key=lambda r: r["ts"])
    return records[-last:] if last else records


def percentile(values, pct):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * pct / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _vals(records, key):
    return [r[key] for r in records if _num(r.get(key)) is not None]


def summarize(records):
    """Numbers for a set of records. Every figure is None when no record carried the measurement."""
    ran = [r for r in records if r.get("total_ms") is not None]
    prompt_known = [r for r in records if _num(r.get("prompt_tokens")) and _num(r.get("cached_tokens")) is not None]
    prompt_sum = sum(r["prompt_tokens"] for r in prompt_known)
    cached_sum = sum(r["cached_tokens"] for r in prompt_known)
    pressure = []
    for r in records:
        if _num(r.get("prompt_tokens")) is not None and _num(r.get("n_ctx")):
            pressure.append((r["prompt_tokens"] + (r.get("max_output") or 0)) / r["n_ctx"])
    prompt_median = (None, None)
    for source in PROMPT_SOURCES:  # same rule: one source only
        values = [r["prompt_tps"] for r in records if (r.get("prompt_tps_source") or "server") == source and _num(r.get("prompt_tps")) is not None]
        if values:
            prompt_median = (percentile(values, 50), source)
            break
    gen_median = (None, None)
    for source in SOURCES:  # never mix sources: the best source that has values
        values = [r["gen_tps"] for r in records if r.get("gen_tps_source") == source and r.get("gen_tps") is not None]
        if values:
            gen_median = (percentile(values, 50), source)
            break
    waits, totals = sum(_vals(records, "wait_ms")), sum(_vals(ran, "total_ms"))
    return {
        "requests": len(records),
        "outcomes": dict(collections.Counter(r.get("outcome") or "unknown" for r in records)),
        "errors": sum(1 for r in records if r.get("outcome") == "error"),
        "overloaded": sum(1 for r in records if r.get("outcome") == "overloaded"),
        "capacity_errors": sum(1 for r in records if r.get("capacity_error")),
        "clients": sorted({str(r.get("client")) for r in records}),
        "queue_wait_ms": {"p50": percentile(_vals(records, "wait_ms"), 50), "p95": percentile(_vals(records, "wait_ms"), 95)},
        "ttfb_ms": {"p50": percentile(_vals(records, "ttfb_ms"), 50), "p95": percentile(_vals(records, "ttfb_ms"), 95)},
        "prompt_tps_median": prompt_median[0], "prompt_tps_source": prompt_median[1],
        "gen_tps_median": gen_median[0], "gen_tps_source": gen_median[1],
        "gen_tps_by_source": {k: percentile([r["gen_tps"] for r in records if r.get("gen_tps_source") == k and r.get("gen_tps") is not None], 50) for k in SOURCES},
        "cache": {"requests_measured": len(prompt_known), "prompt_tokens": prompt_sum, "cached_tokens": cached_sum,
                  "hit_ratio": cached_sum / prompt_sum if prompt_sum else None},
        "context": {"requests_measured": len(pressure), "share_above_80pct": sum(1 for p in pressure if p > PRESSURE) / len(pressure) if pressure else None,
                    "max_use": max(pressure) if pressure else None, "n_ctx": max(_vals(records, "n_ctx"), default=None)},
        "contention": {"max_concurrent": max(_vals(records, "active"), default=None),
                       "max_queued_at_arrival": max(_vals(records, "queued_at_arrival"), default=None),
                       "slots": max(_vals(records, "slots"), default=None),
                       "queue_time_share": waits / (waits + totals) if (waits + totals) and _vals(records, "wait_ms") else None},
        "unavailable": {"prompt_tokens": sum(1 for r in records if r.get("prompt_tokens") is None and not r.get("capacity_error") and r.get("outcome") == "ok"),
                        "speeds": sum(1 for r in records if r.get("outcome") == "ok" and r.get("gen_tps") is None),
                        "server_timings": sum(1 for r in records if r.get("outcome") == "ok" and r.get("gen_tps_source") != "server")},
    }


def findings(summary):
    """[{kind: 'observed'|'possible', text, suggestion}]. Observed: stated by the numbers. Possible: an explanation the
    numbers are consistent with but do not prove, always labelled so."""
    out = []
    s = summary

    def add(kind, text, suggestion):
        out.append({"kind": kind, "text": text, "suggestion": suggestion})

    c = s["contention"]
    share = c["queue_time_share"]
    if share is not None and share >= 0.15:
        add("observed", f"{share:.0%} of request time was spent waiting in the queue: {len(s['clients'])} client(s) shared {c['slots'] or '?'} slot(s).",
            "Run fewer agents at the same time, or let them take turns; each request holds a slot until its answer ends.")
    if s["overloaded"]:
        add("observed", f"{s['overloaded']} request(s) were refused as overloaded (queue full or waited too long).",
            "Run fewer agents at once; see `hearthwork status` for who is queued.")
    if s["capacity_errors"]:
        add("observed", f"{s['capacity_errors']} request(s) were rejected because the prompt did not fit the context.",
            "Raise the context with `hearthwork context <model> <N>` or start a new conversation; nothing was truncated.")
    other = s["errors"] - s["capacity_errors"]
    if other:
        add("observed", f"{other} request(s) ended in an error.", "`hearthwork logs` shows the host's side; `hearthwork logs --server` the model server's.")
    x = s["context"]
    if x["share_above_80pct"] and x["share_above_80pct"] >= 0.2:
        add("observed", f"{x['share_above_80pct']:.0%} of requests used more than 80% of the {x['n_ctx']:,} context (prompt plus requested output).",
            "Raise the context if memory allows (`hearthwork context <model> <N>`), or start fresh conversations more often.")
    ratio = s["cache"]["hit_ratio"]
    if ratio is not None and s["cache"]["requests_measured"] >= 3 and ratio < 0.5:
        add("observed", f"Cache hit ratio is {ratio:.0%} ({s['cache']['cached_tokens']:,} of {s['cache']['prompt_tokens']:,} prompt tokens were already cached).",
            "Prompt processing is repeated work; see the possible causes.")
        add("possible", "Possible cause of the low cache hit ratio: conversations restarting, or several agents evicting each other's cache.",
            "Keep one conversation per agent going instead of restarting it, and avoid sharing the model between more agents than it has slots.")
    t = s["ttfb_ms"]["p95"]
    if t is not None and t > 10000:
        add("observed", f"The slowest 5% of requests took {t / 1000:.1f} s to produce a first byte.",
            "Long prompts are the usual reason (prompt processing); compare with the prompt speed and the cache hit ratio.")
        if ratio is not None and ratio < 0.5:
            add("possible", "Possible cause of the slow first bytes: large prompts being processed again because the cache was not reused.",
                "Fix the cache hits first (above).")
    g = s["gen_tps_median"]
    if g is not None and g < 10 and s["gen_tps_source"] != "relay-total":
        add("observed", f"Median generation speed is {g:.1f} tokens/s.",
            "A smaller or more compressed model, or a shorter context, generates faster; `hearthwork check` shows what fits.")
        add("possible", "Possible cause of the slow generation: the model does not fully fit the GPU memory, or other programs use the GPU.",
            "Close other GPU programs, or choose a smaller model.")
    return out


def unavailable_notes(summary):
    notes = ["Not measured: model load time, memory use, and time spent in tools (the host only sees requests, not what agents do between them)."]
    u = summary["unavailable"]
    if u["speeds"]:
        notes.append(f"Speeds are unavailable for {u['speeds']} request(s): not even token counts were available.")
    if u["prompt_tokens"]:
        notes.append(f"Token counts are unavailable for {u['prompt_tokens']} request(s).")
    return notes


def _client_name(client, mapping):
    if client.startswith("share:"):
        return mapping.setdefault(client, f"share:device-{len(mapping) + 1}")
    return client


def sanitize(records):
    """Records for export: numbers and API names only; paths and device names removed."""
    mapping, out = {}, []
    for r in records:
        item = {k: v for k, v in r.items() if k not in ("prompt", "response", "request")}
        item["client"] = _client_name(str(r.get("client")), mapping)
        if item.get("model"):
            item["model"] = os.path.basename(str(item["model"]).replace("\\", "/"))
        out.append(item)
    return out


def report(records, selection=None):
    records = sanitize(records)
    summary = summarize(records)
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), "selection": selection or {}, "summary": summary,
            "observed": [f for f in findings(summary) if f["kind"] == "observed"],
            "possible_causes": [f for f in findings(summary) if f["kind"] == "possible"],
            "not_measured": unavailable_notes(summary),
            "requests": sanitize(records)}


def _ms(value):
    return "unavailable" if value is None else (f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.0f} ms")


def _rate(value):
    return "unavailable" if value is None else f"{value:.1f} tokens/s"


def _pct(value):
    return "unavailable" if value is None else f"{value:.0%}"


def format_report(rep):
    s = rep["summary"]
    head = f"Selection: {rep['selection'].get('label', 'all recorded requests')}"
    if not s["requests"]:
        return head + "\nNo requests recorded for this selection yet. Use an agent through the Hearthwork host, then run this again."
    o = ", ".join(f"{n} {k}" for k, n in sorted(s["outcomes"].items()))
    lines = [head, f"Requests: {s['requests']} ({o}) from {len(s['clients'])} client(s)",
             f"Queue wait:  p50 {_ms(s['queue_wait_ms']['p50'])}, p95 {_ms(s['queue_wait_ms']['p95'])}",
             f"First byte:  p50 {_ms(s['ttfb_ms']['p50'])}, p95 {_ms(s['ttfb_ms']['p95'])}",
             f"Speed (median): prompt {_rate(s['prompt_tps_median'])}{PROMPT_LABEL.get(s['prompt_tps_source'], '')}, generation {_rate(s['gen_tps_median'])}{SOURCE_LABEL.get(s['gen_tps_source'], '')}",
             f"Cache hit ratio: {_pct(s['cache']['hit_ratio'])}",
             f"Context pressure: {_pct(s['context']['share_above_80pct'])} of requests above 80% of capacity"
             + (f" (context {s['context']['n_ctx']:,})" if s["context"]["n_ctx"] else ""),
             f"Contention: max {s['contention']['max_concurrent'] if s['contention']['max_concurrent'] is not None else 'unavailable'} running, "
             f"max {s['contention']['max_queued_at_arrival'] if s['contention']['max_queued_at_arrival'] is not None else 'unavailable'} queued ahead"
             + (f", {s['contention']['slots']} slots" if s["contention"]["slots"] else "")]
    for title, key in (("Observed", "observed"), ("Possible causes (not proven)", "possible_causes")):
        lines += ["", title + ":"]
        if not rep[key]:
            lines.append("  nothing notable")
        for f in rep[key]:
            lines += [f"  - {f['text']}", f"    Suggestion: {f['suggestion']}"]
    lines += [""] + [f"Note: {n}" for n in rep["not_measured"]]
    return "\n".join(lines)


def status_line(summary):
    """One line for `hearthwork status`. summary is the host's diagnostics payload (summarize() plus "running")."""
    summary = summary or {}
    running = summary.get("running") or 0
    tail = f", {running} running now" if running else ""
    if not summary.get("requests"):
        return f"last 5 min: 0 finished{tail}" if running else "last 5 min: no requests"
    return (f"last 5 min: {summary['requests']} finished, median generation {_rate(summary['gen_tps_median'])}{SOURCE_LABEL.get(summary['gen_tps_source'], '')}, "
            f"queue wait p95 {_ms(summary['queue_wait_ms']['p95'])}{tail}")


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="hearthwork diagnose", description="Explain what limited your agent sessions, from measured requests "
                                     "(HOME/logs/requests.jsonl). Nothing leaves this computer; prompts and code are not recorded.")
    parser.add_argument("--last", type=int, metavar="N", help="only the last N requests")
    parser.add_argument("--since", metavar="DURATION", help="only requests from the last period, e.g. 90s, 30m, 2h, 1d")
    parser.add_argument("--all", action="store_true", help="every recorded request, not only the last 24 hours (the default)")
    parser.add_argument("--client", help="only this client (part of its id, e.g. opencode, api, share)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--export", metavar="FILE", help="write a sanitized JSON report (numbers only: no prompts, code, or credentials)")
    args = parser.parse_args(argv)
    if args.all and args.since:
        parser.error("use --all or --since, not both")
    try:
        since = parse_duration(args.since) if args.since else None
    except ValueError as error:
        parser.error(str(error))
    if args.all:
        label = "all recorded history"
    elif args.last:
        label = f"last {args.last} requests"
    elif args.since:
        label = f"last {args.since}"
    else:
        since = DEFAULT_SINCE
        label = "last 24 h; use --since or --last to change"
    records = select(load_records(), args.last, since, args.client)
    rep = report(records, {"label": label, "last": args.last, "since": args.since, "client": args.client})
    if args.export:
        target = os.path.abspath(args.export)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        except OSError as error:
            print(f"hearthwork diagnose: cannot write {args.export}: {error.strerror or error}", file=sys.stderr)
            return 1
        print(f"Sanitized report written to {target} (numbers only: no prompts, code, or credentials).")
    elif args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(format_report(rep))
    return 0
