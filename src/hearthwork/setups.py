"""Verified setup profiles: a working configuration captured with the evidence that it works.

  hearthwork profile save [NAME] [--doctor]   capture model, runtime, agents, server settings, hardware + Doctor/bench results
  hearthwork profile list | show NAME | check [NAME] [--run] | apply NAME | export NAME FILE | import FILE

Stored in HOME/profiles/<name>.json (schema version 1). "previous" is kept automatically before every apply.
A profile is "verified" only when Doctor passed for every recorded agent; a changed llama.cpp build, agent version,
chat template or model file turns it into "needs recheck" until `profile check --run` passes again.
"""
import argparse
import copy
import hashlib
import json
import platform
import re
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .paths import HOME

SCHEMA = 1
DIR = HOME / "profiles"
FINGERPRINT_BYTES = 16 * 1024 * 1024
DOCTOR_MAX_AGE = 24 * 3600
SERVER_KEYS = ("context", "kvCacheType", "slots", "batch", "fitTargetMiB")
EXTRA_KEYS = ("budget", "budgets", "preset", "budgetPreset")  # written by the resource-budget feature, kept opaque
SECRET_KEY = re.compile(r"key|token|secret|password|credential|bearer", re.I)
VERIFIED, RECHECK, UNVERIFIED = "verified", "needs recheck", "unverified"


# ---------- small helpers ----------

def profile_path(name):
    if not re.fullmatch(r"[\w.-]{1,64}", name or ""):
        raise ValueError("a profile name is letters, digits, '.', '_' and '-' (max 64)")
    return DIR / f"{name}.json"


def load(name):
    path = profile_path(name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store(profile):
    DIR.mkdir(parents=True, exist_ok=True)
    profile_path(profile["name"]).write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def names():
    return sorted(p.stem for p in DIR.glob("*.json")) if DIR.is_dir() else []


def fingerprint(path):
    """Cheap identity of a model file: sha256 of its first 16 MB plus its size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(str(Path(path).stat().st_size).encode())
    return h.hexdigest()[:32]


def sanitize(value, replace):
    """Drop secret-looking keys and replace local paths with placeholders, recursively."""
    if isinstance(value, dict):
        return {k: sanitize(v, replace) for k, v in value.items() if not SECRET_KEY.search(str(k)) and k != "folder"}
    if isinstance(value, list):
        return [sanitize(v, replace) for v in value]
    if isinstance(value, str):
        for real, placeholder in replace:
            for form in {real, real.replace("\\", "/"), real.replace("/", "\\")}:
                if form:
                    value = value.replace(form, placeholder)
        return re.sub(r"(?i)(bearer\s+|sk-)[\w.~+/=-]{8,}", r"\1********", value)
    return value


def replacements(config):
    items = [(str(HOME), "<HEARTHWORK_HOME>"), (str(Path.home()), "<HOME>")]
    if config.get("modelsDir"):
        items.append((str(config["modelsDir"]), "<MODELS_DIR>"))
    return sorted(items, key=lambda x: -len(x[0]))


# ---------- capture ----------

def current_model_path(config):
    from . import doctor
    from .server import served_model
    name = served_model(config.get("server", {}).get("port", 8001))
    path = doctor.model_path(config, name) if name else None
    if not path and config.get("lastModel") and Path(config["lastModel"]).is_file():
        path = Path(config["lastModel"])
    return path


def template_fingerprint(config):
    from . import doctor
    template = None
    for attempt in range(3):
        props = doctor.get_json(f"http://127.0.0.1:{config.get('server', {}).get('port', 8001)}/props", timeout=10)
        template = (props or {}).get("chat_template")
        if isinstance(template, str) and template:
            break
        time.sleep(1 if attempt < 2 else 0)
    return hashlib.sha256(template.encode()).hexdigest()[:12] if isinstance(template, str) and template else None


def model_identity(config, path):
    from . import doctor
    from .context import model_info
    if not path:
        return None
    info = model_info(path) or {}
    return {"file": path.name, "sizeBytes": path.stat().st_size, "fingerprint": fingerprint(path),
            "quantization": doctor.quantization(path, path.stem), "architecture": info.get("arch"),
            "chatTemplate": template_fingerprint(config)}


def agent_versions():
    from .harnesses import HARNESSES, agent_version, installed
    return {k: agent_version(k) for k in HARNESSES if installed(k)}


def hardware_summary():
    from .onboard import detect
    hw = detect()
    out = {k: hw.get(k) for k in ("os", "arch", "backend", "gpus", "vramGB", "ramGB", "cpuThreads")}
    out["osVersion"] = platform.release()
    return out


def settings_of(config, model_file):
    s = config.get("server", {})
    out = {k: s.get(k) for k in SERVER_KEYS}
    override = (config.get("contexts") or {}).get(model_file)
    if override:
        out["modelContext"] = override
    extra = {k: config[k] for k in EXTRA_KEYS if k in config}
    if extra:
        out["extra"] = extra
    return out


def runtime_of(config):
    from . import runtime
    chosen = runtime.selected(config)
    return {"name": chosen[0], "llamaCppBuild": chosen[1]} if chosen else {"name": None, "llamaCppBuild": None}


def latest_doctor(max_age=DOCTOR_MAX_AGE, now=None):
    folder = HOME / "doctor"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    for path in reversed(files):
        if (now or time.time()) - path.stat().st_mtime > max_age:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
    return None


def doctor_evidence(report):
    if not report:
        return None
    stages = [s.get("name", "") for a in (report.get("agents") or {}).values() for s in a.get("stages", [])]
    agents = {k: {"verdict": a.get("verdict"), "version": a.get("version")} for k, a in (report.get("agents") or {}).items()}
    return {"date": report.get("time"),
            "suite": f"hearthwork-{(report.get('environment') or {}).get('hearthwork', __version__)}/doctor",
            "conditions": {"mode": report.get("mode"), "quick": not any("coding" in n for n in stages)},
            "agents": agents}


def bench_evidence(model_name):
    from .bench import RESULTS
    best = {}
    if not model_name or not RESULTS.is_file():
        return None
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("model") == model_name and "score" in r:
            best[r["agent"]] = {"date": r.get("date"), "score": r["score"], "seconds": r.get("seconds"),
                                "attempt": r.get("attempt")}
    return {"suite": f"hearthwork-{__version__}/bench", "conditions": {"tasks": 8}, "agents": best} if best else None


def is_verified(evidence):
    agents = (evidence or {}).get("agents") or {}
    return bool(agents) and all(a.get("verdict") == "works" for a in agents.values())


def template_note(profile, now):
    a, b = (profile.get("model") or {}).get("chatTemplate"), (now.get("model") or {}).get("chatTemplate")
    return None if a and b else "template not compared (unavailable)"


def default_doctor(config, agents):
    from . import doctor
    report = doctor.run_doctor(config, agents, quick=True)
    doctor.save(report)
    return report


def capture(config, name, evidence=True, force_doctor=False, run_doctor=None):
    """A profile dict of the current setup. `run_doctor(config, agents)` -> report is injectable for tests."""
    path = current_model_path(config)
    model = model_identity(config, path)
    profile = {"schema": SCHEMA, "name": name, "created": datetime.now().isoformat(timespec="seconds"),
               "hearthwork": __version__, "model": model, "runtime": runtime_of(config), "agents": agent_versions(),
               "settings": settings_of(config, model["file"] if model else ""), "hardware": hardware_summary(),
               "evidence": {"doctor": None, "bench": None}, "status": UNVERIFIED}
    if evidence:
        report = None if force_doctor else latest_doctor()
        if report is None and profile["agents"]:
            report = (run_doctor or default_doctor)(config, list(profile["agents"]))
        profile["evidence"]["doctor"] = doctor_evidence(report)
        profile["evidence"]["bench"] = bench_evidence(path.stem if path else None)
        profile["status"] = VERIFIED if is_verified(profile["evidence"]["doctor"]) else UNVERIFIED
    return profile


# ---------- comparing ----------

def differences(profile, now):
    """Human-readable changes between a recorded profile and a freshly captured `now` (same shape)."""
    out = []

    def diff(label, a, b):
        if a is not None and a != b:
            out.append(f"{label}: recorded {a}, now {b if b is not None else 'missing'}")
    pm, nm = profile.get("model") or {}, now.get("model") or {}
    if not nm:
        out.append(f"model: recorded {pm.get('file')}, none found")
    else:
        diff("model file", pm.get("file"), nm.get("file"))
        diff("model size", pm.get("sizeBytes"), nm.get("sizeBytes"))
        if pm.get("file") == nm.get("file"):
            diff("model fingerprint", pm.get("fingerprint"), nm.get("fingerprint"))
        diff("quantization", pm.get("quantization"), nm.get("quantization"))
        if nm.get("chatTemplate"):
            diff("chat template", pm.get("chatTemplate"), nm.get("chatTemplate"))
    pr, nr = profile.get("runtime") or {}, now.get("runtime") or {}
    diff("runtime", pr.get("name"), nr.get("name"))
    diff("llama.cpp build", pr.get("llamaCppBuild"), nr.get("llamaCppBuild"))
    for key, version in (profile.get("agents") or {}).items():
        diff(f"agent {key} version", version, (now.get("agents") or {}).get(key))
    return out


def hardware_differences(recorded, current):
    out = []
    for key, label in (("vramGB", "VRAM"), ("ramGB", "RAM")):
        a, b = recorded.get(key), current.get(key)
        if a and b is not None and b < a * 0.95:
            out.append(f"hardware smaller than recorded: {label} {b} GB < {a} GB")
    if recorded.get("backend") and current.get("backend") != recorded["backend"]:
        out.append(f"GPU backend differs: recorded {recorded['backend']}, now {current.get('backend')}")
    if recorded.get("os") and current.get("os") != recorded["os"]:
        out.append(f"OS differs: recorded {recorded['os']}, now {current.get('os')}")
    return out


def quick_changes(config, profile):
    """Cheap change detection for `status` (no subprocesses): llama.cpp build and model file size."""
    out = []
    rec = (profile.get("runtime") or {}).get("llamaCppBuild")
    if rec is not None and runtime_of(config)["llamaCppBuild"] != rec:
        out.append("llama.cpp build")
    model = profile.get("model")
    if model:
        found, problems = find_model(config, model, deep=False)
        if problems:
            out.append("model file")
    return out


def effective_status(config, profile):
    if profile.get("status") == VERIFIED and quick_changes(config, profile):
        return RECHECK
    return profile.get("status", UNVERIFIED)


def status_line(config):
    name = config.get("activeProfile")
    try:
        profile = load(name) if name else None
    except (ValueError, OSError):
        return None
    return f"profile: {name} ({effective_status(config, profile)})" if profile else None


# ---------- resolve local artifacts ----------

def find_model(config, model, deep=True):
    """(path, problems): the local file matching name+size(+fingerprint); problems say what is missing or different."""
    folder = config.get("modelsDir")
    if not folder or not Path(folder).is_dir():
        return None, ["models folder is not configured"]
    candidates = list(Path(folder).rglob(model["file"]))
    if not candidates:
        return None, [f"model not found: {model['file']}"]
    for path in candidates:
        if path.stat().st_size == model.get("sizeBytes") and (not deep or not model.get("fingerprint")
                                                              or fingerprint(path) == model["fingerprint"]):
            return path, []
    return candidates[0], [f"model file {model['file']} differs from the recorded one (size or content)"]


# ---------- validate / export / import ----------

def validate(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"not a Hearthwork profile of schema {SCHEMA}")
    for key in ("name", "model", "runtime", "settings", "hardware"):
        if key not in data:
            raise ValueError(f"profile is missing '{key}'")
    if not isinstance(data["settings"], dict) or not isinstance(data["runtime"], dict) \
            or not (data["model"] is None or isinstance(data["model"].get("file"), str)):
        raise ValueError("profile has malformed settings or model")
    profile_path(data["name"])
    return data


def export_profile(config, profile):
    data = sanitize(profile, replacements(config))
    data.pop("imported", None)
    data.pop("autoSaved", None)
    data.pop("_restore", None)
    return data


def import_profile(config, data, name=None, hw=None, agents=None):
    """(profile, [differences]). The profile is stored 'unverified': this machine has not checked it."""
    validate(data)
    profile = dict(data)
    if name:
        profile["name"] = name
        profile_path(name)
    problems = []
    if profile.get("model"):
        problems += find_model(config, profile["model"])[1]
    from . import runtime
    rec = profile.get("runtime") or {}
    if rec.get("llamaCppBuild") is not None and rec["llamaCppBuild"] not in runtime.state(config)["installed"].get(rec.get("name"), []):
        now = runtime_of(config)
        problems.append(f"different llama.cpp build: recorded {rec.get('name')} b{rec['llamaCppBuild']}, "
                        f"installed {now['name']} b{now['llamaCppBuild']}")
    if profile.get("agents"):
        present = agent_versions() if agents is None else agents
        for key, version in profile["agents"].items():
            if key not in present:
                problems.append(f"agent {key} is not installed (recorded {version})")
            elif present[key] != version:
                problems.append(f"agent {key} version differs: recorded {version}, installed {present[key]}")
    problems += hardware_differences(profile.get("hardware") or {}, hw if hw is not None else hardware_summary())
    profile["recordedStatus"] = data.get("status")
    profile["status"] = UNVERIFIED
    profile["imported"] = True
    return profile, problems


# ---------- check ----------

def check(config, profile, run=False, run_doctor=None, now=None):
    """Revalidate; returns (status, differences). With run=True, a passing Doctor restores 'verified' and refreshes versions."""
    now = now or capture(config, profile["name"], evidence=False)
    changes = differences(profile, now)
    status = profile.get("status", UNVERIFIED)
    if status == VERIFIED and changes:
        status = RECHECK
    if run:
        report = (run_doctor or default_doctor)(config, list(now["agents"]))
        evidence = doctor_evidence(report)
        profile["evidence"]["doctor"] = evidence
        if is_verified(evidence):
            for key in ("model", "runtime", "agents"):
                profile[key] = now[key]
            status, changes = VERIFIED, []
        elif status == VERIFIED:
            status = RECHECK
    profile["status"] = status
    profile["checked"] = datetime.now().isoformat(timespec="seconds")
    return status, changes


# ---------- apply ----------

def snapshot(config):
    """The managed parts of the config, enough to put them back."""
    from . import runtime
    st = runtime.state(copy.deepcopy(config))
    return {"server": {k: (config.get("server") or {}).get(k) for k in SERVER_KEYS}, "contexts": copy.deepcopy(config.get("contexts") or {}),
            "lastModel": config.get("lastModel"), "selected": st.get("selected"), "active": dict(st["active"]),
            "extra": {k: copy.deepcopy(config[k]) for k in EXTRA_KEYS if k in config}}


def restore_snapshot(config, snap):
    from . import runtime
    config.setdefault("server", {}).update({k: v for k, v in snap["server"].items() if v is not None})
    config["contexts"] = snap["contexts"]
    if snap.get("lastModel"):
        config["lastModel"] = snap["lastModel"]
    for key in EXTRA_KEYS:
        config.pop(key, None)
    config.update(snap["extra"])
    st = runtime.state(config)
    st["active"].update(snap["active"])
    if snap.get("selected"):
        st["selected"] = snap["selected"]
    runtime.point_server(config)


def apply_settings(config, profile, model_path):
    """Write the managed parts into config (model, context/slots/batch/KV, runtime build). Returns (notes, context)."""
    from . import context as ctx, runtime
    notes = []
    s = profile["settings"]
    for key in SERVER_KEYS:
        if s.get(key) is not None:
            config.setdefault("server", {})[key] = s[key]
    if model_path:
        contexts = config.setdefault("contexts", {})
        if s.get("modelContext"):
            contexts[model_path.name] = s["modelContext"]
        else:
            contexts.pop(model_path.name, None)
        config["lastModel"] = str(model_path)
    for key, value in (s.get("extra") or {}).items():
        config[key] = value
    rec = profile.get("runtime") or {}
    if rec.get("name") and rec.get("llamaCppBuild") is not None:
        name, build = rec["name"], rec["llamaCppBuild"]
        st = runtime.state(config)
        if st.get("selected") == name and st["active"].get(name) == build:
            pass
        elif runtime.server_path(config, name, build):
            st["active"][name] = build
            st["selected"] = name
            runtime.point_server(config)
            notes.append(f"runtime switched to {name} b{build}")
        else:
            notes.append(f"runtime {name} b{build} is not installed; keeping the current one "
                         f"(`hearthwork runtime install {name}` installs the latest)")
    return notes, (ctx.choose(config, model_path)[0] if model_path else None)


def running_state(config):
    """{model, context, slots} of the running host, or None."""
    from . import host
    got = host.status()
    if not got or not got.get("ready"):
        return None
    return {"model": got.get("model"), "context": got.get("context"), "slots": got.get("slots")}


def settings_changes(old, new):
    """Names of managed settings that differ between two snapshots."""
    out = [k for k in SERVER_KEYS if old["server"].get(k) != new["server"].get(k)]
    if old["lastModel"] != new["lastModel"]:
        out.append("model file")
    if old["contexts"] != new["contexts"]:
        out.append("model context")
    if old["extra"] != new["extra"]:
        out.append("budget")
    if (old["selected"], old["active"]) != (new["selected"], new["active"]):
        out.append("runtime build")
    return out


def apply(config, profile, save_config, ensure, validator=None, restore=True, say=print, versions=None, running=None):
    """Apply `profile`. The previous settings are stored as profile 'previous' first; if applying or validating fails
    they are restored. All side effects (save_config, ensure(config, model, context), validator(config)) are injected."""
    model_path, problems = find_model(config, profile["model"]) if profile.get("model") else (None, [])
    if profile.get("model") and (not model_path or problems):
        say("Cannot apply: " + "; ".join(problems))
        return False
    before = copy.deepcopy(config)
    if profile["name"] != "previous":
        previous = capture(config, "previous", evidence=False)
        previous["autoSaved"] = f"before applying {profile['name']}"
        previous["_restore"] = snapshot(before)
        store(previous)
    notes, context = apply_settings(config, profile, model_path)
    for note in notes:
        say("  " + note)
    save_config(config)
    present = agent_versions() if versions is None else versions
    for key, version in (profile.get("agents") or {}).items():
        if present.get(key) != version:
            say(f"  note: agent {key} is {present.get(key) or 'not installed'}, profile recorded {version} "
                "(externally managed, not changed)")
    changed = settings_changes(snapshot(before), snapshot(config))
    live = (running or running_state)(config) if model_path else None
    if not changed and live and live.get("model") == model_path.stem and live.get("context") == context             and live.get("slots") in (None, config.get("server", {}).get("slots")):
        say("already active: the running model already has these settings")
        config["activeProfile"] = profile["name"]
        save_config(config)
        return True
    if changed:
        say("  changed: " + ", ".join(changed))
    ok = ensure(config, model_path, context) if model_path else True
    if ok and validator:
        ok = validator(config)
    if ok:
        config["activeProfile"] = profile["name"]
        save_config(config)
        return True
    say("Validation failed after applying." + (" Restoring the previous settings." if restore else ""))
    if restore:
        config.clear()
        config.update(before)
        save_config(config)
        old = Path(before["lastModel"]) if before.get("lastModel") else None
        if old and old.is_file():
            ensure(config, old, None)
    return False


def apply_previous(config, save_config, ensure, say=print):
    """`profile apply previous`: put back the configuration saved before the last apply."""
    previous = load("previous")
    saved = (previous or {}).get("_restore")
    if not saved:
        say("No previous configuration saved.")
        return False
    current = snapshot(config)
    restore_snapshot(config, saved)
    config.pop("activeProfile", None)
    save_config(config)
    model = Path(saved["lastModel"]) if saved.get("lastModel") else None
    ok = ensure(config, model, None) if model and model.is_file() else True
    previous["_restore"] = current  # applying "previous" again swaps back
    store(previous)
    return ok


# ---------- command line ----------

def show_text(profile, config=None):
    m = profile.get("model") or {}
    r = profile.get("runtime") or {}
    s = profile["settings"]
    hw = profile.get("hardware") or {}
    status = effective_status(config, profile) if config is not None else profile.get("status")
    lines = [f"profile {profile['name']}  [{status}]  created {profile.get('created')}  (schema {profile['schema']})",
             f"  model: {m.get('file')}  {m.get('quantization') or '?'}  {m.get('architecture') or '?'}  "
             f"{(m.get('sizeBytes') or 0) / 1e9:.1f} GB  template {m.get('chatTemplate') or '?'}",
             f"  runtime: {r.get('name')} b{r.get('llamaCppBuild')}",
             "  agents: " + (", ".join(f"{k} {v}" for k, v in (profile.get("agents") or {}).items()) or "none"),
             "  settings: " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "extra") + (f"  extra={s['extra']}" if s.get("extra") else ""),
             f"  hardware: {', '.join(hw.get('gpus') or ['no GPU'])}  VRAM {hw.get('vramGB')} GB  RAM {hw.get('ramGB')} GB  {hw.get('os')}"]
    d = (profile.get("evidence") or {}).get("doctor")
    if d:
        lines.append(f"  doctor: {d['date']}  suite {d['suite']}  {'quick' if d['conditions'].get('quick') else 'full'}  "
                     + ", ".join(f"{k}={v['verdict']}" for k, v in d["agents"].items()))
    b = (profile.get("evidence") or {}).get("bench")
    if b:
        lines.append(f"  bench: suite {b['suite']}  " + ", ".join(f"{k} {v['score']}/8 ({v['date']})" for k, v in b["agents"].items()))
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(prog="hearthwork profile", description="Verified setup profiles.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("save")
    p.add_argument("name", nargs="?", default="default")
    p.add_argument("--doctor", action="store_true", help="run Doctor now instead of using a report from the last 24 hours")
    sub.add_parser("list")
    sub.add_parser("show").add_argument("name")
    p = sub.add_parser("check")
    p.add_argument("name", nargs="?")
    p.add_argument("--run", action="store_true", help="re-run Doctor; a pass makes the profile verified again")
    p = sub.add_parser("apply")
    p.add_argument("name")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--keep-failed", action="store_true", help="do not restore the previous settings when validation fails")
    p = sub.add_parser("export")
    p.add_argument("name")
    p.add_argument("file")
    p = sub.add_parser("import")
    p.add_argument("file")
    p.add_argument("--name")
    args = parser.parse_args(argv)
    from .onboard import load_config, save_config
    from .procs import setup_stdio
    setup_stdio()
    config = load_config()
    try:
        if args.cmd == "list":
            for n in names():
                pr = load(n)
                print(f"{n}{' *' if config.get('activeProfile') == n else ''}  {effective_status(config, pr)}  "
                      f"{(pr.get('model') or {}).get('file')}  {pr.get('created')}")
            return 0
        if args.cmd == "save":
            if config.get("remote"):
                print("Profiles describe a local setup; this computer uses a remote host.")
                return 1
            profile = capture(config, args.name, force_doctor=args.doctor)
            if not profile["model"]:
                print("No model found: start one first (`hearthwork start`).")
                return 1
            store(profile)
            config["activeProfile"] = args.name
            save_config(config)
            print(show_text(profile))
            return 0
        if args.cmd == "import":
            data = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
            profile, problems = import_profile(config, data, args.name)
            store(profile)
            print(f"Imported profile '{profile['name']}' as unverified (recorded: {profile.get('recordedStatus')}).")
            print("Differences on this machine:" if problems else "Everything the profile records is present here.")
            for line in problems:
                print("  - " + line)
            if problems:
                print("`hearthwork profile check --run` re-verifies it here.")
            return 0
        name = args.name if getattr(args, "name", None) else config.get("activeProfile")
        if not name:
            print("No profile named and none active. `hearthwork profile list` shows them.")
            return 1
        if args.cmd == "apply" and name == "previous":
            from . import host
            ok = apply_previous(config, save_config, lambda c, m, x: host.ensure(c, m, x))
            print("Restored the previous configuration." if ok else "Could not restore.")
            return 0 if ok else 1
        profile = load(name)
        if not profile:
            print(f"No profile '{name}'. `hearthwork profile list` shows them.")
            return 1
        if args.cmd == "show":
            print(show_text(profile, config))
        elif args.cmd == "export":
            Path(args.file).write_text(json.dumps(export_profile(config, profile), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Exported '{name}' to {args.file} (no credentials, no local paths).")
        elif args.cmd == "check":
            status, changes = check(config, profile, args.run)
            store(profile)
            print(f"profile {name}: {status}")
            for line in changes:
                print("  - " + line)
            note = template_note(profile, capture(config, name, evidence=False))
            if note:
                print("  note: " + note)
            if status == RECHECK:
                print("Run `hearthwork profile check --run` to re-run Doctor.")
            return 0 if status == VERIFIED else 1
        elif args.cmd == "apply":
            from . import host
            agents = list(profile.get("agents") or {})
            validator = None if args.no_validate or not agents else (lambda c: bool(default_doctor(c, agents).get("ok")))
            ok = apply(config, profile, save_config, lambda c, m, x: host.ensure(c, m, x), validator, restore=not args.keep_failed)
            print("Applied." if ok else "Not applied.")
            return 0 if ok else 1
    except (ValueError, OSError) as error:
        print(f"profile: {error}")
        return 1
    return 0
