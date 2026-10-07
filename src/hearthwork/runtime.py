"""Runtimes: several llama.cpp engines installed side by side, the recommended one used by default.

  hearthwork runtime                 installed / recommended / available for this PC
  hearthwork runtime install rocm    download next to the others (smoke-tested before it counts)
  hearthwork runtime use rocm        switch (applies at the next model start)
  hearthwork runtime update          newest build of the installed runtimes, kept only if it passes the smoke test
  hearthwork runtime rollback        back to the previous build of the selected runtime

Each build lives in bin/<runtime>-b<build>/. config.json -> "runtime" records what is installed:
  selected   the runtime the server starts from
  installed  {runtime: [builds, newest first]}: the current build and the one before it (for rollback) are kept
  active     {runtime: build in use}
  legacy     {"<runtime>-b<build>": path} for a build an older Hearthwork put directly in bin/ (adopted in place)
  checked, latest   when the newest release was last looked up, and its build (for the update notice)
"""
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .onboard import (CYAN, GREEN, RED, RESET, SERVER_NAME, WINDOWS, YELLOW, detect, download, extract, github_json,
                      load_config, save_config, server_version)
from .paths import BIN

DIM = "\033[2m"
RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
NOTICE_EVERY = 24 * 3600
NOTICE_TIMEOUT = 3

LABELS = {"cuda13": "CUDA 13 (NVIDIA)", "cuda12": "CUDA 12 (NVIDIA, older drivers)", "rocm": "ROCm / HIP (AMD)",
          "vulkan": "Vulkan (any GPU)", "sycl": "SYCL (Intel Arc)", "metal": "Metal (Apple)", "cpu": "CPU only"}

# Official llama.cpp release assets, checked against the b11462 release (2026-10).
AVAILABLE = {
    ("windows", "x64"): ["cuda13", "cuda12", "rocm", "vulkan", "sycl", "cpu"],
    ("windows", "arm64"): ["cuda13", "vulkan", "cpu"],
    ("linux", "x64"): ["cuda13", "cuda12", "rocm", "vulkan", "sycl", "cpu"],
    ("linux", "arm64"): ["cuda13", "vulkan", "cpu"],
    ("macos", "arm64"): ["metal"],
    ("macos", "x64"): ["metal"],
}

# AMD GPUs the ROCm (HIP) builds run on, by name: RDNA2 and newer (RX 6500+, 7000, 9000; Radeon PRO W6/W7/W9;
# Radeon 7xxM/8xxM iGPUs and Strix Halo 8050S/8060S; Instinct). Older cards (RX 400/500, Vega, RX 5000) need Vulkan.
ROCM_GPU = re.compile(r"\bRX\s*(6[5-9]\d\d|7\d\d\d|9\d\d\d)|Radeon(\(TM\))?\s+(PRO\s+)?W[679]\d\d\d|Radeon\s+AI\s+PRO"
                      r"|Radeon(\(TM\))?\s+[789]\d\d[MS]\b|Radeon(\(TM\))?\s+80[56]0S|Instinct|\bMI\d\d\d", re.I)


def available(os_, arch):
    return AVAILABLE.get((os_, arch), ["cpu"])


def assets_for(name, os_, arch):
    """Release asset name patterns (main, extra runtime libraries) for runtime `name`, or None if there is none here."""
    if name not in available(os_, arch):
        return None
    if os_ == "macos":
        return [rf"llama-b\d+-bin-macos-{arch}\.tar\.gz$"], []
    plat = "win" if os_ == "windows" else "ubuntu"
    ext = r"\.zip$" if plat == "win" else r"\.tar\.gz$"
    if name.startswith("cuda"):
        cuda = rf"{name[4:]}\.\d+"
        return [rf"llama-b\d+-bin-{plat}-cuda-{cuda}-{arch}{ext}"], [rf"cudart-llama-(b\d+-)?bin-{plat}-cuda-{cuda}-{arch}{ext}"]
    if name == "rocm":
        return [rf"llama-b\d+-bin-{plat}-rocm-[\d.]+-{arch}{ext}"], []
    if name == "sycl":
        return [rf"llama-b\d+-bin-{plat}-sycl(-fp16)?-{arch}{ext}"], []
    if name == "vulkan":
        return [rf"llama-b\d+-bin-{plat}-vulkan-{arch}{ext}"], []
    cpu = f"cpu-{arch}" if plat == "win" else arch
    return [rf"llama-b\d+-bin-{plat}-{cpu}{ext}"], []


def recommend(hw):
    """(runtime, reason) that suits this computer best."""
    os_, arch, backend = hw["os"], hw["arch"], hw["backend"]
    have = available(os_, arch)
    names = " ".join(hw.get("gpus") or [])
    if backend == "metal":
        return "metal", "Apple GPU"
    if backend == "cuda":
        driver = hw.get("cudaDriver") or "12.0"
        if int(driver.split(".")[0]) >= 13 and "cuda13" in have:
            return "cuda13", f"NVIDIA GPU, driver supports CUDA {driver}"
        if "cuda12" in have:
            return "cuda12", f"NVIDIA GPU, driver supports CUDA {driver} (CUDA 13 needs a newer driver)"
        if "cuda13" in have:
            return "cuda13", "NVIDIA GPU"
    if backend == "vulkan":
        if re.search(r"AMD|Radeon|\bATI\b", names):
            gpu = next((g for g in hw["gpus"] if ROCM_GPU.search(g)), None)
            if gpu and "rocm" in have:
                return "rocm", f"{gpu} is supported by the ROCm build (try Vulkan too: speed varies by card and driver)"
            if gpu:
                return "vulkan", f"no official ROCm build for {os_} {arch}"
            return "vulkan", "this AMD GPU generation is not supported by ROCm builds (RDNA2 / RX 6000 and newer are)"
        if re.search(r"\bArc\b", names) and "sycl" in have:
            return "sycl", "Intel Arc GPU (try Vulkan too: speed varies)"
        return "vulkan", "works on any GPU"
    return "cpu", "no usable GPU"


# ---------- state in config.json ----------

def state(config):
    s = config.setdefault("runtime", {})
    for key in ("installed", "active", "legacy"):
        s.setdefault(key, {})
    return s


def key_of(name, build):
    return f"{name}-b{build}"


def server_path(config, name, build):
    """llama-server of runtime `name`, build `build`; None if it is missing."""
    legacy = state(config)["legacy"].get(key_of(name, build))
    if legacy:
        return legacy if Path(legacy).is_file() else None
    found = next((p for p in (BIN / key_of(name, build)).rglob(SERVER_NAME) if p.is_file()), None)
    return str(found) if found else None


def selected(config):
    """(runtime, build, llama-server path) in use, or None."""
    s = config.get("runtime") or {}
    name = s.get("selected")
    build = (s.get("active") or {}).get(name)
    path = name and build is not None and server_path(config, name, build)
    return (name, build, path) if path else None


def selected_server(config):
    chosen = selected(config)
    return chosen[2] if chosen else None


def point_server(config):
    chosen = selected(config)
    if chosen:
        config["llamaServer"] = chosen[2]


def remove_build(config, name, build):
    s = state(config)
    key = key_of(name, build)
    if key in s["legacy"]:  # files sit directly in bin/: remove those, never the runtime folders next to them
        for item in BIN.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except OSError:
                    pass
        s["legacy"].pop(key)
    else:
        shutil.rmtree(BIN / key, ignore_errors=True)


def register(config, name, build):
    """Make `build` the one in use for `name`; keep it and the build before it, delete older ones."""
    s = state(config)
    builds = sorted(set(s["installed"].get(name, [])) | {build}, reverse=True)
    s["active"][name] = build
    keep = builds[:builds.index(build) + 2]
    for old in builds[len(keep):]:
        remove_build(config, name, old)
    s["installed"][name] = keep
    if not s.get("selected"):
        s["selected"] = name
    point_server(config)


# ---------- smoke test ----------

def required_flags():
    """Every option Hearthwork passes to llama-server (taken from server.command, so they cannot drift apart)."""
    from .server import command
    s = {"context": 1, "fitTargetMiB": 1, "kvCacheType": "q8_0", "slots": 1, "batch": 1, "port": 1}
    cmd = command({"llamaServer": "llama-server", "server": s}, Path("model.gguf"), slot_path="slots")
    return sorted({a for a in cmd if re.fullmatch(r"--?[A-Za-z][\w-]*", a)} - {"--chat-template-file"})


def missing_flags(help_text, flags):
    return [f for f in flags if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", help_text)]


def smoke_test(server):
    """Does this llama-server start and still accept every option Hearthwork uses? -> (ok, why not)."""
    try:
        version = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=60)
        help_ = subprocess.run([str(server), "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"it does not start here: {error}"
    if not re.search(r"build[:\s]+\d+", version.stdout + version.stderr):
        return False, "`--version` printed no build number (it does not run correctly here)"
    missing = missing_flags(help_.stdout + help_.stderr, required_flags())
    if missing:
        return False, f"it no longer accepts: {', '.join(missing)}"
    return True, ""


# ---------- install / update ----------

def build_of(server):
    match = re.search(r"\d+", server_version(server))
    return int(match.group()) if match else 0


def find_release(name, hw, releases):
    """(build, [(asset, url)]) of the newest release that has runtime `name` for this OS/arch, or None."""
    main_patterns, extra_patterns = assets_for(name, hw["os"], hw["arch"])
    for release in releases:
        tag = re.fullmatch(r"b(\d+)", release.get("tag_name", ""))
        names = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
        main = [n for p in main_patterns for n in names if re.match(p, n)]
        extra = [n for p in extra_patterns for n in names if re.match(p, n)]
        if tag and main and len(extra) >= len(extra_patterns):
            return int(tag.group(1)), [(n, names[n]) for n in sorted(main)[-1:] + sorted(extra)[-1:]]
    return None


def install(config, name, hw, releases=None):
    """Download runtime `name` into bin/<name>-b<build>/ and smoke-test it. Only a passing build is registered
    (and becomes the one in use for `name`); otherwise the new folder is deleted. -> build, or None."""
    if assets_for(name, hw["os"], hw["arch"]) is None:
        print(f"{RED}No official llama.cpp build of '{name}' for {hw['os']} {hw['arch']}. "
              f"Available: {', '.join(available(hw['os'], hw['arch']))}.{RESET}")
        return None
    print(f"Looking up the newest llama.cpp {name} build for {hw['os']} {hw['arch']}...")
    try:
        found = find_release(name, hw, releases or github_json(RELEASES))
    except Exception as error:
        print(f"{RED}Could not reach GitHub: {error}{RESET}")
        return None
    if not found:
        print(f"{RED}No recent llama.cpp release has a '{name}' build for {hw['os']} {hw['arch']}. "
              f"See https://github.com/ggml-org/llama.cpp/releases{RESET}")
        return None
    build, files = found
    s = state(config)
    if build in s["installed"].get(name, []) and server_path(config, name, build):
        s["active"][name] = build
        register(config, name, build)
        print(f"  {name} b{build} is already installed and up to date.")
        return build
    print(f"  b{build}: {', '.join(n for n, _ in files)}")
    target = BIN / key_of(name, build)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    ok, why = False, ""
    try:
        archives = []
        for asset, url in files:
            download(url, target / asset, asset)
            archives.append(target / asset)
        extract(archives[0], target)
        server = next((p for p in target.rglob(SERVER_NAME) if p.is_file()), None)
        if not server:
            raise RuntimeError(f"{SERVER_NAME} not found in {archives[0].name}")
        for archive in archives[1:]:  # runtime libraries go next to llama-server
            extract(archive, server.parent, flatten=True)
        for archive in archives:
            archive.unlink()
        if not WINDOWS:
            server.chmod(server.stat().st_mode | 0o111)
        print("  testing the new build...")
        ok, why = smoke_test(server)
    except Exception as error:
        why = str(error) or type(error).__name__
    finally:
        if not ok:
            shutil.rmtree(target, ignore_errors=True)
    if not ok:
        print(f"{RED}  {name} b{build} was not installed: {why}.{RESET}\n"
              f"  The build you have is unchanged; the new download was deleted.")
        return None
    register(config, name, build)
    print(f"{GREEN}  {name} b{build} installed and tested: {server}{RESET}")
    return build


def install_recommended(config, hw):
    name, reason = recommend(hw)
    print(f"  Recommended runtime: {name} ({reason})")
    if not install(config, name, hw):
        return False
    state(config)["selected"] = name
    point_server(config)
    return True


def migrate(config):
    """Adopt a llama.cpp that an older Hearthwork put directly in bin/ as the current runtime, in place
    (a running server keeps its files; nothing is moved or downloaded). True if config changed."""
    path = config.get("llamaServer")
    if (config.get("runtime") or {}).get("selected") or not path or not Path(path).is_file():
        return False
    if BIN.resolve() not in Path(path).resolve().parents:
        return False  # a llama.cpp from PATH (e.g. brew) is used as it is
    folder = Path(path).parent
    files = " ".join(p.name.lower() for p in folder.iterdir())
    hw = config.get("hardware") or detect()
    cuda = re.search(r"(?:cudart|cublas)64_(\d+)|libcudart\.so\.(\d+)", files)
    if "ggml-cuda" in files:
        name = f"cuda{next(g for g in cuda.groups() if g)}" if cuda else recommend(hw)[0]
        if name not in ("cuda13", "cuda12"):
            name = "cuda13" if int((hw.get("cudaDriver") or "12").split(".")[0]) >= 13 else "cuda12"
    else:
        name = next((n for n, tag in (("rocm", "ggml-hip"), ("sycl", "ggml-sycl"), ("vulkan", "ggml-vulkan"),
                                      ("metal", "ggml-metal")) if tag in files),
                    "metal" if hw["backend"] == "metal" else "vulkan" if hw["backend"] == "vulkan" else "cpu")
    try:
        build = build_of(path)
    except (OSError, subprocess.SubprocessError):
        build = 0
    s = state(config)
    s["legacy"][key_of(name, build)] = str(path)
    s["installed"][name] = [build]
    s["active"][name] = build
    s["selected"] = name
    save_config(config)
    return True


def update(config, hw):
    """Newest build of every installed runtime (each smoke-tested; the old build stays for rollback)."""
    s = state(config)
    if not s["installed"]:
        print("No runtime installed yet.")
        return install_recommended(config, hw)
    try:
        releases = github_json(RELEASES)
    except Exception as error:
        print(f"{RED}Could not reach GitHub: {error}{RESET}")
        return False
    done = True
    for name in list(s["installed"]):
        current = s["active"].get(name)
        found = find_release(name, hw, releases)
        if not found:
            print(f"  {name}: no newer build found.")
        elif current is not None and found[0] <= current:
            print(f"  {name} b{current} is the newest build.")
        else:
            was = f"b{current} -> " if current is not None else ""
            print(f"  {name}: {was}b{found[0]}")
            done = bool(install(config, name, hw, releases)) and done
    return done


def rollback(config):
    chosen = selected(config)
    if not chosen:
        print("No runtime selected.")
        return False
    name, build, _ = chosen
    older = sorted((b for b in state(config)["installed"].get(name, []) if b < build), reverse=True)
    path = older and server_path(config, name, older[0])
    if not path:
        print(f"No previous {name} build to go back to (only the build before the current one is kept).")
        return False
    state(config)["active"][name] = older[0]
    point_server(config)
    print(f"{GREEN}Back to {name} b{older[0]} (was b{build}).{RESET}")
    return True


# ---------- update notice ----------

def latest_build(timeout=NOTICE_TIMEOUT):
    request = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=3",
                                     headers={"User-Agent": "hearthwork", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for release in json.load(response):
            tag = re.fullmatch(r"b(\d+)", release.get("tag_name", ""))
            if tag:
                return int(tag.group(1))
    return None


def update_notice(config, now=None, fetch=latest_build):
    """A line like 'llama.cpp b11435 -> b11457 available', or None. Looks the newest release up at most once a day,
    quietly (offline is fine), with a short timeout; between lookups it reuses the last answer."""
    chosen = selected(config)
    if not chosen:
        return None
    s, now = state(config), now or time.time()
    if now - s.get("checked", 0) >= NOTICE_EVERY:
        s["checked"] = now  # also when offline: no second attempt today
        try:
            s["latest"] = fetch() or s.get("latest")
        except Exception:
            pass
        save_config(config)
    latest = s.get("latest")
    if latest and chosen[1] is not None and latest > chosen[1]:
        return f"llama.cpp b{chosen[1]} -> b{latest} available: hearthwork runtime update"
    return None


def print_notice(config):
    note = update_notice(config)
    if note:
        print(f"{CYAN}{note}{RESET}")


# ---------- command ----------

def show(config, hw):
    s = state(config)
    name, reason = recommend(hw)
    chosen = selected(config)
    print(f"\nRuntimes (llama.cpp engines in {BIN})")
    if not s["installed"]:
        print(f"  {YELLOW}none installed by Hearthwork{RESET}" + (f" (using {config['llamaServer']})" if config.get("llamaServer") else ""))
    for rt, builds in s["installed"].items():
        active = s["active"].get(rt)
        mark = f"{GREEN}selected{RESET}" if chosen and chosen[0] == rt else ""
        old = [f"b{b}" for b in builds if b != active]
        print(f"  {'*' if mark else ' '} {rt:<7} b{active}  {mark}" + (f"  {DIM}(previous: {', '.join(old)}){RESET}" if old else ""))
    print(f"\nRecommended for this PC: {GREEN}{name}{RESET}, {reason}")
    have = available(hw["os"], hw["arch"])
    print(f"Available for {hw['os']} {hw['arch']}:")
    for rt in have:
        print(f"  {rt:<7} {LABELS[rt]}" + ("   (installed)" if rt in s["installed"] else ""))
    print(f"\n{DIM}hearthwork runtime install <name> | use <name> | update | rollback{RESET}")


def running_note(config):
    from .server import served_model
    port = (config.get("server") or {}).get("port", 8001)
    if served_model(port, timeout=1):
        print(f"{YELLOW}A model is running with the old runtime; the change applies the next time it starts "
              f"(stop it and start it again).{RESET}")


def main(argv):
    config = load_config()
    migrate(config)
    hw = config.get("hardware") or detect()
    action, rest = (argv[0], argv[1:]) if argv else ("list", [])
    s = state(config)
    if action in ("list", "ls"):
        show(config, hw)
        return 0
    if action in ("install", "use") and len(rest) != 1:
        print(f"Usage: hearthwork runtime {action} <{'|'.join(available(hw['os'], hw['arch']))}>")
        return 2
    if action == "install":
        ok = install(config, rest[0], hw) is not None
        if ok and s.get("selected") != rest[0]:
            print(f"{DIM}Installed next to the others. Switch to it with: hearthwork runtime use {rest[0]}{RESET}")
    elif action == "use":
        if rest[0] not in s["installed"] or not server_path(config, rest[0], s["active"].get(rest[0])):
            print(f"{RED}'{rest[0]}' is not installed.{RESET} Install it with: hearthwork runtime install {rest[0]}")
            return 1
        s["selected"] = rest[0]
        point_server(config)
        print(f"{GREEN}Using {rest[0]} b{s['active'][rest[0]]}.{RESET}")
        running_note(config)
        ok = True
    elif action == "update":
        ok = update(config, hw)
        if ok and selected(config):
            running_note(config)
    elif action == "rollback":
        ok = rollback(config)
        if ok:
            running_note(config)
    else:
        print(f"Unknown runtime command: {action}\nUsage: hearthwork runtime [install <name> | use <name> | update | rollback]")
        return 2
    save_config(config)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
