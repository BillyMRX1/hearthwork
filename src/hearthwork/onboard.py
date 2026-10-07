#!/usr/bin/env python3
"""Onboarding: detect this computer, get the right llama.cpp, pick settings, find the models folder.

Runs automatically the first time `hearthwork` is used. Run it again any time with `hearthwork setup`;
`--update-llama` also replaces llama.cpp with the newest build.

Settings are sized from the hardware. The biggest speed factor is how much of the model sits on the GPU,
which llama.cpp's --fit maximises at start time on any machine; so the context is kept at what Claude Code
needs (its own prompt is ~20K tokens) instead of growing it, which would push model layers off the GPU.
"""
import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .paths import BIN, CONFIG, HOME
WINDOWS, MAC = os.name == "nt", sys.platform == "darwin"
SERVER_NAME = "llama-server.exe" if WINDOWS else "llama-server"
GREEN, YELLOW, RED, CYAN, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"
if WINDOWS:
    os.system("")  # ANSI colors in the Windows console

# Suggested when the models folder is empty: the best in Hearthwork's benchmark (bench.py) on the reference PC.
# `hearthwork model <repo>` lists every file in a repo with its size and whether it fits this computer; hearthwork-check
# ranks a longer list for this machine.
SUGGESTED = [
    ("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B: MoE, fast even partly in RAM. Best in the benchmark (8/8 with "
     "Claude Code and Codex). Q4_K_M is 20.5 GB; smaller quantizations exist."),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B: coding agent, MoE. 8/8 in the benchmark. "
     "Q4_K_M is 17.3 GB, so a better fit for smaller machines."),
]


# ---------- config ----------

def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, ValueError):
        return {}


def save_config(config):
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")


def ask(prompt, default=""):
    try:
        answer = input(prompt).replace("﻿", "").strip().strip('"').strip("'")  # PowerShell pipes add a BOM
    except EOFError:
        sys.exit("\nNo input; exiting.")
    return answer or default


def yes(prompt, default=True):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}] ").lower()
    return default if not answer else answer.startswith("y")


def run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------- hardware ----------

def memory_gb():
    """(total, available) system RAM in GB."""
    if WINDOWS:
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status(length=ctypes.sizeof(Status))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total / 2**30, status.available / 2**30
    if MAC:
        total = int(run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30
        return total, total * 0.6  # macOS keeps RAM as cache; a fixed share is a fair estimate
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    kb = lambda key: int(info.get(key, "0 kB").split()[0])  # noqa: E731
    return kb("MemTotal") / 2**20, kb("MemAvailable") / 2**20


def detect():
    """What this computer has, as far as running models is concerned."""
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine())
    ram_total, _ = memory_gb()
    hw = {"os": "windows" if WINDOWS else "macos" if MAC else "linux", "arch": arch,
          "ramGB": round(ram_total, 1), "cpuThreads": os.cpu_count(), "gpus": [], "vramGB": 0, "backend": "cpu"}
    if MAC:
        hw["gpus"] = ["Apple GPU (Metal, unified memory)" if arch == "arm64" else "Mac GPU (Metal)"]
        hw["backend"] = "metal"
        # The GPU can use up to ~3/4 of unified memory by default.
        hw["vramGB"] = round(ram_total * 0.75, 1) if arch == "arm64" else 0
        return hw
    if shutil.which("nvidia-smi"):
        rows = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).strip().splitlines()
        gpus = [(name.strip(), int(mib) / 1024) for name, mib in (r.rsplit(",", 1) for r in rows if "," in r)]
        if gpus:
            hw["gpus"] = [f"{name} ({gb:.0f} GB)" for name, gb in gpus]
            hw["vramGB"] = round(sum(gb for _, gb in gpus), 1)
            hw["backend"] = "cuda"
            # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones.
            version = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", run(["nvidia-smi"]))
            if version:
                hw["cudaDriver"] = f"{version.group(1)}.{version.group(2)}"
            else:  # drivers 580+ support CUDA 13
                driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip().split(".")[0]
                hw["cudaDriver"] = "13.0" if driver.isdigit() and int(driver) >= 580 else "12.0"
            return hw
    # AMD / Intel: Vulkan works across vendors. VRAM is measured by llama.cpp itself at start (--fit).
    if WINDOWS:
        names = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]).splitlines()
    else:
        names = [line.split(":", 2)[-1] for line in run(["lspci"]).splitlines() if re.search(r"VGA|3D|Display", line)]
    names = [n.strip() for n in names if n.strip() and not re.search(r"Microsoft Basic|Remote|Virtual", n)]
    if any(re.search(r"AMD|Radeon|Intel|NVIDIA", n, re.I) for n in names):
        hw["gpus"] = names
        hw["backend"] = "vulkan"
        hw["vramGB"] = round(dedicated_vram_gb(), 1)
    return hw


def dedicated_vram_gb():
    """Largest dedicated VRAM among AMD/Intel GPUs (integrated GPUs report little or none). 0 if unknown."""
    sizes = []
    if WINDOWS:  # the display-adapter class key holds the real size (WMI's AdapterRAM caps at 4 GB)
        out = run(["powershell", "-NoProfile", "-Command",
                   "Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
                   "-ErrorAction SilentlyContinue | ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" }"])
        for line in out.splitlines():
            name, _, size = line.rpartition("|")
            if size.strip().isdigit() and not re.search(r"NVIDIA", name, re.I):
                sizes.append(int(size.strip()))
    else:
        for path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
            try:
                sizes.append(int(path.read_text().strip()))
            except (OSError, ValueError):
                pass
    return max(sizes, default=0) / 2**30


def judge(hw):
    """Is a local model worth it on this computer for Claude Code? -> (verdict, reasons, what to expect).

    Reference, measured: RTX 5060 Ti 16 GB + 22.6 GB RAM runs Qwen3-Coder-30B-A3B Q4_K_M (17 GB; ~20 GB with a
    64K context) at 18-31 tokens/s and completed real coding tasks. Smaller ~9B models were much weaker with
    Claude Code's tools. Thresholds for other hardware are estimates from that.
    """
    vram, ram, backend = hw.get("vramGB") or 0, hw.get("ramGB", 0), hw.get("backend")
    if backend == "metal":
        usable, gpu_ok = ram * 0.75, hw.get("arch") == "arm64"
        big_gpu, small_gpu = ram >= 32, ram >= 16
    else:
        usable = vram + max(0.0, ram - 6)  # ~6 GB of RAM stays with Windows/macOS/Linux and other programs
        gpu_ok, big_gpu, small_gpu = backend in ("cuda", "vulkan"), vram >= 12, vram >= 6
    reasons = []
    if not gpu_ok or not small_gpu:
        reasons.append("no GPU with enough memory of its own" if backend != "metal" else "Intel Mac: no fast GPU")
        reasons.append("on the CPU a capable coding model generates only a few tokens/s, and Claude Code's ~20K-token "
                       "prompt takes minutes to read")
    if usable < 18:
        reasons.append(f"~{usable:.0f} GB usable for models: a capable coding model needs ~17-20 GB with Claude Code's "
                       "context; smaller models handle Claude Code's tools poorly")
    if reasons:
        return "not recommended", reasons, ("Too slow or too weak to be useful for coding. Cloud Claude Code is the "
                                            "better choice on this computer.")
    if big_gpu and usable >= 26:  # the reference PC has ~32 GB usable
        return "recommended", [f"{vram if backend != 'metal' else usable:.0f} GB GPU memory, ~{usable:.0f} GB usable in total"], (
            "Similar to the reference PC: about 20-30 tokens/s with a 30B coding model, ~30 s for the first message, "
            "then seconds to a minute per step. Fine for small, well-defined coding tasks; weaker than cloud Claude.")
    return "limited", [f"{vram if backend != 'metal' else usable:.0f} GB GPU memory, ~{usable:.0f} GB usable in total: "
                       "a 30B coding model needs a smaller quantization or runs mostly from RAM"], (
        "Works, but expect roughly 5-15 tokens/s and/or lower answer quality. OK for experimenting; "
        "cloud Claude Code will be much faster for real work.")


def settings_for(hw):
    """Server settings for this hardware (all overridable in config.json -> "server")."""
    vram = hw.get("vramGB") or 0
    total = vram + hw.get("ramGB", 0) if hw.get("backend") != "metal" else hw.get("ramGB", 0)
    return {
        # Claude Code's own prompt is ~20K tokens; 32K makes it compact after the first message.
        "context": 65536 if total >= 24 else 32768,
        # Bigger prompt batches are much faster (20K-token prompt: 84 s at 512, 26 s at 4096) but need VRAM.
        "batch": 4096 if vram >= 12 else 2048 if vram >= 6 else 512,
        # VRAM left free for the desktop and other programs; --fit fills the rest.
        "fitTargetMiB": 1536 if vram >= 12 else 1024,
        "kvCacheType": "q8_0",  # half the memory of f16 at nearly the same quality
        "slots": 2,             # Claude Code's background requests use their own slot
        "port": 8001,
    }


# ---------- llama.cpp ----------

def server_version(path):
    out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=30)
    match = re.search(r"build (\d+)", out.stdout + out.stderr)  # llama.cpp prints its version to stderr
    return f"build {match.group(1)}" if match else "unknown version"


def wanted_assets(hw):
    """Release asset name patterns (main, extra runtime) for this computer."""
    arch, backend = hw["arch"], hw["backend"]
    if hw["os"] == "macos":
        return [rf"llama-b\d+-bin-macos-{arch}\.tar\.gz$"], []
    plat = "win" if hw["os"] == "windows" else "ubuntu"
    ext = r"\.zip$" if plat == "win" else r"\.tar\.gz$"
    if backend == "cuda":
        major = int((hw.get("cudaDriver") or "12.0").split(".")[0])
        cuda = r"13\.\d+" if major >= 13 else r"12\.\d+"
        return [rf"llama-b\d+-bin-{plat}-cuda-{cuda}-{arch}{ext}"], [rf"cudart-llama-(b\d+-)?bin-{plat}-cuda-{cuda}-{arch}{ext}"]
    if backend == "vulkan":
        return [rf"llama-b\d+-bin-{plat}-vulkan-{arch}{ext}"], []
    cpu = f"cpu-{arch}" if plat == "win" else ("x64" if arch == "x64" else "arm64")
    return [rf"llama-b\d+-bin-{plat}-{cpu}{ext}"], []


def github_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "hearthwork", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download(url, target, label):
    """Download with a progress line; resumes a partial .part file."""
    part = Path(str(target) + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "hearthwork"}
    if done:
        headers["Range"] = f"bytes={done}-"
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if done and response.status != 206:  # server ignored the range: start over
            done = 0
        total = done + int(response.headers.get("content-length") or 0)
        with open(part, "ab" if done else "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                pct = f"{done / total:6.1%}" if total else ""
                print(f"\r  {label}: {done / 2**30:6.2f} / {total / 2**30:.2f} GB {pct}", end="", flush=True)
    print()
    part.replace(target)


def install_llama(hw, config):
    """Download the newest official llama.cpp build for this computer into bin/."""
    main_patterns, extra_patterns = wanted_assets(hw)
    print(f"Looking up the newest llama.cpp build for {hw['os']} {hw['arch']} ({hw['backend']})...")
    for release in github_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"):
        names = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
        main = [n for p in main_patterns for n in names if re.match(p, n)]
        extra = [n for p in extra_patterns for n in names if re.match(p, n)]
        if main and len(extra) >= len(extra_patterns):
            break
    else:
        sys.exit(f"{RED}No matching llama.cpp build found. Get one from https://github.com/ggml-org/llama.cpp/releases "
                 f"and run setup again.{RESET}")
    files = sorted(main)[-1:] + sorted(extra)[-1:]
    print(f"  release {release['tag_name']}: {', '.join(files)}")
    if BIN.exists():
        shutil.rmtree(BIN)
    BIN.mkdir(parents=True)
    archives = []
    for name in files:
        target = BIN / name
        download(names[name], target, name)
        archives.append(target)
    extract(archives[0], BIN)
    server = next((p for p in BIN.rglob(SERVER_NAME) if p.is_file()), None)
    if not server:
        sys.exit(f"{RED}{SERVER_NAME} not found in {archives[0].name}.{RESET}")
    for archive in archives[1:]:  # runtime libraries go next to llama-server
        extract(archive, server.parent, flatten=True)
    for archive in archives:
        archive.unlink()
    if not WINDOWS:
        server.chmod(server.stat().st_mode | 0o111)
    config["llamaServer"] = str(server)
    print(f"{GREEN}  llama.cpp installed: {server} ({server_version(server)}){RESET}")


def extract(archive, target, flatten=False):
    target = Path(target)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                out = target / (Path(member.filename).name if flatten else member.filename)
                if not out.resolve().is_relative_to(target.resolve()):
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(member))
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                if not (member.isfile() or member.issym()):
                    continue
                if flatten:
                    member.name = Path(member.name).name
                if not (target / member.name).resolve().is_relative_to(target.resolve()):
                    continue
                t.extract(member, target, **({"filter": "tar"} if hasattr(tarfile, "data_filter") else {}))


def find_llama(config):
    for candidate in (config.get("llamaServer"), BIN / SERVER_NAME, *BIN.rglob(SERVER_NAME), shutil.which("llama-server")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


# ---------- models folder ----------

def find_models(folder):
    """GGUF models under `folder`. Multi-file models start from their first part; mmproj files are not models."""
    found = []
    for path in sorted(Path(folder).rglob("*.gguf"), key=lambda p: str(p).lower()):
        name = path.name
        if "mmproj" in name.lower() or (re.search(r"-\d{5}-of-\d{5}", name) and "-00001-of-" not in name):
            continue
        found.append(path)
    return found


def lmstudio_folders():
    home = Path.home()
    candidates = [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]
    settings = home / ".lmstudio" / "settings.json"
    try:
        custom = json.loads(settings.read_text(encoding="utf-8")).get("downloadsFolder")
        if custom:
            candidates.insert(0, Path(custom))
    except (OSError, ValueError):
        pass
    return [c for c in candidates if c.is_dir()]


def choose_models_folder(config, reset=False):
    current = config.get("modelsDir")
    if current and not reset and Path(current).is_dir():
        return current
    found = [str(c) for c in lmstudio_folders()]
    default = current or (found[0] if found else str(HOME / "models"))
    print(f"\nWhere should models live? This folder is scanned for .gguf files, and `hearthwork model` downloads into it.")
    if found:
        print(f"  Found LM Studio's models folder: {found[0]} (sharing it lets LM Studio and this tool use the same files)")
    while True:
        answer = ask(f"Models folder [Enter = {default}]: ", default)
        path = Path(os.path.expanduser(answer))
        if path.is_dir() or yes(f"{path} does not exist. Create it?"):
            path.mkdir(parents=True, exist_ok=True)
            config["modelsDir"] = str(path.resolve())
            return config["modelsDir"]


# ---------- onboarding ----------

def suggest_models(hw, settings):
    room = (hw.get("vramGB") or 0) + hw.get("ramGB", 0) * 0.7
    print(f"\n{YELLOW}No models yet.{RESET} This computer can hold a model of roughly {room:.0f} GB "
          f"(GPU memory + ~70% of RAM). Suggested:")
    for repo, about in SUGGESTED:
        print(f"  {repo}\n    {about}")
        print(f"    Download:  {CYAN}hearthwork model {repo}{RESET}")
    print("  Any GGUF model with tool-calling support works; pass its Hugging Face link to `hearthwork model`.")


def onboard(config, update_llama=False, reset_folder=False):
    print(f"{CYAN}=== Setup ==={RESET}\nChecking this computer...")
    hw = detect()
    gpus = ", ".join(hw["gpus"]) or "none usable (CPU only)"
    print(f"  OS: {hw['os']} {hw['arch']}   CPU threads: {hw['cpuThreads']}   RAM: {hw['ramGB']} GB")
    vram_note = f", {hw['vramGB']} GB" if hw.get("vramGB") and hw["backend"] == "vulkan" else ""
    print(f"  GPU: {gpus}{vram_note}   ->  llama.cpp backend: {hw['backend']}"
          + (f" (driver CUDA {hw['cudaDriver']})" if hw.get("cudaDriver") else ""))
    config["hardware"] = hw

    verdict, reasons, expect = judge(hw)
    color = {"recommended": GREEN, "limited": YELLOW}.get(verdict, RED)
    print(f"\n{color}Local model for Claude Code on this computer: {verdict.upper()}{RESET}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"  {expect}")
    hw["verdict"] = verdict
    if verdict == "not recommended" and not yes("Set it up anyway?", default=False):
        save_config(config)
        sys.exit("Skipped. Nothing was downloaded. Run setup again any time to change your mind.")
    if verdict == "limited" and not yes("Continue with setup?", default=True):
        save_config(config)
        sys.exit("Skipped. Nothing was downloaded.")

    server = None if update_llama else find_llama(config)
    if server:
        print(f"  llama.cpp: {server} ({server_version(server)})")
        config["llamaServer"] = server
    elif yes(f"\nllama.cpp is not installed here. Download the newest official {hw['backend']} build (~0.2-0.7 GB)?"):
        install_llama(hw, config)
    else:
        sys.exit(f"Get llama.cpp from https://github.com/ggml-org/llama.cpp/releases (or `brew install llama.cpp`), "
                 f"put it in {BIN}, and run setup again.")

    folder = choose_models_folder(config, reset_folder)
    config["server"] = dict(settings_for(hw), **{k: v for k, v in config.get("server", {}).items() if k == "port"})
    s = config["server"]
    print(f"\nSettings for this computer: context {s['context']} tokens, prompt batch {s['batch']}, "
          f"KV cache {s['kvCacheType']}, {s['fitTargetMiB']} MiB VRAM kept free, {s['slots']} slots, port {s['port']}")
    if s["context"] < 65536:
        print(f"{YELLOW}  Less than ~24 GB of GPU memory + RAM: 32K context. Claude Code's own prompt is ~20K tokens, "
              f"so expect it to compact long conversations often.{RESET}")
    if not shutil.which("claude"):
        print(f"{YELLOW}  Claude Code ('claude') is not installed; get it from https://code.claude.com{RESET}")
    save_config(config)
    print(f"{GREEN}Setup saved to {CONFIG}{RESET}")
    if not find_models(folder):
        suggest_models(hw, s)
    return config


def setup_complete(config):
    return bool(config.get("hardware") and config.get("server") and config.get("modelsDir")
                and Path(config["modelsDir"]).is_dir() and find_llama(config))


def legacy_folders():
    """Folders of a pre-0.2 Hearthwork (a git clone with config.json and bin/ next to the scripts)."""
    here = Path(__file__).resolve()
    candidates = [here.parents[2], Path.cwd(), *Path.cwd().parents]  # a source checkout, then cwd upwards
    found = []
    for folder in candidates:
        try:
            old = json.loads((folder / "config.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if "llamaServer" in old and folder.resolve() != HOME.resolve() and folder not in found:
            found.append(folder)
    return found


def import_legacy(folder, config):
    """Copy settings, the llama.cpp build, prompt caches and benchmark results from an old Hearthwork folder.
    Copies, so the old folder keeps working until you delete it."""
    folder = Path(folder)
    old = json.loads((folder / "config.json").read_text(encoding="utf-8-sig"))
    for key in ("modelsDir", "lastModel", "hardware", "server"):
        if key in old:
            config[key] = old[key]
    for name in ("bin", "cache", "bench", "templates"):
        source = folder / name
        if source.is_dir():
            print(f"  copying {source} -> {HOME / name}")
            shutil.copytree(source, HOME / name, dirs_exist_ok=True)
    config["llamaServer"] = None
    server = find_llama(config)
    if server:
        config["llamaServer"] = server
    save_config(config)
    print(f"{GREEN}Imported your Hearthwork setup from {folder}.{RESET} The old folder was left as it is.")
    return config


def offer_import(config):
    """First run without a config: offer to import an older clone-style setup if one is found."""
    if config.get("modelsDir"):
        return config
    for folder in legacy_folders():
        if yes(f"Found an earlier Hearthwork setup in {folder}. Import its settings, llama.cpp and caches?"):
            return import_legacy(folder, config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hearthwork setup", description="Set up llama.cpp for Claude Code on this computer.")
    parser.add_argument("--update-llama", action="store_true", help="download the newest llama.cpp build")
    parser.add_argument("--reset", action="store_true", help="ask for the models folder again")
    parser.add_argument("--import", dest="import_from", metavar="FOLDER",
                        help="copy settings, llama.cpp and caches from an older Hearthwork folder (a git clone)")
    args = parser.parse_args(argv)
    config = load_config()
    if args.import_from:
        config = import_legacy(args.import_from, config)
    onboard(config, update_llama=args.update_llama, reset_folder=args.reset)


if __name__ == "__main__":
    main()
