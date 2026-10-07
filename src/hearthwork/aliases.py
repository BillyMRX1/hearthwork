"""`hearthwork alias`: short commands (e.g. `ccl`) that run `hearthwork claude` / `hearthwork codex`.

A launcher is a tiny file next to the installed `hearthwork` command (a .cmd on Windows, a shell script elsewhere),
so it is already on PATH. Plain `claude` and `codex` are never touched. Launchers carry a marker line, and only
files with it are listed, replaced or removed.
"""
import os
import re
import shutil
import sys
from pathlib import Path

from .harnesses import HARNESSES

MARKER = "created by `hearthwork alias`"
WINDOWS = os.name == "nt"
USAGE = """Usage:
  hearthwork alias add <name> claude|codex   create a short command, e.g. `hearthwork alias add ccl claude`
  hearthwork alias list
  hearthwork alias remove <name>"""


def bin_dir():
    """The folder holding the `hearthwork` command (where a launcher is on PATH too), or None."""
    found = shutil.which("hearthwork")
    if found:
        return Path(found).resolve().parent if not WINDOWS else Path(found).parent
    for candidate in (sys.argv[0], sys.executable):
        folder = Path(candidate).resolve().parent
        if (folder / "hearthwork.exe").exists() or (folder / "hearthwork").exists():
            return folder
    return None


def launcher_file(folder, name):
    return folder / (name + (".cmd" if WINDOWS else ""))


def is_ours(path):
    try:
        return MARKER in path.read_text(encoding="utf-8", errors="replace")[:300]
    except OSError:
        return False


def launcher_text(agent):
    if WINDOWS:
        return f"@echo off\r\nrem {MARKER}\r\n@hearthwork {agent} %*\r\n"
    return f'#!/bin/sh\n# {MARKER}\nexec hearthwork {agent} "$@"\n'


def add(name, agent):
    if agent not in HARNESSES:
        print(f"Unknown agent '{agent}'. Choose one of: {', '.join(HARNESSES)}")
        return 2
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        print("Use letters, digits, '-' and '_' for the name.")
        return 2
    folder = bin_dir()
    if not folder:
        print("Could not find where the `hearthwork` command is installed (is it on PATH?).")
        return 1
    target = launcher_file(folder, name)
    if target.exists() and not is_ours(target):
        print(f"{target} already exists and was not created by `hearthwork alias`; not touching it.")
        return 1
    other = shutil.which(name)
    if other and not is_ours(Path(other)) and Path(other).resolve() != target.resolve():
        print(f"`{name}` is already a command on this computer ({other}); pick another name.")
        return 1
    target.write_text(launcher_text(agent), encoding="utf-8", newline="")
    if not WINDOWS:
        target.chmod(0o755)
    print(f"Created {target}\nNow `{name}` runs `hearthwork {agent}` (your plain `{HARNESSES[agent]['binary']}` is unchanged).")
    return 0


def launchers():
    folder = bin_dir()
    if not folder:
        return []
    found = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and (path.suffix.lower() == ".cmd" if WINDOWS else os.access(path, os.X_OK)) and is_ours(path):
            match = re.search(r"hearthwork (\w+)", path.read_text(encoding="utf-8", errors="replace").split(MARKER, 1)[1])
            found.append((path.stem if WINDOWS else path.name, match.group(1) if match else "?", path))
    return found


def remove(name):
    folder = bin_dir()
    target = launcher_file(folder, name) if folder else None
    if not target or not target.exists():
        print(f"No alias named '{name}'.")
        return 1
    if not is_ours(target):
        print(f"{target} was not created by `hearthwork alias`; not removing it.")
        return 1
    target.unlink()
    print(f"Removed {target}")
    return 0


def main(args):
    if len(args) == 3 and args[0] == "add":
        return add(args[1], args[2])
    if len(args) == 1 and args[0] == "list":
        found = launchers()
        for name, agent, path in found:
            print(f"{name}  ->  hearthwork {agent}   ({path})")
        if not found:
            print("No aliases yet. Create one with: hearthwork alias add <name> claude|codex")
        return 0
    if len(args) == 2 and args[0] == "remove":
        return remove(args[1])
    print(USAGE)
    return 2
