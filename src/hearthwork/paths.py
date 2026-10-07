"""Where Hearthwork keeps per-user data: settings, the llama.cpp build, prompt caches and benchmark results.

Outside the installed package, so installing or upgrading never touches them:
  Windows  %LOCALAPPDATA%\\hearthwork
  macOS    ~/Library/Application Support/hearthwork
  Linux    $XDG_DATA_HOME/hearthwork  (default ~/.local/share/hearthwork)
Set HEARTHWORK_HOME to use another folder.
"""
import os
import sys
from pathlib import Path


def _data_dir():
    if os.environ.get("HEARTHWORK_HOME"):
        return Path(os.environ["HEARTHWORK_HOME"]).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "hearthwork"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "hearthwork"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "hearthwork"


HOME = _data_dir()
CONFIG = HOME / "config.json"
BIN = HOME / "bin"                  # llama.cpp, downloaded by setup
SLOTS = HOME / "cache" / "slots"    # saved prompt caches
BENCH = HOME / "bench"              # benchmark runs and scoreboard
TEMPLATES = HOME / "templates"      # optional fixed chat templates: <model file name>.jinja
STATE = HOME / "server.json"        # the background server: pid, model, port
LOG = HOME / "server.log"           # its output on macOS/Linux (Windows shows it in its own window)

HOME.mkdir(parents=True, exist_ok=True)
