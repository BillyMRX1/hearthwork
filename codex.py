#!/usr/bin/env python3
"""Codex with the local model (server started with hearthwork or start). Run from your project folder:
  codex.bat / ./codex.sh / python codex.py   [--context N] [--max-output N] [any codex arguments]
Your ~/.codex/config.toml is not changed: the local provider is passed with -c overrides.
"""
import sys

from agent_cli import run

if __name__ == "__main__":
    sys.exit(run("codex"))
