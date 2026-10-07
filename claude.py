#!/usr/bin/env python3
"""Claude Code with the local model (server started with hearthwork or start). Run from your project folder:
  claude.bat / ./claude.sh / python claude.py   [--context N] [--max-output N] [any claude arguments]
"""
import sys

from agent_cli import run

if __name__ == "__main__":
    sys.exit(run("claude"))
