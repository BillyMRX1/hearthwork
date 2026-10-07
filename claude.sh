#!/bin/sh
# macOS / Linux: run from the project you want to work on:  /path/to/hearthwork/claude.sh
exec python3 "$(dirname "$0")/claude.py" "$@"
