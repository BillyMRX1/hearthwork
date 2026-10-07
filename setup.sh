#!/bin/sh
# macOS / Linux: re-run setup (hardware check, llama.cpp download/update, models folder).
exec python3 "$(dirname "$0")/onboard.py" "$@"
