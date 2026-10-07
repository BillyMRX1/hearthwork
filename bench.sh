#!/bin/sh
# macOS / Linux: benchmark the running model:  ./bench.sh [--agent claude|codex] [--show]
exec python3 "$(dirname "$0")/bench.py" "$@"
