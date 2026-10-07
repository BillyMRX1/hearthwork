#!/bin/sh
# macOS / Linux:  ./start.sh   (first time: chmod +x start.sh claude.sh)
exec python3 "$(dirname "$0")/start.py" "$@"
