#!/bin/sh
# macOS / Linux: download a GGUF model from Hugging Face into your models folder:  ./model.sh <link>
exec python3 "$(dirname "$0")/model.py" "$@"
