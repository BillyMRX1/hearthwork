"""Shared command line for claude.py / codex.py: connect a harness to the already-running local server."""
import argparse

from harnesses import HARNESSES, launch
from onboard import load_config
from server import served_model


def run(key):
    server = load_config().get("server", {})
    parser = argparse.ArgumentParser(description=f"{HARNESSES[key]['title']} with the local llama.cpp model.")
    parser.add_argument("--port", type=int, default=server.get("port", 8001))
    parser.add_argument("--context", type=int, default=server.get("context", 65536),
                        help="must match the server's context (start.py --context)")
    # Longest single reply. 512 cut off file writes mid-way (the model then retries): seen in a bench run.
    parser.add_argument("--max-output", type=int, default=4096)
    args, extra = parser.parse_known_args()
    name = served_model(args.port)
    if not name:
        print(f"No server on port {args.port}. Start one with hearthwork.bat / ./hearthwork.sh (or start.bat / ./start.sh) first.")
        return 1
    return launch(key, args.port, name, args.context, args.max_output, extra)
