"""A stand-in for llama-server (stdlib only), spawned as a real process by the lifecycle tests.

  python fake_llama.py --port N -m MODEL
Env: FAKE_FAIL_FILE (while this file exists the process dies at startup with an out-of-memory line),
     FAKE_LOAD_DELAY (seconds before it listens).
Special POSTs: /__crash (exit 1 at once), /__oom (log an OOM line, exit 1), /__hang {"seconds": n} (/health answers slowly).
Generation (any other POST): {"stream": true, "count": n, "delay": s} streams n chunks (chunked), else one JSON answer.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
PORT = int(args[args.index("--port") + 1])
MODEL = args[args.index("-m") + 1] if "-m" in args else "model.gguf"
state = {"hang": 0.0}

if os.environ.get("FAKE_FAIL_FILE") and os.path.exists(os.environ["FAKE_FAIL_FILE"]):
    print("ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate CUDA0 buffer", flush=True)
    os._exit(2)
time.sleep(float(os.environ.get("FAKE_LOAD_DELAY") or 0))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, payload, code=200):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            time.sleep(state["hang"])
            return self.reply({"status": "ok"})
        if path == "/v1/models":
            return self.reply({"data": [{"id": MODEL}]})
        if path == "/props":
            return self.reply({"total_slots": 2, "default_generation_settings": {"n_ctx": 4096}})
        if path == "/slots":
            return self.reply([])
        self.reply({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        path = self.path.split("?")[0]
        if path == "/__crash":
            os._exit(1)
        if path == "/__oom":
            print("CUDA error: out of memory", flush=True)
            os._exit(1)
        if path == "/__hang":
            state["hang"] = float(body.get("seconds") or 0)
            return self.reply({"ok": True})
        if path.startswith("/slots/"):
            return self.reply({"n_saved": 0, "n_restored": 0})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for i in range(int(body.get("count") or 3)):
                chunk = f"data: tok{i}\n\n".encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(float(body.get("delay") or 0))
            if body.get("final"):
                chunk = body["final"].encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            if body.get("die"):
                os._exit(1)
            self.wfile.write(b"0\r\n\r\n")
            return
        time.sleep(float(body.get("delay") or 0))
        self.reply({"model": MODEL, "choices": [{"message": {"content": "ok"}}]})


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
