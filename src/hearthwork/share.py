"""The host side of LAN sharing: `hearthwork share`, `hearthwork devices`.

`share` serves the running model on the local network through the same relay the agents use on this computer
(harnesses.make_relay), so strict chat templates work for every client. Every proxied request needs the key of a
trusted device. A client gets its key by pairing: it asks, this terminal shows a 6-digit PIN, the user types the PIN
on the client. Only a SHA-256 of each key is stored (config["share"]["devices"]). Plain HTTP: home networks only.
"""
import hashlib
import hmac
import json
import secrets
import socket
import subprocess
import threading
import time

from . import __version__
from .harnesses import make_relay
from .onboard import CYAN, GREEN, RED, RESET, WINDOWS, YELLOW, load_config, save_config
from .server import served_model

DEFAULT_PORT = 8484
DISCOVERY_PORT = 8485
DISCOVERY_QUERY = b"HEARTHWORK?"
PIN_SECONDS = 120          # a PIN stops working after 2 minutes
PIN_ATTEMPTS = 3           # wrong PINs per pairing request
LOCK_FAILURES = 5          # wrong PINs overall within LOCK_WINDOW lock pairing for LOCK_SECONDS
LOCK_WINDOW = LOCK_SECONDS = 600
SEEN_EVERY = 60            # lastSeen is written at most this often per device


# ---------- pairing ----------

class Pairing:
    """Pairing requests, their PINs, and the lockout. Pure logic: `clock` is injectable for tests."""

    def __init__(self, clock=time.time):
        self.clock, self.requests, self.failures, self.locked_until = clock, {}, [], 0
        self.lock = threading.Lock()

    def locked(self):
        return self.clock() < self.locked_until

    def start(self, device, ip):
        """(request id, PIN), or None while pairing is locked."""
        with self.lock:
            if self.locked():
                return None
            now = self.clock()
            self.requests = {k: v for k, v in self.requests.items() if v["expires"] > now}
            request, pin = secrets.token_urlsafe(8), f"{secrets.randbelow(10 ** 6):06d}"
            self.requests[request] = {"device": device, "ip": ip, "pin": pin, "expires": now + PIN_SECONDS, "left": PIN_ATTEMPTS}
            return request, pin

    def confirm(self, request, pin):
        """(status, detail): ("ok", device name), ("wrong", attempts left), ("expired"|"unknown"|"locked", None)."""
        with self.lock:
            now = self.clock()
            if self.locked():
                return "locked", None
            entry = self.requests.get(request)
            if not entry:
                return "unknown", None
            if entry["expires"] <= now:
                del self.requests[request]
                return "expired", None
            if hmac.compare_digest(str(pin).encode(), entry["pin"].encode()):
                del self.requests[request]
                return "ok", entry["device"]
            entry["left"] -= 1
            if entry["left"] <= 0:
                del self.requests[request]
            self.failures = [t for t in self.failures if now - t < LOCK_WINDOW] + [now]
            if len(self.failures) >= LOCK_FAILURES:
                self.locked_until, self.failures, self.requests = now + LOCK_SECONDS, [], {}
                return "locked", None
            return "wrong", max(entry["left"], 0)


# ---------- devices ----------

def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def find_device(devices, key):
    """The trusted device whose key this is, or None. Compares every entry, so timing tells nothing."""
    if not key:
        return None
    digest, found = hash_key(key), None
    for device in devices:
        if hmac.compare_digest(digest, str(device.get("keyHash", ""))):
            found = device
    return found


def request_key(headers):
    """The key a client sent: `Authorization: Bearer <key>` (Claude Code, Codex) or `x-api-key`."""
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("x-api-key") or "").strip()


def now_text():
    return time.strftime("%Y-%m-%d %H:%M")


def devices_of(config):
    return config.setdefault("share", {}).setdefault("devices", [])


def add_device(name, key):
    """Trust a device: replaces an older one of the same name. Saved at once, so `share` and `devices` agree."""
    config = load_config()
    devices = [d for d in devices_of(config) if d.get("name") != name]
    devices.append({"name": name, "keyHash": hash_key(key), "added": now_text(), "lastSeen": now_text()})
    config["share"]["devices"] = devices
    save_config(config)


def clean_name(text, fallback):
    name = " ".join(str(text or "").split())[:40]
    return name or fallback


# ---------- discovery ----------

def discovery_reply(hostname, port, model):
    return json.dumps({"hostname": hostname, "port": port, "model": model, "version": __version__}).encode()


def parse_discovery(data, ip):
    """A host entry from a discovery reply, or None when it is not one."""
    try:
        info = json.loads(data.decode("utf-8"))
        return {"host": ip, "port": int(info["port"]), "hostname": str(info["hostname"]),
                "model": str(info.get("model") or ""), "version": str(info.get("version") or "")}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def serve_discovery(port, answer):
    """Answer b"HEARTHWORK?" datagrams on UDP `port` with `answer()` (bytes), in a background thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))

    def loop():
        while True:
            try:
                data, address = sock.recvfrom(1024)
                if data.strip() == DISCOVERY_QUERY:
                    sock.sendto(answer(), address)
            except OSError:
                if sock.fileno() == -1:
                    return
    threading.Thread(target=loop, daemon=True).start()
    return sock


def lan_addresses():
    """This computer's private IPv4 addresses on the local network(s)."""
    found = set()
    try:
        found.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    try:  # the interface a packet to the LAN would leave by (nothing is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 9))
            found.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith(("127.", "169.254.", "0.")))


# ---------- network guard ----------

def parse_profiles(text):
    """Names of the Public networks in `Name|Category` lines (output of the PowerShell query)."""
    public = []
    for line in text.splitlines():
        name, _, category = line.rpartition("|")
        if category.strip().lower() == "public":
            public.append(name.strip() or "unnamed network")
    return public


def public_networks():
    # Only Windows tells a home network from a public one (the network profile the user picked). macOS and Linux
    # have no such setting, so the check is skipped there.
    if not WINDOWS:
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-NetConnectionProfile | ForEach-Object { $_.Name + '|' + $_.NetworkCategory }"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_profiles(out)


# ---------- the shared server ----------

class Share:
    def __init__(self, config, port):
        self.config, self.port, self.pairing = config, port, Pairing()
        self.seen, self.written = set(), {}
        self.model_port = config["server"]["port"]

    def info(self):
        return {"hearthwork": __version__, "host": socket.gethostname(), "model": served_model(self.model_port),
                "context": self.config["server"]["context"], "pairing": not self.pairing.locked()}

    def authenticate(self, handler):
        config = load_config()  # fresh each time: `hearthwork devices remove` takes effect at once
        device = find_device(devices_of(config), request_key(handler.headers))
        if device:
            self.touch(device["name"], handler.client_address[0])
        return device

    def touch(self, name, ip):
        now = time.time()
        if name not in self.seen:
            self.seen.add(name)
            print(f"{GREEN}{name} ({ip}) connected.{RESET}", flush=True)
        if now - self.written.get(name, 0) > SEEN_EVERY:
            self.written[name] = now
            config = load_config()
            for device in devices_of(config):
                if device.get("name") == name:
                    device["lastSeen"] = now_text()
            save_config(config)

    def pair(self, handler):
        try:
            data = json.loads(handler.read_body(4096) or b"{}")
            device = clean_name(data.get("device"), handler.client_address[0])
        except (ValueError, AttributeError):
            return handler._send_json(400, {"error": {"message": "bad request"}})
        started = self.pairing.start(device, handler.client_address[0])
        if not started:
            return handler._send_json(429, {"error": {"message": "Pairing is locked for a few minutes after too many wrong PINs."}})
        request, pin = started
        print(f"\n{CYAN}{device} ({handler.client_address[0]}) wants to connect. PIN: {pin}{RESET}"
              f"  (valid {PIN_SECONDS // 60} minutes)", flush=True)
        handler._send_json(200, {"request": request})

    def confirm(self, handler):
        try:
            data = json.loads(handler.read_body(4096) or b"{}")
            request, pin = str(data.get("request", "")), str(data.get("pin", ""))
        except (ValueError, AttributeError):
            return handler._send_json(400, {"error": {"message": "bad request"}})
        status, detail = self.pairing.confirm(request, pin)
        ip = handler.client_address[0]
        if status == "ok":
            key = secrets.token_urlsafe(32)
            add_device(detail, key)
            print(f"{GREEN}{detail} ({ip}) is now a trusted device.{RESET}", flush=True)
            return handler._send_json(200, {"key": key, "name": detail})
        if status == "wrong":
            print(f"{YELLOW}Wrong PIN from {ip} ({detail} attempts left).{RESET}", flush=True)
            return handler._send_json(403, {"error": {"message": "Wrong PIN."}, "attemptsLeft": detail})
        if status == "locked":
            print(f"{RED}Too many wrong PINs from {ip}: pairing is locked for {LOCK_SECONDS // 60} minutes.{RESET}", flush=True)
            return handler._send_json(429, {"error": {"message": "Pairing is locked for a few minutes after too many wrong PINs."}})
        message = "That PIN expired. Run `hearthwork connect` again." if status == "expired" else "Unknown pairing request."
        handler._send_json(410 if status == "expired" else 404, {"error": {"message": message}})

    def intercept(self, handler):
        path = handler.path.split("?")[0].rstrip("/")
        if path.startswith("/hearthwork/"):
            if path == "/hearthwork/info" and handler.command == "GET":
                handler._send_json(200, self.info())
            elif path == "/hearthwork/pair" and handler.command == "POST":
                self.pair(handler)
            elif path == "/hearthwork/pair/confirm" and handler.command == "POST":
                self.confirm(handler)
            else:
                handler._send_json(404, {"error": {"message": "not found"}})
            return True
        if self.authenticate(handler):
            return False
        handler._send_json(401, {"type": "error", "error": {
            "type": "authentication_error",
            "message": "This device is not trusted by the host. Run `hearthwork connect` on this computer to pair it."}})
        return True


def share_main(argv):
    import argparse
    from .menu import ensure_server
    parser = argparse.ArgumentParser(prog="hearthwork share")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-public", action="store_true", help="share even on a network Windows calls Public")
    args = parser.parse_args(argv)
    public = public_networks()
    if public and not args.allow_public:
        print(f"{RED}Not sharing: this computer is on a Public network ({', '.join(public)}).{RESET}\n"
              "Sharing uses plain HTTP, so it is only for your own home network. If this is your home network, mark it "
              "private:\n  Windows Settings > Network & internet > Wi-Fi (or Ethernet) > your network > Network profile type > Private network\n"
              "or run `hearthwork share --allow-public` if you are sure.")
        return 1
    from .cli import configured
    config = configured()
    if not ensure_server(config):
        return 1
    share = Share(config, args.port)
    try:
        server = make_relay(config["server"]["port"], ("0.0.0.0", args.port), share.intercept)
    except OSError as error:
        print(f"{RED}Cannot listen on port {args.port}: {error}{RESET}")
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        serve_discovery(DISCOVERY_PORT, lambda: discovery_reply(socket.gethostname(), args.port, served_model(share.model_port)))
    except OSError as error:
        print(f"{YELLOW}Discovery is off (UDP port {DISCOVERY_PORT}: {error}); clients must use `hearthwork connect <ip>`.{RESET}")
    ips = lan_addresses()
    print(f"\n{GREEN}Sharing {served_model(share.model_port)} on port {args.port}.{RESET}  Ctrl+C stops sharing.")
    print("This computer's address: " + (", ".join(f"{ip}:{args.port}" for ip in ips) or "(no local network found)"))
    print(f"On the other computer: {CYAN}hearthwork connect{RESET}" + (f"   (or `hearthwork connect {ips[0]}`)" if ips else ""))
    print(f"{len(devices_of(config))} trusted device(s); `hearthwork devices` lists them. A PIN will show here when a computer asks to connect.")
    if WINDOWS:
        print(f"{YELLOW}Windows Firewall may ask to allow Python: allow it on private networks, or other computers cannot "
              f"connect.{RESET}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopped sharing.")
    return 0


def devices_main(argv):
    config = load_config()
    devices = devices_of(config)
    if argv and argv[0] == "remove":
        name = " ".join(argv[1:])
        kept = [d for d in devices if d.get("name") != name]
        if len(kept) == len(devices):
            print(f"No trusted device named '{name}'. `hearthwork devices` lists them.")
            return 1
        config["share"]["devices"] = kept
        save_config(config)
        print(f"Removed {name}. It can no longer use this computer's model.")
        return 0
    if argv:
        print("Usage: hearthwork devices [remove <name>]")
        return 2
    if not devices:
        print("No trusted devices. Run `hearthwork share` here and `hearthwork connect` on the other computer.")
    for device in devices:
        print(f"  {device.get('name'):<24} added {device.get('added', '-')}   last seen {device.get('lastSeen', '-')}")
    return 0
