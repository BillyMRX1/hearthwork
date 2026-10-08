"""The client side of LAN sharing: `hearthwork connect`, `hearthwork disconnect`, and using the host's model.

config["remote"] = {"host", "port", "key", "name", "hostName", "addresses"} once paired. "addresses" lists every way
to reach the host (LAN IP, Tailscale IP, MagicDNS name); older configs have only "host". Each run tries the LAN first,
then Tailscale (session), and sets remote["host"] to the one that answered, in memory only, with remote["via"] naming it. The agent commands, the menu, `status`
and `bench` then talk to the host (which normalizes requests) instead of starting a model here.
"""
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from .onboard import CYAN, GREEN, RED, RESET, YELLOW, ask, load_config, save_config
from .share import (DEFAULT_PORT, DISCOVERY_PORT, DISCOVERY_QUERY, is_tailscale_ip, lan_addresses, parse_discovery,
                    tailscale_info)

LAN_TIMEOUT, TAILSCALE_TIMEOUT = 1.0, 3.0   # connect timeouts per route
PEER_TIMEOUT = 1.5


class RemoteError(Exception):
    """The host cannot be used; the message says what to do."""


def call(host, port, path, key=None, body=None, timeout=5):
    """(status, JSON body) of a request to the host. Raises OSError when it cannot be reached."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"http://{host}:{port}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except urllib.error.URLError as error:
        raise OSError(str(error.reason))
    try:
        return status, json.loads(raw or b"{}")
    except ValueError:
        return status, {}


def parse_host(text, default_port=DEFAULT_PORT):
    host, _, port = text.strip().partition(":")
    if port and not port.isdigit():
        raise ValueError(f"'{text}' is not host or host:port")
    return host, int(port) if port else default_port


def discover(timeout=2.0):
    """Hosts that answer a broadcast on the local network (one entry per host address)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.2)
    targets = {"255.255.255.255"} | {ip.rsplit(".", 1)[0] + ".255" for ip in lan_addresses()}  # per-interface (assumes /24)
    for target in sorted(targets):
        try:
            sock.sendto(DISCOVERY_QUERY, (target, DISCOVERY_PORT))
        except OSError:
            pass
    found, end = {}, time.time() + timeout
    while time.time() < end:
        try:
            data, address = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        host = parse_discovery(data, address[0])
        if host:
            found[host["host"]] = host
    sock.close()
    return list(found.values())


def is_tailscale_address(address):
    return is_tailscale_ip(address) or address.lower().rstrip(".").endswith(".ts.net")


def route_label(address):
    return "Tailscale" if is_tailscale_address(address) else "LAN"


def ordered_addresses(remote):
    """Every address of the host to try, LAN first, then Tailscale (older configs: just "host")."""
    found = []
    for address in [remote["host"]] + list(remote.get("addresses") or []):
        if address and address not in found:
            found.append(address)
    return sorted(found, key=is_tailscale_address)  # stable: LAN addresses keep their order


def merged_addresses(host, info):
    """Addresses to remember for a host paired through `host`, from its /hearthwork/info (older hosts: just `host`)."""
    given = info.get("addresses") if isinstance(info.get("addresses"), dict) else {}
    found = [host]
    for address in list(given.get("lan") or []) + list(given.get("tailscale") or []) + [given.get("dns") or ""]:
        if isinstance(address, str) and address and address not in found:
            found.append(address)
    return found


def discover_tailscale(timeout=PEER_TIMEOUT, port=DEFAULT_PORT):
    """Hearthwork hosts among the online Tailscale peers (asked concurrently), as discover() entries marked "tailscale"."""
    state = tailscale_info()
    peers = [p for p in (state or {}).get("peers", []) if p["online"] and p["ips"]]
    found, lock = [], threading.Lock()

    def ask_peer(peer):
        try:
            status, info = call(peer["ips"][0], port, "/hearthwork/info", timeout=timeout)
        except OSError:
            return
        if status == 200 and info.get("hearthwork"):
            with lock:
                found.append({"host": peer["ips"][0], "port": port, "hostname": str(info.get("host") or peer["hostname"]),
                              "model": str(info.get("model") or ""), "version": str(info.get("hearthwork") or ""),
                              "tailscale": True})
    threads = [threading.Thread(target=ask_peer, args=(p,), daemon=True) for p in peers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1)
    return sorted(found, key=lambda h: h["hostname"])


def find_hosts():
    """LAN hosts (broadcast) followed by Tailscale peers that run `hearthwork share` and are not already listed."""
    results, lan = [], []
    thread = threading.Thread(target=lambda: results.extend(discover_tailscale()), daemon=True)
    thread.start()
    lan = discover()
    thread.join(PEER_TIMEOUT + 2)
    names = {h["hostname"] for h in lan}
    return lan + [h for h in results if h["hostname"] not in names]


def unreachable(remote, error):
    tried = ", ".join(ordered_addresses(remote))
    return (f"Cannot reach {remote.get('hostName') or remote['host']} ({tried}; port {remote['port']}): {error}\n"
            "The computer may be off or asleep, not running `hearthwork share`, or on another network. "
            "Run `hearthwork connect` to find it again, or `hearthwork disconnect` to go back to local models.")


def resolve(remote):
    """(address, info, models status) for the first address of the host that answers as this host, LAN before
    Tailscale. Raises OSError (the last error) when none does; a 401 is kept as the answer when no other address works."""
    last, refused = OSError("no address"), None
    for address in ordered_addresses(remote):
        timeout = TAILSCALE_TIMEOUT if is_tailscale_address(address) else LAN_TIMEOUT
        try:
            status, info = call(address, remote["port"], "/hearthwork/info", timeout=timeout)
            if status != 200 or not info.get("hearthwork"):
                last = OSError(f"{address}:{remote['port']} is not a Hearthwork share")
                continue
            status, _ = call(address, remote["port"], "/v1/models", key=remote["key"], timeout=5)
        except OSError as error:
            last = error
            continue
        if status == 401:  # maybe another Hearthwork host on a network that reuses this LAN address
            refused = refused or (address, info, status)
            continue
        return address, info, status
    if refused:
        return refused
    raise last


def session(config):
    """(model name, context) of the host's model, checking that the host answers and still trusts this device.
    Sets remote["host"] to the address that answered (this run only) and remote["via"] to "LAN <ip>" / "Tailscale <ip>".
    Raises RemoteError with what to do otherwise."""
    remote = config["remote"]
    try:
        address, info, status = resolve(remote)
    except OSError as error:
        raise RemoteError(unreachable(remote, error))
    remote["host"], remote["via"] = address, f"{route_label(address)} {address}"
    if status == 401:
        raise RemoteError(f"{remote.get('hostName') or remote['host']} no longer trusts this device (it was removed). "
                          "Run `hearthwork connect` to pair again, or `hearthwork disconnect` to go back to local models.")
    if not info.get("model"):
        raise RemoteError(f"{remote.get('hostName') or remote['host']} is sharing, but no model is running on it.")
    slots = info.get("slots")
    remote["slots"] = slots if isinstance(slots, int) and slots > 0 else None   # in memory only, like "via"
    return info["model"], int(info.get("context") or 32768)


def remote_slots(config, default=2):
    """How many sessions the host's model server runs in parallel (from /hearthwork/info "slots", read by the last
    session()/require() call). `default` when not connected, not asked yet, or the host is older and does not say."""
    slots = (config.get("remote") or {}).get("slots")
    return slots if isinstance(slots, int) and slots > 0 else default


def status_lines(config, got):
    """The `hearthwork status` block in remote mode; `got` is the (model, context) of session()/require(), or None."""
    remote = config["remote"]
    name = remote.get("hostName") or remote["host"]
    lines = [f"connected to: {name} as {remote['name']}"]
    if remote.get("via"):
        lines.append(f"address: {remote['host']}:{remote['port']} (via {route_label(remote['host'])})")
        others = [a for a in ordered_addresses(remote) if a != remote["host"]]
        if others:
            lines.append(f"other addresses: {', '.join(others)}")
    else:  # nothing answered: show what was tried
        lines.append(f"address: {', '.join(ordered_addresses(remote))}:{remote['port']} (not reachable)")
    if got:
        slots = remote_slots(config, None)
        lines.append(f"model: {GREEN}{got[0]}{RESET}  (shared, context {got[1]:,} = {got[1] // 1024}K"
                     + (f", {slots} parallel session{'s' if slots != 1 else ''}" if slots else "") + ")")
    else:
        lines.append("model: unavailable")
    return lines


def require(config):
    """session(config), or print why not and return None."""
    try:
        return session(config)
    except RemoteError as error:
        print(f"{RED}{error}{RESET}")
        return None


def connect_main(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="hearthwork connect")
    parser.add_argument("host", nargs="?", help="host or host:port (default: look on the local network)")
    parser.add_argument("--pin", help="the PIN shown on the host (otherwise asked)")
    parser.add_argument("--name", help="name for this device on the host (default: this computer's name)")
    args = parser.parse_args(argv)
    if args.host:
        try:
            host, port = parse_host(args.host)
        except ValueError as error:
            print(error)
            return 2
    else:
        print("Looking for computers sharing a model on this network and your Tailscale network...")
        hosts = find_hosts()
        if not hosts:
            print(f"{YELLOW}None found.{RESET} Is `hearthwork share` running on the other computer, on the same network? "
                  "Or give its address: `hearthwork connect <ip or name>[:port]` (e.g. a Tailscale IP).")
            return 1
        for i, h in enumerate(hosts, 1):
            print(f"  {i}) {h['hostname']}  {h['host']}:{h['port']}  {h['model']}" + ("  (tailscale)" if h.get("tailscale") else ""))
        choice = 1
        if len(hosts) > 1:
            answer = ask(f"Connect to which? [1-{len(hosts)}, Enter = 1]: ")
            choice = int(answer) if answer.isdigit() and 1 <= int(answer) <= len(hosts) else 1
        host, port = hosts[choice - 1]["host"], hosts[choice - 1]["port"]
    try:
        status, info = call(host, port, "/hearthwork/info")
    except OSError as error:
        print(f"{RED}Cannot reach {host}:{port}: {error}{RESET}\nIs `hearthwork share` running there, and did "
              "Windows Firewall allow it on private networks?")
        return 1
    if status != 200 or "hearthwork" not in info:
        print(f"{RED}{host}:{port} did not answer like a Hearthwork share.{RESET}")
        return 1
    host_name = info.get("host") or host
    config = load_config()
    old = config.get("remote") or {}
    addresses = merged_addresses(host, info)
    if port == old.get("port") and old.get("key") and set(addresses) & set(ordered_addresses(old) if old.get("host") else []):
        try:
            if call(host, port, "/v1/models", key=old["key"])[0] == 200:
                known = ordered_addresses(old)
                config["remote"] = dict(old, hostName=host_name, addresses=known + [a for a in addresses if a not in known])
                save_config(config)
                print(f"{GREEN}Already connected to {host_name}.{RESET} Model: {info.get('model')}")
                return 0
        except OSError:
            pass
    device = args.name or ask(f"Name for this computer [{socket.gethostname()}]: ", socket.gethostname())
    try:
        status, reply = call(host, port, "/hearthwork/pair", body={"device": device})
    except OSError as error:
        print(f"{RED}Lost the connection to {host}: {error}{RESET}")
        return 1
    if status != 200:
        print(f"{RED}{reply.get('error', {}).get('message', 'The host refused to pair.')}{RESET}")
        return 1
    print(f"{CYAN}{host_name} now shows a 6-digit PIN in its terminal.{RESET}")
    request, pin = reply["request"], args.pin
    while True:
        pin = pin or ask("PIN: ")
        try:
            status, reply = call(host, port, "/hearthwork/pair/confirm", body={"request": request, "pin": pin})
        except OSError as error:
            print(f"{RED}Lost the connection to {host}: {error}{RESET}")
            return 1
        if status == 200:
            break
        print(f"{RED}{reply.get('error', {}).get('message', 'Pairing failed.')}{RESET}")
        left = reply.get("attemptsLeft", 0)
        if status != 403 or args.pin or not left:
            if status == 403:
                print("Too many wrong PINs; run `hearthwork connect` to start again.")
            return 1
        print(f"{left} attempt(s) left.")
        pin = None
    config["remote"] = {"host": host, "port": port, "key": reply["key"], "name": reply["name"], "hostName": host_name,
                        "addresses": addresses}
    save_config(config)
    print(f"{GREEN}Connected to {host_name} as '{reply['name']}'.{RESET} Model: {info.get('model')}\n"
          "Now `hearthwork claude` / `hearthwork codex` use it. `hearthwork disconnect` goes back to local models.")
    return 0


def disconnect_main():
    config = load_config()
    if not config.pop("remote", None):
        print("Not connected to another computer.")
        return 0
    save_config(config)
    print("Disconnected. Hearthwork uses local models again. (The host still lists this device: remove it there with "
          "`hearthwork devices remove <name>`.)")
    return 0
