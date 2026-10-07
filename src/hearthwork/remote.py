"""The client side of LAN sharing: `hearthwork connect`, `hearthwork disconnect`, and using the host's model.

config["remote"] = {"host", "port", "key", "name", "hostName"} once paired. The agent commands, the menu, `status`
and `bench` then talk to the host (which normalizes requests) instead of starting a model here.
"""
import json
import socket
import sys
import time
import urllib.error
import urllib.request

from .onboard import CYAN, GREEN, RED, RESET, YELLOW, ask, load_config, save_config
from .share import DEFAULT_PORT, DISCOVERY_PORT, DISCOVERY_QUERY, lan_addresses, parse_discovery


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


def unreachable(remote, error):
    return (f"Cannot reach {remote.get('hostName') or remote['host']} ({remote['host']}:{remote['port']}): {error}\n"
            "The computer may be off or asleep, not running `hearthwork share`, or on another network. "
            "Run `hearthwork connect` to find it again, or `hearthwork disconnect` to go back to local models.")


def session(config):
    """(model name, context) of the host's model, checking that the host answers and still trusts this device.
    Raises RemoteError with what to do otherwise."""
    remote = config["remote"]
    try:
        status, info = call(remote["host"], remote["port"], "/hearthwork/info")
        if status != 200 or not info.get("hearthwork"):
            raise RemoteError(f"{remote['host']}:{remote['port']} is not a Hearthwork share. Run `hearthwork connect` again.")
        status, _ = call(remote["host"], remote["port"], "/v1/models", key=remote["key"])
    except OSError as error:
        raise RemoteError(unreachable(remote, error))
    if status == 401:
        raise RemoteError(f"{remote.get('hostName') or remote['host']} no longer trusts this device (it was removed). "
                          "Run `hearthwork connect` to pair again, or `hearthwork disconnect` to go back to local models.")
    if not info.get("model"):
        raise RemoteError(f"{remote.get('hostName') or remote['host']} is sharing, but no model is running on it.")
    return info["model"], int(info.get("context") or 32768)


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
        print("Looking for computers sharing a model on this network...")
        hosts = discover()
        if not hosts:
            print(f"{YELLOW}None found.{RESET} Is `hearthwork share` running on the other computer, on the same network? "
                  "Or give its address: `hearthwork connect <ip>[:port]` (e.g. a Tailscale IP).")
            return 1
        for i, h in enumerate(hosts, 1):
            print(f"  {i}) {h['hostname']}  {h['host']}:{h['port']}  {h['model']}")
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
    if old.get("host") == host and old.get("port") == port and old.get("key"):
        try:
            if call(host, port, "/v1/models", key=old["key"])[0] == 200:
                config["remote"] = dict(old, hostName=host_name)
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
    config["remote"] = {"host": host, "port": port, "key": reply["key"], "name": reply["name"], "hostName": host_name}
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
