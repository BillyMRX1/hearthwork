import json
import unittest
from unittest import mock

from hearthwork import remote, share

STATUS = json.dumps({
    "Version": "1.76.1", "BackendState": "Running", "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
    "Self": {"HostName": "home-pc", "DNSName": "home-pc.tail1234.ts.net.", "OS": "windows",
             "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"], "Online": True},
    "Peer": {
        "nodekey:a": {"HostName": "laptop", "DNSName": "laptop.tail1234.ts.net.", "OS": "macOS",
                      "TailscaleIPs": ["100.64.0.5"], "Online": True},
        "nodekey:b": {"HostName": "phone", "DNSName": "phone.tail1234.ts.net.", "OS": "android",
                      "TailscaleIPs": ["100.64.0.6"], "Online": False},
        "nodekey:c": {"HostName": "broken", "DNSName": "", "TailscaleIPs": None, "Online": True},
    },
})


class TailscaleDetection(unittest.TestCase):
    def test_parse_status(self):
        info = share.parse_tailscale_status(STATUS)
        self.assertEqual(info["ips"], ["100.101.102.103", "fd7a:115c:a1e0::1"])
        self.assertEqual(info["dns"], "home-pc.tail1234.ts.net")
        self.assertEqual([p["hostname"] for p in info["peers"]], ["laptop", "phone"])
        self.assertEqual([p["online"] for p in info["peers"]], [True, False])

    def test_parse_status_not_connected(self):
        self.assertIsNone(share.parse_tailscale_status(json.dumps({"BackendState": "NeedsLogin", "Self": {"TailscaleIPs": None}})))
        self.assertIsNone(share.parse_tailscale_status(json.dumps({"BackendState": "Stopped", "Self": {"TailscaleIPs": ["100.1.1.1"]}})))
        self.assertIsNone(share.parse_tailscale_status("not json"))
        self.assertIsNone(share.parse_tailscale_status(""))

    def test_cgnat_range(self):
        for ip in ("100.64.0.1", "100.127.255.254", "fd7a:115c:a1e0::5", "::ffff:100.100.1.1"):
            self.assertTrue(share.is_tailscale_ip(ip), ip)
        for ip in ("100.63.255.255", "100.128.0.1", "192.168.0.7", "10.0.0.1", "fd00::1", "garbage", ""):
            self.assertFalse(share.is_tailscale_ip(ip), ip)

    def test_lan_addresses_exclude_tailscale(self):
        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("100.101.102.103", 0)), (2, 1, 6, "", ("192.168.0.7", 0))]):
            self.assertEqual(share.lan_addresses(), ["192.168.0.7"])

    def test_no_tailscale_is_silent(self):
        with mock.patch.object(share, "find_tailscale", return_value=None), mock.patch.object(share, "interface_tailscale_ips", return_value=[]):
            self.assertIsNone(share.tailscale_info())

    def test_cli_failure_falls_back_to_interfaces(self):
        with mock.patch.object(share, "find_tailscale", return_value="ts"), \
                mock.patch("subprocess.run", side_effect=OSError), \
                mock.patch.object(share, "interface_tailscale_ips", return_value=["100.90.0.1"]):
            self.assertEqual(share.tailscale_info()["ips"], ["100.90.0.1"])

    def test_cli_status_used(self):
        run = mock.Mock(stdout=STATUS)
        with mock.patch.object(share, "find_tailscale", return_value="ts"), mock.patch("subprocess.run", return_value=run):
            self.assertEqual(share.tailscale_info()["dns"], "home-pc.tail1234.ts.net")


class PeerFiltering(unittest.TestCase):
    def test_default_accepts_everyone(self):
        self.assertTrue(share.peer_allowed("8.8.8.8"))

    def test_tailscale_only_on_public_network(self):
        for ip in ("127.0.0.1", "::1", "100.64.0.5", "fd7a:115c:a1e0::9"):
            self.assertTrue(share.peer_allowed(ip, tailscale_only=True, lan_ok=False), ip)
        for ip in ("192.168.0.9", "10.1.1.1", "8.8.8.8"):
            self.assertFalse(share.peer_allowed(ip, tailscale_only=True, lan_ok=False), ip)

    def test_tailscale_only_with_private_lan(self):
        self.assertTrue(share.peer_allowed("192.168.0.9", tailscale_only=True, lan_ok=True))
        self.assertFalse(share.peer_allowed("8.8.8.8", tailscale_only=True, lan_ok=True))

    def test_profiles_ignore_tailscale(self):
        text = "Home|Wi-Fi|Private\nCafe|Wi-Fi 2|Public\ntailnet.ts.net|Tailscale|Public\n"
        self.assertEqual(share.parse_profiles(text, skip_tailscale=True), ["Cafe"])
        self.assertEqual(share.parse_profiles(text), ["Cafe", "tailnet.ts.net"])
        self.assertEqual(share.parse_profiles("Net|Public"), ["Net"])  # older two-field lines


class Addresses(unittest.TestCase):
    def test_order_lan_first(self):
        r = {"host": "100.64.0.5", "addresses": ["100.64.0.5", "home.tail1.ts.net", "192.168.0.7"]}
        self.assertEqual(remote.ordered_addresses(r), ["192.168.0.7", "100.64.0.5", "home.tail1.ts.net"])

    def test_old_config_only_host(self):
        self.assertEqual(remote.ordered_addresses({"host": "192.168.0.7"}), ["192.168.0.7"])

    def test_merged_from_info(self):
        info = {"addresses": {"lan": ["192.168.0.7"], "tailscale": ["100.1.1.1"], "dns": "pc.x.ts.net"}}
        self.assertEqual(remote.merged_addresses("192.168.0.7", info), ["192.168.0.7", "100.1.1.1", "pc.x.ts.net"])

    def test_info_backward_compat(self):
        self.assertEqual(remote.merged_addresses("10.0.0.2", {"hearthwork": "0.4.1"}), ["10.0.0.2"])
        self.assertEqual(remote.merged_addresses("10.0.0.2", {"addresses": "junk"}), ["10.0.0.2"])

    def test_fallback_to_next_address(self):
        calls = []

        def fake_call(host, port, path, key=None, body=None, timeout=5):
            calls.append((host, timeout))
            if host == "192.0.2.1":
                raise OSError("timed out")
            return (200, {"hearthwork": "x", "model": "m", "context": 1000}) if path.endswith("info") else (200, {})
        config = {"remote": {"host": "192.0.2.1", "port": 8484, "key": "k", "name": "n",
                             "addresses": ["192.0.2.1", "100.64.0.5"]}}
        with mock.patch.object(remote, "call", fake_call):
            self.assertEqual(remote.session(config), ("m", 1000))
        self.assertEqual(calls[0], ("192.0.2.1", remote.LAN_TIMEOUT))
        self.assertEqual(calls[1][1], remote.TAILSCALE_TIMEOUT)
        self.assertEqual(config["remote"]["host"], "100.64.0.5")
        self.assertEqual(config["remote"]["via"], "Tailscale 100.64.0.5")

    def test_all_unreachable(self):
        config = {"remote": {"host": "192.0.2.1", "port": 8484, "key": "k", "addresses": ["192.0.2.1", "100.64.0.5"]}}
        with mock.patch.object(remote, "call", side_effect=OSError("nope")):
            with self.assertRaises(remote.RemoteError) as caught:
                remote.session(config)
        self.assertIn("100.64.0.5", str(caught.exception))

    def test_discover_tailscale_lists_online_hearthwork_peers(self):
        state = share.parse_tailscale_status(STATUS)

        def fake_call(host, port, path, **kw):
            return (200, {"hearthwork": "0.5", "host": "LAPTOP", "model": "Qwen"})
        with mock.patch.object(remote, "tailscale_info", return_value=state), mock.patch.object(remote, "call", fake_call):
            hosts = remote.discover_tailscale()
        self.assertEqual([(h["host"], h["tailscale"]) for h in hosts], [("100.64.0.5", True)])  # the offline phone is skipped


if __name__ == "__main__":
    unittest.main()


class TailscaleOnlyDiscoveryTest(unittest.TestCase):
    def test_reply_round_trip(self):
        from hearthwork import share
        raw = share.discovery_reply("PC", 8484, "m", lan=False, tailscale=["100.1.2.3"])
        host = share.parse_discovery(raw, "192.168.0.7")
        self.assertFalse(host["lan"])
        self.assertEqual(host["tailscaleIps"], ["100.1.2.3"])
        self.assertTrue(share.parse_discovery(share.discovery_reply("PC", 8484, "m"), "192.168.0.7")["lan"])

    def test_lan_refusing_host_listed_by_tailscale_address(self):
        from unittest import mock
        from hearthwork import remote
        lan = [{"host": "192.168.0.7", "port": 8484, "hostname": "PC", "model": "m", "version": "x",
                "lan": False, "tailscaleIps": ["100.1.2.3"]}]
        with mock.patch.object(remote, "discover", return_value=lan), \
             mock.patch.object(remote, "discover_tailscale", return_value=[]):
            hosts = remote.find_hosts()
        self.assertEqual([(h["host"], h.get("tailscale")) for h in hosts], [("100.1.2.3", True)])
