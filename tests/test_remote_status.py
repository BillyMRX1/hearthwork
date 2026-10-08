import unittest
from unittest import mock

from hearthwork import harnesses, remote, share

REMOTE = {"host": "192.168.0.7", "port": 8002, "key": "k", "name": "laptop", "hostName": "MRX-ALLY",
          "addresses": ["192.168.0.7", "100.64.0.9"]}


class StatusLine(unittest.TestCase):
    def test_local(self):
        self.assertEqual(harnesses.statusline_text(["Qwen", "128K"]), "local · Qwen · 128K")

    def test_remote_host_and_route(self):
        self.assertEqual(harnesses.statusline_text(["Qwen", "128K", "MRX-ALLY", "LAN"]), "MRX-ALLY (LAN) · Qwen · 128K")
        self.assertEqual(harnesses.statusline_text(["Qwen", "128K", "MRX-ALLY"]), "MRX-ALLY · Qwen · 128K")

    def test_command_carries_host(self):
        command = harnesses.statusline_command("Qwen", 131072, "MRX-ALLY", "LAN")
        self.assertTrue(command.endswith('"Qwen" 128K "MRX-ALLY" "LAN"'))
        self.assertTrue(harnesses.statusline_command("Qwen", 131072).endswith('"Qwen" 128K'))

    def test_claude_settings_use_remote(self):
        with mock.patch.object(harnesses, "_remote", dict(REMOTE, via="Tailscale 100.64.0.9")):
            self.assertEqual(harnesses.host_parts(harnesses._remote), ("MRX-ALLY", "Tailscale"))
        self.assertEqual(harnesses.host_parts(None), ("", ""))


class StatusBlock(unittest.TestCase):
    def test_block_uses_route_actually_used(self):
        config = {"remote": dict(REMOTE, host="100.64.0.9", via="Tailscale 100.64.0.9", slots=2)}
        lines = remote.status_lines(config, ("Qwen", 131072))
        self.assertEqual(lines[0], "connected to: MRX-ALLY as laptop")
        self.assertIn("100.64.0.9:8002 (via Tailscale)", lines[1])
        self.assertIn("other addresses: 192.168.0.7", lines[2])
        self.assertIn("128K, 2 parallel sessions", lines[3])

    def test_unreachable(self):
        lines = remote.status_lines({"remote": dict(REMOTE)}, None)
        self.assertIn("not reachable", lines[1])
        self.assertEqual(lines[-1], "model: unavailable")


class Slots(unittest.TestCase):
    def test_session_keeps_slots(self):
        config = {"remote": dict(REMOTE)}
        info = {"hearthwork": "1", "model": "Qwen", "context": 131072, "slots": 4}
        with mock.patch.object(remote, "resolve", return_value=("192.168.0.7", info, 200)):
            self.assertEqual(remote.session(config), ("Qwen", 131072))
        self.assertEqual(remote.remote_slots(config), 4)
        self.assertEqual(remote.remote_slots({}), 2)

    def test_older_host_without_slots(self):
        config = {"remote": dict(REMOTE)}
        with mock.patch.object(remote, "resolve", return_value=("192.168.0.7", {"hearthwork": "1", "model": "Q"}, 200)):
            remote.session(config)
        self.assertEqual(remote.remote_slots(config, 3), 3)

    def test_info_reports_slots(self):
        host = share.Share({"server": {"port": 1, "slots": 3}}, 8002)
        with mock.patch.object(share, "served_model", return_value="Q"), mock.patch.object(share, "agent_context", return_value=1), \
                mock.patch.object(share, "host_addresses", return_value={}), mock.patch.object(share.urllib.request, "urlopen", side_effect=OSError):
            self.assertEqual(host.info()["slots"], 3)


if __name__ == "__main__":
    unittest.main()
