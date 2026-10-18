"""`hearthwork update`: a running host locks the files on Windows, so it is stopped first and the model restarted
afterwards; a failed halfway upgrade checks importability and prints recovery steps. All mocks: nothing upgrades
for real and no host is contacted."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hearthwork import cli, host


class UpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "m.gguf"
        self.model.write_text("x")
        self.config = {"lastModel": str(self.model), "server": {"port": 8001}}
        self.events = []

    def run_update(self, status, stop_result=True, code=0, raises=None, importable=True):
        out = io.StringIO()
        call = mock.Mock(side_effect=raises or (lambda c: self.events.append("upgrade") or code))
        with contextlib.redirect_stdout(out), \
                mock.patch.object(cli, "_upgrade_cmd", return_value=["uv", "tool", "upgrade", "hearthwork"]), \
                mock.patch.object(cli, "load_config", return_value=self.config), \
                mock.patch.object(cli, "_importable", return_value=importable), \
                mock.patch.object(cli.subprocess, "call", call), \
                mock.patch.object(host, "status", side_effect=lambda: self.events.append("check") or status), \
                mock.patch.object(host, "stop", side_effect=lambda c, **kw: self.events.append("stop") or stop_result), \
                mock.patch.object(host, "ensure", side_effect=lambda c, m=None, **kw: self.events.append("start") or True):
            rc = cli.update()
        return rc, out.getvalue(), call

    def test_running_host_stops_upgrades_and_starts(self):
        rc, text, call = self.run_update({"pid": 5, "ready": True})
        self.assertEqual(rc, 0)
        self.assertEqual(self.events, ["check", "stop", "upgrade", "start"])
        call.assert_called_once_with(["uv", "tool", "upgrade", "hearthwork"])
        self.assertIn("Stopping the model to update (restarts after)", text)
        self.assertIn("Model restarted (m.gguf)", text)

    def test_idle_upgrades_straight_away(self):
        rc, text, call = self.run_update(None)
        self.assertEqual(rc, 0)
        self.assertEqual(self.events, ["check", "upgrade"])  # no stop, no start
        self.assertEqual(text.strip(), "Update done.")

    def test_upgrade_raises_is_reported(self):
        rc, text, call = self.run_update(None, raises=OSError("uv vanished"))
        self.assertEqual(rc, 1)
        self.assertIn("the updater did not run", text)
        self.assertIn("Hearthwork still works", text)

    def test_broken_install_prints_recovery(self):
        rc, text, call = self.run_update({"pid": 5}, code=2, importable=False)
        self.assertEqual(rc, 1)
        self.assertNotIn("start", self.events)  # a broken install cannot even start the host
        self.assertIn("can no longer be imported", text)
        self.assertIn("uv tool install --force hearthwork", text)
        self.assertIn("taskkill", text)
        self.assertIn("hearthwork start", text)

    def test_stop_refusal_does_not_upgrade(self):
        rc, text, call = self.run_update({"pid": 5}, stop_result=False)
        self.assertEqual(rc, 1)
        self.assertEqual(self.events, ["check", "stop"])
        call.assert_not_called()
        self.assertIn("requests are running", text)
        self.assertIn("still running", text)


if __name__ == "__main__":
    unittest.main()
