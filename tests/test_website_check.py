"""The website's CLI reference must stay in step with the code: docs-site/scripts/gen_cli_reference.py --check
regenerates the help text in memory and exits 1 when src/content/docs/reference/cli.md differs. Running it in the
normal suite means a changed `--help` without a regenerated page fails the tests. Skipped when the docs site is not
part of the checkout (e.g. a pip-installed source tree without docs)."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs-site" / "scripts" / "gen_cli_reference.py"


@unittest.skipIf(not SCRIPT.is_file(), "docs-site/scripts/gen_cli_reference.py is not in this checkout")
class WebsiteCheck(unittest.TestCase):
    def test_cli_reference_is_current(self):
        cmd = ["py", "-3", str(SCRIPT), "--check"] if os.name == "nt" else [sys.executable, str(SCRIPT), "--check"]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0,
                         "the CLI reference is stale or --check failed (run: python docs-site/scripts/"
                         f"gen_cli_reference.py):\n{result.stdout}{result.stderr}")


if __name__ == "__main__":
    unittest.main()
