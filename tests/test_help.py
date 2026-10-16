"""`hearthwork <command> --help` prints usage and exits 0 for every command in the dispatch of cli.main.

The command names are read from cli.py itself, so a new command is covered without touching this file.
Nothing here starts the model, opens a socket or changes the config: --help returns before any of that.
"""
import contextlib
import io
import re
import unittest
from pathlib import Path

from hearthwork import cli
from hearthwork.harnesses import HARNESSES

SOURCE = Path(cli.__file__).read_text(encoding="utf-8")
# Agent commands (claude, codex, ...) pass every following argument to the agent itself, so `claude --help` is the
# agent's help. statusline is hidden and has no help: it prints the status line Claude Code shows.
NOT_HELP_COMMANDS = set(HARNESSES) | {"statusline"}


def dispatch_commands():
    names = set(re.findall(r'command == "([^"]+)"', SOURCE))
    for group in re.findall(r'command in \(([^)]*)\)', SOURCE):
        names.update(re.findall(r'"([^"]+)"', group))
    return sorted(name for name in names
                  if name[:1].isalpha() and name not in ("help", "version") and name not in NOT_HELP_COMMANDS)


class HelpTests(unittest.TestCase):
    def test_every_dispatched_command_prints_help_and_exits_zero(self):
        commands = dispatch_commands()
        self.assertIn("budget", commands)
        self.assertIn("host", commands)
        for command in commands:
            with self.subTest(command=command):
                out = io.StringIO()
                code = 0
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        cli.main([command, "--help"])
                    except SystemExit as error:
                        code = error.code
                self.assertFalse(code, f"exit code {code!r}")
                self.assertTrue(out.getvalue().strip(), "no help printed")


if __name__ == "__main__":
    unittest.main()
