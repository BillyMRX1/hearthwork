---
title: Tests
description: Run the Hearthwork test suite.
---

The tests use Python's built-in `unittest`. From the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests -t .
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -t .
```

Run a single file or test with its module name, for example `python -m unittest tests.test_help`.

No GPU or model is needed. `tests/fake_llama.py` stands in for llama-server, so the tests exercise the host, relay, sharing and agents without a real model.

`tests/test_help.py` reads the commands from `cli.py` and checks that `hearthwork <command> --help` prints usage and exits 0 for each, without starting anything. A new command is covered automatically.
