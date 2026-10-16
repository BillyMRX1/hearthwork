---
title: Benchmark
description: Measure your own setup with 8 graded coding tasks driven through an agent.
---

`hearthwork bench` (or the menu) drives the running model through a coding agent (`--agent`, Claude Code by default) in a fresh folder, with 8 prompts in one conversation. It grades each step by checking the files and running the code, not by trusting the agent's reply.

| # | Task | Graded by |
|---|---|---|
| 1 | chat | a reply came back |
| 2 | list files | both files named |
| 3 | read a file | the secret phrase is in the reply |
| 4 | write + run `fizzbuzz.py` | running it gives the exact FizzBuzz output |
| 5 | add `is_prime` + run | function and docstring exist, FizzBuzz still right, the primes below 30 are printed |
| 6 | fix 3 planted bugs | `buggy.py` prints `20.0`, `[9, 5]`, `0` |
| 7 | write unit tests | `test_buggy.py` passes (half points if it copies the functions instead of importing `buggy.py`) |
| 8 | summary | names all 3 files |

## Reading results

- **Results are per machine.** Speed and score depend on your hardware, model, quantization and context, so compare setups on your own computer. Results are added to `bench/results.jsonl`, and the scoreboard (`bench --show`) ranks every model and agent you have tried, by score and then time.
- **Each run's folder** stays in `bench/runs/<date-time>/<agent>/`, so you can inspect what the agent wrote.
- **Isolated history.** Each run gives Codex and Claude Code an empty config folder (`CODEX_HOME`, `CLAUDE_CONFIG_DIR`), so benchmark sessions never fill your real `~/.codex` or `~/.claude`. The run refuses to start if that folder would be your real one. No login is needed.

## Release check

`hearthwork bench --all` runs every installed agent one after another (never in parallel) and prints and saves a summary table (`summary.md` in the run's folder; `bench --show` prints the latest).

- A warmup (about 30K-token prompts through the Anthropic and OpenAI endpoints) runs first and is not timed. `--warmup` does the same before a single-agent run.
- If the agent itself fails (it crashes, does not start, or reaches another provider), the run is repeated once in a fresh folder. A failed attempt has an `error` and no score.
- A wrong answer is a score, not a failure, and is not retried.
