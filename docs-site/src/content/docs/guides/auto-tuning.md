---
title: Auto-tuning
description: Try a few installed configurations and save the best tested one as a profile.
---

`hearthwork tune` tries a small set of the configurations already on your computer, measures each one with a short version of the [benchmark](/guides/benchmark/), and saves the best tested one as a [setup profile](/guides/setup-profiles/) you can inspect and apply. Nothing is downloaded, and your working setup is put back at the end.

```sh
hearthwork tune --dry-run                     # list the candidates and what they are estimated to need, then exit
hearthwork tune --time 30m                    # try them (default limit 30 minutes)
hearthwork tune --min-context 64K --agent claude --quick-tasks 3
```

## What it tries

The candidates are bounded (at most 5) and built from what is installed, in this order:

- the model that is running now, always first,
- the other installed models that fit your [budget](/guides/budgets/) at their automatic context, **ranked for this computer by the same `models.rank()` that [`hearthwork check`](/reference/commands/) uses**,
- the current model at the next smaller context step, and
- a smaller quantization of the same model family, when one is already on disk.

`--dry-run` lists them in that order, each with a one-line reason ("rank 1 for this computer: …"), its estimated memory, and switches nothing (it does not touch the running host). `--min-context` drops any candidate whose model cannot reach that context here.

## What it measures

For each candidate it switches through the host (so agent sessions must be closed first), warms the model up, then runs the first `--quick-tasks` graded tasks (default 3). It records:

- task score (the fraction of graded points),
- total seconds,
- first-token latency (median) and generation speed from the host's diagnostics,
- peak memory while the candidate was loaded.

Candidates are ranked by score first, then by time, then by first-token latency. If the best two are within 10% on score-seconds, the best is re-run once (when the time limit allows) and both runs are reported.

## The result

The report says explicitly that the winner is the **best among the configurations tested** — a specific machine, runtime and model set at a specific time, not a universal optimum. The run (candidates, conditions, per-candidate table, winner) goes to `tune.jsonl` in the [data folder](/reference/data-and-config/), rotated at 5 MB, and the winner is saved as the profile `tune-<date>` with the run as its evidence.

Applying it is your decision:

```sh
hearthwork profile show tune-2026-01-02
hearthwork profile apply tune-2026-01-02
```

The original model and context are always restored when the run ends, including on Ctrl+C or if a candidate fails, so a failed or interrupted run never leaves your setup changed.