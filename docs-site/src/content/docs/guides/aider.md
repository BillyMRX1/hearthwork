---
title: Aider
description: Run Aider on your local model with hearthwork aider, and what makes it different from the other agents.
---

| | |
|---|---|
| Command | `hearthwork aider` |
| Install | `uv tool install --python 3.12 aider-chat` |
| API used | OpenAI Chat Completions |

Run `hearthwork aider` from your project folder. Extra arguments go to Aider: `hearthwork aider --message "..."`.

## Install it with Python 3.12

`aider-chat` fails to build `scipy` on Python 3.13 and newer on Windows, so install it with `--python 3.12` as shown above.

## It is not an agent loop

Aider edits files from a reply format. It cannot read or run anything on its own, and it never calls tools. Because of that:

- In `hearthwork task`, shell commands are switched off for Aider.
- In [Doctor](/troubleshooting/doctor/), the tool stages do not count for Aider.
- Aider keeps its own git behaviour.

## Dry run

Hearthwork's `--dry-run` works when it is the first argument (`hearthwork aider --dry-run`). Aider has a `--dry-run` of its own, which still works after another argument.
