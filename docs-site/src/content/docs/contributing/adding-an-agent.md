---
title: Adding an agent
description: How a new coding agent is added to Hearthwork.
---

Adding an agent means adding one entry to `HARNESSES` in `src/hearthwork/harnesses.py`. The menu, `hearthwork agents`, `task`, MCP and `bench` pick it up from there.

An entry describes the agent: its title, the binary to run, how to get its version, which API it speaks, how to install it, and how Hearthwork connects it to the local endpoint.

Before proposing an agent, check that there is a clean, non-invasive way to point it at a custom endpoint, one that does not edit the user's own configuration. Hearthwork uses environment variables, command-line flags or a generated file in its own data folder. Agents without such a way are left out rather than guessed.

Add tests for the new agent in `tests/test_agents.py`, and run [`hearthwork doctor`](/troubleshooting/doctor/) against it with a real model.
