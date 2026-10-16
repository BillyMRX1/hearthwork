---
title: Short commands (aliases)
description: Create short commands for your agents, and which other agents are not supported yet.
---

Create a short command for an agent:

```sh
hearthwork alias add ccl claude     # `ccl` now runs `hearthwork claude`
hearthwork alias list
hearthwork alias remove ccl
```

A launcher (`ccl.cmd` on Windows, a shell script elsewhere) is created next to the `hearthwork` command. Names of existing commands are refused, and only launchers Hearthwork created can be removed.

## See which agents are available

`hearthwork agents` lists which agents are installed (with version), the API each uses, how Hearthwork connects it, and the install command of the missing ones.

## Agents that are not supported yet

Goose, Crush, Kilo Code CLI, OpenHands and Hermes. A clean, non-invasive way to point each at a custom endpoint was not verified, so they are left out rather than guessed. For tools without a launcher, [API mode](/guides/api-mode/) gives setup snippets (`hearthwork api --show`).

To add an agent to Hearthwork itself, see [Adding an agent](/contributing/adding-an-agent/).
