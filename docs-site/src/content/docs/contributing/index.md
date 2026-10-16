---
title: Contributing
description: How to set up a development copy of Hearthwork and where the code lives.
---

Hearthwork is open source under the MIT license: [github.com/BillyMRX1/hearthwork](https://github.com/BillyMRX1/hearthwork).

## Get the code

```sh
git clone https://github.com/BillyMRX1/hearthwork
cd hearthwork
```

The package has no runtime dependencies. The code is in `src/hearthwork/`, the tests in `tests/`, and this website in `docs-site/`.

Run your working copy without installing it by setting `PYTHONPATH=src`:

```sh
PYTHONPATH=src python -m hearthwork --version
```

On Windows PowerShell, set `$env:PYTHONPATH = "src"` first.

## Pages

- [Tests](/contributing/tests/)
- [Adding an agent](/contributing/adding-an-agent/)
- [Release (maintainers)](/contributing/release/)
- [This website](/contributing/website/)

Open an issue before starting a large change, so the idea can be discussed first.
