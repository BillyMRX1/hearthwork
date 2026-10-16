---
title: Release and trusted publishing
description: For maintainers. How releases reach PyPI from GitHub Actions with Trusted Publishing.
---

Releases go to PyPI from GitHub Actions via Trusted Publishing (`.github/workflows/publish.yml`). No API token is stored anywhere.

## One-time setup

- **On PyPI:** project `hearthwork` > Settings > Publishing > add a GitHub trusted publisher: owner `BillyMRX1`, repo `hearthwork`, workflow `publish.yml`, environment `pypi`.
- **On GitHub:** Settings > Environments > New environment `pypi`, with Required reviewers set to the maintainer.

## Each release

1. Bump `__version__` in `src/hearthwork/__init__.py`, commit, and push a tag matching it, for example `git tag v0.9.1 && git push origin v0.9.1`. The workflow fails if the tag and the version differ.
2. Approve the `pypi` deployment on the workflow run in GitHub Actions.

The workflow builds with `uv build`, checks the distributions with `twine check`, then publishes with `pypa/gh-action-pypi-publish`.

Manual publishing with `uv publish` still works.
