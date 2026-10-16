---
title: This website
description: How the Hearthwork website is built, run locally, deployed to GitHub Pages and moved to a custom domain.
---

The website is an [Astro](https://astro.build) + [Starlight](https://starlight.astro.build) project in `docs-site/`. Pages are Markdown files in `docs-site/src/content/docs/`.

## Run it locally

```sh
cd docs-site
npm install
npm run dev        # http://localhost:4321/hearthwork/
npm run build      # static site in docs-site/dist
npm run preview    # serve the built site
```

## The generated CLI reference

`docs-site/src/content/docs/reference/cli.md` is generated from `hearthwork <command> --help`. Do not edit it by hand.

```sh
python docs-site/scripts/gen_cli_reference.py          # regenerate
python docs-site/scripts/gen_cli_reference.py --check  # exit 1 if it is out of date
```

The deploy workflow runs `--check`, so a change to a command's help that is not regenerated fails CI.

## Deploying

`.github/workflows/website.yml` builds and deploys to GitHub Pages on pushes to `main` that touch `docs-site/`, `src/hearthwork/` or the README. One-time step: in the repository on GitHub, go to Settings > Pages and set Source to **GitHub Actions**.

## Custom domain

The site builds for `https://billymrx1.github.io/hearthwork/` by default. To use a custom domain, see `docs-site/README.md` in the repository.
