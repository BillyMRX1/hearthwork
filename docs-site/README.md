# Hearthwork website

Astro + Starlight site: landing page and documentation. Pages are Markdown in `src/content/docs/`.

## Run locally

```sh
cd docs-site
npm install
npm run dev        # http://localhost:4321/hearthwork/
npm run build      # static output in dist/ (also checks internal links)
npm run preview    # serve dist/
```

## CLI reference

`src/content/docs/reference/cli.md` is generated from `hearthwork <cmd> --help` (Python stdlib only):

```sh
python docs-site/scripts/gen_cli_reference.py          # from the repo root: regenerate
python docs-site/scripts/gen_cli_reference.py --check  # exit 1 if out of date (used in CI)
```

## Deploy (GitHub Pages)

`.github/workflows/website.yml` deploys on pushes to `main`. One-time: repo Settings > Pages > Source: **GitHub Actions**.

## Custom domain

The site URL and base path come from environment variables at build time:

| Variable | Default | Custom domain |
|---|---|---|
| `SITE_URL` | `https://billymrx1.github.io` | `https://hearthwork.dev` |
| `SITE_BASE` | `/hearthwork` | `/` |

To switch:

1. Add a file `docs-site/public/CNAME` containing only the domain, for example `hearthwork.dev`.
2. In the workflow's build step set `SITE_URL: https://hearthwork.dev` and `SITE_BASE: /` (the `env:` lines are already there, commented).
3. In DNS, point the domain at GitHub Pages (a `CNAME` record to `billymrx1.github.io` for a subdomain, or the `A`/`AAAA` records GitHub documents for an apex domain).
4. In the repo, Settings > Pages > Custom domain: enter the domain and tick Enforce HTTPS once the certificate is ready.

Test locally with `SITE_URL=https://hearthwork.dev SITE_BASE=/ npm run build` (in Git Bash on Windows prefix it with `MSYS_NO_PATHCONV=1`, which otherwise rewrites `/`).
