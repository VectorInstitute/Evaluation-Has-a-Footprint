# Project website

Static project page for *Sustainable Responsible-AI Evaluation: When Compute
Savings Change Benchmark Conclusions*.

The presentation layer is based on the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
adapted from the Nerfies project page.

This directory is self-contained: `index.html` plus `static/` assets. GitHub
Pages publishes these files from the `gh-pages` branch (folder `/`).

## Local preview

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Deployment

On push to `main`, `.github/workflows/docs.yml` copies this folder to the
`gh-pages` branch. Keep GitHub Pages set to **Deploy from a branch**, branch
`gh-pages`, folder `/ (root)`.

The current page still includes `noindex, nofollow` metadata and should not
be treated as a public release until that is removed.
