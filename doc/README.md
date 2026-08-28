# Project website

Static project page for *Sustainable Responsible-AI Evaluation: When Compute
Savings Change Benchmark Conclusions*.

The presentation layer is based on the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
adapted from the Nerfies project page.

This directory is self-contained: `index.html` plus `static/` assets. It is
separate from the MkDocs sources in `docs/`.

## Local preview

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Deployment

Vercel (or any static host) can serve this folder with no build step. Point
the site root at `doc/`.

The current page still includes `noindex, nofollow` metadata and should not
be treated as a public release until that is removed.
