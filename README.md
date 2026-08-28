# Evaluation Has a Footprint

Research code.

This repository accompanies *Evaluation Has a Footprint*. It studies
evaluation-efficiency interventions across the frozen BBQ and BBQ-V benchmarks
while tracking quality and environmental footprint measures.

## Development

Set up the development environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the checks locally:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Build the documentation:

```bash
uv run mkdocs build
```

## Data, model weights, and results

Benchmark data, images, model weights, and experimental result tables are not
bundled with this repository. Prepared data must be obtained separately. The
runner covers the finalized Qwen campaign profiles and the Gemma 4 12B
extension. Frozen M4 memberships are under `data/manifests/`.

## Optional campaign reproduction

Install GPU inference dependencies with `uv sync --group campaign-reproduction`.
Prepared benchmark data and upstream model access are supplied explicitly;
default CI does not download models or run GPU inference. The exact M4
membership manifests are available under `data/manifests/`; telemetry remains
an optional dependency.
