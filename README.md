<div align="center">

# Evaluation Has a Footprint

**Sustainable responsible-AI evaluation: when compute savings change benchmark conclusions**

[Project page](https://evaluation-has-a-footprint-website.vercel.app/) · **Paper:** arXiv coming soon · **Documentation:** coming soon

</div>

Responsible-AI evaluation is repeated throughout model development, yet its computational footprint is rarely assessed alongside the conclusions benchmarks are designed to support. This project studies whether batching, quantization, and benchmark reduction can lower evaluation cost without materially changing accuracy, measured bias, or reasoning quality.

- **Benchmarks:** BBQ (text) and BBQ-V (image–text).
- **Models:** Qwen2.5-VL-7B, Qwen3-VL-30B-A3B, and Gemma 4 12B.
- **Interventions:** larger batches, INT8, INT4/NF4, reduced benchmark subsets, and selected combinations.
- **Outcomes:** accuracy, bias, reasoning quality, runtime, GPU energy, estimated carbon, and estimated water use.

The results show that efficiency interventions are not interchangeable. Larger batching generally preserves evaluation quality and often lowers energy. INT8 can increase runtime and energy in the tested stack, while INT4 produces larger model- and context-dependent quality changes. Reduced benchmarks provide the most consistent energy savings, although the smallest subsets are more sensitive to which examples are retained.

## Repository contents

| Path | Description |
| --- | --- |
| [`src/evaluation_has_a_footprint/`](src/evaluation_has_a_footprint/) | Portable evaluation runner, conditions, metrics, parsing, and telemetry |
| [`data/manifests/`](data/manifests/) | Frozen M4 subset memberships and provenance metadata |
| [`results/figures/`](results/figures/) | Curated figures from the study |
| [`docs/methodology.md`](docs/methodology.md) | Metrics and footprint-measurement boundaries |
| [`docs/reproduction.md`](docs/reproduction.md) | Prepared-data contract and campaign reproduction guidance |
| [`docs/data-and-licenses.md`](docs/data-and-licenses.md) | Upstream datasets, model revisions, and licenses |

This public release does **not** redistribute benchmark examples, images, model weights, predictions, judge outputs, or run-level result tables. Obtain upstream assets separately and follow their respective licenses.

## Quick start

Install the development environment and run the validation suite:

```bash
uv sync
uv run pytest
uv run pre-commit run --all-files
```

For GPU inference and optional telemetry:

```bash
uv sync --group campaign-reproduction --group telemetry

uv run python -m evaluation_has_a_footprint.cli \
  --model qwen25_vl_7b \
  --dataset bbq \
  --condition M0 \
  --prepared-dataset /path/to/prepared/bbq \
  --output /path/to/output \
  --telemetry \
  --carbon-intensity-g-per-kwh YOUR_LOCAL_INTENSITY
```

M4 and M5 conditions additionally require the matching frozen membership manifest. See the [reproduction guide](docs/reproduction.md) for supported configurations, data preparation, measurement boundaries, and the Gemma-specific environment.

## Paper

- **Title:** *Sustainable Responsible-AI Evaluation: When Compute Savings Change Benchmark Conclusions*
- **Authors:** _To be added_
- **arXiv:** _Coming soon_
- **Project page:** [evaluation-has-a-footprint-website.vercel.app](https://evaluation-has-a-footprint-website.vercel.app/)

## Citation

_Replace the placeholders below when the paper is public._

```bibtex
@misc{evaluation_has_a_footprint_2026,
  title         = {Sustainable Responsible-AI Evaluation: When Compute Savings Change Benchmark Conclusions},
  author        = {TODO},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Acknowledgements

_To be added._

## Contact

- **Research questions:** _Name and email to be added_
- **Code and reproducibility:** open a [GitHub issue](https://github.com/VectorInstitute/Evaluation-Has-a-Footprint/issues)

## License and contributing

Code is released under the [Apache 2.0 License](LICENSE.md). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
