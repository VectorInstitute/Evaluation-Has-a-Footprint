# Sustainable Responsible-AI Evaluation: When Compute Savings Change Benchmark Conclusions

Research code, frozen-membership metadata, and aggregate results for the paper
*Sustainable Responsible-AI Evaluation: When Compute Savings Change Benchmark
Conclusions*.

Responsible-AI evaluation has a computational footprint, yet efficiency is
rarely assessed alongside the conclusions that benchmarks are meant to support.
We evaluate Qwen2.5-VL-7B, Qwen3-VL-30B-A3B, and Gemma-4-12B-it on BBQ and
BBQ-V under seven conditions spanning batching, quantization, benchmark
reduction, and their combinations.

## Main findings

- Larger batching changes accuracy by no more than 0.35 percentage points
  across the six reported model--dataset settings, while measured energy is
  lower in five of six settings.
- INT8 preserves average quality in many settings but uses 79--326% more GPU
  energy than BF16 in this campaign; INT4 produces the largest model- and
  context-dependent quality changes.
- The smallest reduced subsets lower measured GPU energy by about 87% across
  all six settings, while very small subsets are less stable. M5a (50% subset
  plus larger batch) generally retains near-baseline accuracy with substantial
  runtime and energy savings.

These are campaign results, not universal claims. In particular, runtime is
not a proxy for measured GPU energy, and the repeated M4 memberships overlap;
their spread is descriptive rather than independent sampling uncertainty.

## Links

- **Project website source:** [goushaa/Evaluation-Has-a-Footprint-Website](https://github.com/goushaa/Evaluation-Has-a-Footprint-Website)
- **Code:** [VectorInstitute/Evaluation-Has-a-Footprint](https://github.com/VectorInstitute/Evaluation-Has-a-Footprint)
- **BBQ (original benchmark):** [NYU BBQ repository](https://github.com/nyu-mll/BBQ)
- **BBQ-V (original benchmark):** [UCF-CRCV BBQ-Vision repository](https://github.com/UCF-CRCV/BBQ-Vision)

No public paper or arXiv URL is currently recorded in the source materials.

## Datasets and frozen evaluation sets

BBQ and BBQ-V are third-party benchmarks created by their original authors.
This repository does **not** redistribute their questions, answer choices,
labels, annotations, or images. It does not host a dataset download.

Instead, [`data/manifests/`](data/manifests/) contains metadata-only frozen
evaluation manifests. They identify the exact source records used in this
paper without carrying any benchmark payload. Obtain the original datasets
from their official sources under their original terms, then use the manifests
and [reproduction guide](REPRODUCIBILITY.md) to reconstruct and verify the
same evaluation membership.

The full frozen evaluation sets contain:

| Benchmark | Frozen rows | Natural units | Membership key |
| --- | ---: | ---: | --- |
| BBQ | 2,000 | 500 complete cases | `(category, case_id)` |
| BBQ-V | 1,998 | 389 complete scenarios | `unique_question_id` |

The exact M4 reduced-subset memberships used by the campaign remain available
as `data/manifests/bbq_m4_subsets.json` and
`data/manifests/bbq_v_m4_subsets.json`.

## Reproduction overview

1. Obtain BBQ and BBQ-V from their official upstream projects and comply with
   their respective terms and licenses.
2. Prepare the frozen records using the metadata-only manifests in
   [`data/manifests/`](data/manifests/); do not substitute or augment records.
3. Validate the prepared membership:

   ```bash
   python scripts/verify_frozen_manifests.py \
     --bbq-sample /path/to/bbq/sample.csv \
     --bbq-v-sample /path/to/bbq_v/sample.csv
   ```

4. Install the optional evaluation dependencies and run the portable Qwen
   campaign runner as documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository structure

```text
data/manifests/       Metadata-only full-set and M4 membership manifests
docs/                 Public methodology, data-license, and API documentation
results/              Curated aggregate-only Qwen campaign release
scripts/              Lightweight public verification utilities
src/                  Portable evaluation package
REPRODUCIBILITY.md    Frozen-set reconstruction and validation guide
```

## License and provenance

The code in this repository is released under its own license. That license
does not grant rights to BBQ, BBQ-V, model weights, or any upstream material.
See [docs/data-and-licenses.md](docs/data-and-licenses.md) and the upstream
projects for applicable dataset and model terms.

Citation metadata will be added when public paper metadata are finalized.
