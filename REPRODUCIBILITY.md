# Reproducibility

This repository reproduces the frozen evaluation membership and portable
evaluation profiles without redistributing BBQ or BBQ-V. The paper's scientific
results cover Qwen2.5-VL-7B, Qwen3-VL-30B-A3B, and Gemma-4-12B-it. Gemma
support requires a separate compatible Transformers 5.x environment; see
[Gemma environment](#5-gemma-4-environment).

## 1. Obtain the original benchmarks

- **BBQ:** clone or download the [NYU BBQ repository](https://github.com/nyu-mll/BBQ).
  The frozen selection was prepared from commit
  `bea11bd97d79217245b5871acd247b9d6eb24598`.
- **BBQ-V:** obtain the original benchmark from the
  [UCF-CRCV BBQ-Vision project](https://github.com/UCF-CRCV/BBQ-Vision) under
  its terms. The preparation records use repository commit
  `16408994a0607d673c18ad6331a144fed741f9dc` and the pinned visual-release
  revision `a1c78b8f73bc40408993414e3d94714a6a9169d3`.

Do not treat this repository as a data source. The original datasets, images,
and annotations remain subject to their upstream licenses and access terms.

## 2. Reconstruct the full frozen membership

Use the metadata-only files in [`data/manifests/`](data/manifests/):

- `bbq_frozen_ids.csv` selects 2,000 rows forming 500 complete BBQ cases.
  Rows are identified by `category` and official `example_id`; a case is
  `category:case_id` and must retain its four `(context_condition,
  question_polarity)` rows.
- `bbq_v_frozen_ids.csv` selects 1,998 rows forming 389 complete BBQ-V
  scenarios. Rows are identified by `id`, `cross_id`, and
  `unique_question_id`; a scenario is a complete `unique_question_id` unit.
  `image_sha256` is verification metadata only, not an image payload.

The CSV row order is the accepted frozen order. After preparing `sample.csv`
files with the expected columns, validate them:

```bash
python scripts/verify_frozen_manifests.py \
  --bbq-sample /path/to/bbq/sample.csv \
  --bbq-v-sample /path/to/bbq_v/sample.csv
```

The verifier uses only the prepared CSVs and metadata-only manifests. It
checks the frozen row order, source identifiers, complete natural units,
counts, and expected BBQ-V image-hash cardinality; it does not download data
or access an image payload.

## 3. Use the fixed M4 memberships

The full-set manifests above identify the M0 source evaluation sets. M4 and
M5 use the accepted nested reduced-set memberships in:

```text
data/manifests/bbq_m4_subsets.json
data/manifests/bbq_v_m4_subsets.json
```

These JSON files contain only selected natural-unit IDs and provenance. Their
source-sample fingerprints must match the prepared full-set sample before use.
The repeated 50% and smallest M4 memberships overlap substantially, so they
are descriptive repeated memberships rather than independent uncertainty
draws.

## 4. Run the portable Qwen profiles

Install validation dependencies:

```bash
uv sync
```

For GPU inference, add the optional profile:

```bash
uv sync --group campaign-reproduction
```

Run `python -m eval_efficiency.cli --help` for the public command
interface. A run requires an explicit model, dataset, condition,
`--prepared-dataset`, and output directory. M4/M5 conditions additionally
require the corresponding explicit `--membership` JSON, target subset rows,
and replicate.

The accepted campaign used Python 3.11.15, PyTorch 2.11.0+cu130,
Transformers 4.57.6, Accelerate 1.14.0, bitsandbytes 0.50.1,
qwen-vl-utils 0.0.14, torchvision 0.26.0+cu130, and Pillow 12.3.0. Hardware,
CUDA builds, and upstream model access must be supplied by the reproducer.

Telemetry is optional. With the telemetry group installed, `--telemetry`
measures prediction generation only. The explicit
`--carbon-profile accepted-campaign` records the campaign's 59.0 gCO2e/kWh
Ontario/ECCC, location-unconfirmed scenario; it is not a universal emissions
factor.

## 5. Gemma 4 environment

The accepted Qwen runs use the existing `transformers==4.57.6` environment
declared by the `campaign-reproduction` dependency group. The accepted
Gemma-4-12B-it runs require a Transformers 5.x build exposing
`Gemma4UnifiedForConditionalGeneration`; the canonical runs used
`transformers==5.14.1`. The public PyPI release
`transformers==5.14.1` includes that class, but numerical equivalence with the
canonical cluster build has not been verified here. Do not assume the
canonical cluster build is installable from PyPI. Configure a compatible
Transformers 5.x environment and confirm the class is available before running
Gemma; the public runtime intentionally fails closed when it is unavailable.

## What this does not reproduce

This release does not redistribute benchmark content, model weights, raw
predictions, judge archives, or the internal run archive. Two historical
Qwen2.5 BBQ-V INT4 records required artifact-specific recorded policy
resolutions; those decisions are not generalized here. It also does not
claim byte-for-byte recreation of every historical accepted prediction artifact
or package a platform-independent Gemma environment. The paper is the authority
for the full three-model scientific claims.
