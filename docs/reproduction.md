# Reproduction

For the full current release procedure, including metadata-only frozen-set
manifests, see the repository's
[REPRODUCIBILITY.md](https://github.com/VectorInstitute/Evaluation-Has-a-Footprint/blob/main/REPRODUCIBILITY.md).

The portable runner reproduces the finalized Qwen evaluation protocol and
includes Gemma-4-12B-it support from caller-prepared BBQ and BBQ-V data. It
never downloads benchmark data, schedules jobs, or configures infrastructure.
Gemma requires a separate compatible Transformers 5.x environment; the
canonical requirement and fail-closed behavior are documented in
[REPRODUCIBILITY.md](https://github.com/VectorInstitute/Evaluation-Has-a-Footprint/blob/main/REPRODUCIBILITY.md#5-gemma-4-environment).

Install lightweight validation dependencies with `uv sync`.
For inference, install `uv sync --group campaign-reproduction`. Prepared data
must be obtained from their original upstream sources and validated against
the metadata-only full frozen manifests in `data/manifests/` before use.

```bash
python scripts/verify_frozen_manifests.py \
  --bbq-sample /path/to/bbq/sample.csv \
  --bbq-v-sample /path/to/bbq_v/sample.csv
```

The exact M4/M5 membership manifests are also in `data/manifests/`.
`evaluation_has_a_footprint.subsets` validates them against separately prepared
upstream data. M4 repeated memberships overlap and are descriptive rather than
independent sampling uncertainty.

Telemetry is optional: install it with `uv sync --group telemetry` in addition
to inference dependencies. `--telemetry` measures prediction generation only.
`--carbon-profile accepted-campaign` records the 59.0 gCO2e/kWh
Ontario/ECCC, location-unconfirmed campaign scenario; for another setting,
provide `--carbon-intensity-g-per-kwh`. Water is always an estimated
1.8–4.0 L/kWh range, with CodeCarbon total energy as the primary basis and
NVML GPU energy as a secondary basis.

The public parser reproduces the finalized general Qwen parsing and deterministic
maintenance-recovery methodology. Two historical Qwen2.5 BBQ-V INT4 records
required artifact-specific recorded policy resolutions; those decisions are not
generalized here. This release is therefore not a byte-for-byte recreation
mechanism for every historical accepted prediction artifact.
