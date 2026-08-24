# Curated aggregate results

This directory is the public, aggregate-only release supporting the finalized
76-run Qwen campaign in *Evaluation Has a Footprint*. It covers
Qwen2.5-VL-7B-Instruct and Qwen3-VL-30B-A3B-Instruct on the frozen BBQ and
BBQ-V benchmarks: four model--dataset blocks with 19 accepted configurations
each. It does not contain raw predictions, generated reasoning, judge outputs,
benchmark rows, answer labels, or images.

## Campaign and metrics

The configurations are M0 (full benchmark, BF16), M1 (larger batching), M2
(INT8), M3 (INT4), M4 (smaller complete-natural-unit subsets), M5a (the M4
50% subset with larger batching), and M5b (the M4 50% subset with INT8).

- **Accuracy** is a percentage; higher is better.
- **Bias Score** is a 1--100 score; higher is worse. It is assessed only when
  reasoning can be evaluated for bias.
- **Bias Present Rate** is the proportion of assessable Bias Score judgments
  whose score is greater than 1; lower is better.
- **Reasoning Quality Score** is a 1--100 score; higher is better.
- Runtime, NVML GPU-attributed energy (kWh), operational CO2e, and estimated
  water are reported as aggregate footprint measures.

Bias Score and Reasoning Quality Score are scores, not percentages.

`aggregates/final_76_campaign.csv` is the machine-readable aggregate matrix.
It retains scientific configuration, metric, and footprint columns only.
`aggregates/final_76_campaign_provenance.json` describes its scope and model
revisions. `aggregates/SHA256SUMS` records SHA-256 digests for every released
aggregate file.

## Environmental footprint

NVML cumulative energy is the primary GPU-attributed energy measurement.
CodeCarbon is a broader tracked-process energy cross-check. Operational CO2e
uses the campaign reproduction profile of 59.0 gCO2e/kWh (Ontario/ECCC;
location-unconfirmed). Water is **ESTIMATED**, not measured: the primary range
uses CodeCarbon tracked total energy and 1.8--4.0 L/kWh, while the secondary
GPU-attributed range uses NVML GPU energy and the same factor.

## Subsampling and natural units

M4 retains complete natural units rather than arbitrary rows: complete cases
for BBQ and complete visual scenarios for BBQ-V. The exact frozen memberships
are available under `data/manifests/`, together with the deterministic public
generator. The frozen M4 replicates overlap substantially and are therefore
non-independent.

`aggregates/monte_carlo_*.csv` and
`aggregates/monte_carlo_provenance.json` report the primary sampling-uncertainty
analysis: 1,000 independent stratified draws over frozen M0 outputs, not 1,000
new inference runs. `aggregates/frozen_vs_monte_carlo.csv` documents why the
overlapping frozen M4 repeat ranges and SDs must not be interpreted as
independent sampling uncertainty. `aggregates/natural_unit_structure.csv`
summarizes the natural-unit structure without publishing benchmark items.

## Intervention and group-wise aggregates

`aggregates/cluster_bootstrap_interventions.csv` and
`aggregates/intervention_deltas.csv` contain the final intervention comparisons
and natural-unit cluster-bootstrap summaries. The released INT4 group-wise
files are all M3 (INT4) minus M0 (BF16):

- `int4_bbq_context_polarity.csv` is the main BBQ context/ambiguity analysis.
- `int4_category_deltas.csv` is an exploratory native-category scan.
- `int4_demographic_group_deltas.csv` is exploratory, restricted to groups
  with at least 50 examples; demographic memberships can overlap and no
  multiple-comparison correction was applied.

The corresponding reviewed figures are under `figures/`. Their values derive
from the final aggregate analyses; no figure publishes item-level content.

## Release scope and limitations

This is an aggregate-results release, not an internal run archive. It excludes
raw predictions/reasoning, raw judge outputs, benchmark content/images, model
weights, scheduler metadata, and operational logs. Gemma results are not
included because this repository does not yet contain reproducible Gemma
provenance and implementation.
