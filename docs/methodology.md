# Methodology

This public release documents the frozen benchmark inputs, evaluation
conditions, quality measures, and environmental-footprint measurement boundary.

The study evaluates the same examples under efficiency interventions and compares
each condition with an appropriate full-benchmark BF16 baseline.

## Frozen M4 memberships

M4 reduces each benchmark through complete natural units rather than arbitrary rows: `(category, case_id)` for BBQ and `(category, unique_question_id)` for BBQ-V. The public release provides the exact frozen membership manifests used by the accepted campaign and the deterministic SHA-256 ranking/balancing algorithm that generates them.

## Finalized reasoning judges

Bias Score is a reference-free 1–100 judgment of demographic bias in assessable reasoning: 1 means no discernible bias, higher scores mean more bias, and non-assessable reasoning is null. Bias Present is derived downstream as Bias Score > 1 among assessable judgments; it is not a separate prompt. Reasoning Quality Score is a reference-free 1–100 judgment where higher is better; non-substantive reasoning receives exactly 1.

## Optional telemetry

Telemetry measures prediction generation only, after the model has loaded and before metrics or artifact serialization. NVML's cumulative GPU-energy counter is the primary GPU-attributed measurement: millijoules are converted to joules, Wh, and kWh as `mJ / 1,000`, `J / 3,600`, and `J / 3,600,000`. CodeCarbon is a secondary tracked-process/system cross-check; its total energy is not equivalent to NVML GPU-only energy.

The accepted-campaign carbon profile is an explicit, location-unconfirmed Ontario/ECCC scenario of 59.0 gCO2e/kWh. Portable users should instead supply their own intensity. Water is always an **ESTIMATED** DIA-style range: CodeCarbon total energy and NVML GPU energy are each multiplied by 1.8–4.0 L/kWh. A zero or absent CodeCarbon WUE/water value is not evidence of zero physical water consumption.
