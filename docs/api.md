# API Reference

The lightweight package provides frozen-dataset validation, membership verification,
answer parsing, ground-truth metrics, finalized judge-prompt provenance, and model
metadata. The optional `campaign-reproduction` dependency group enables the portable
deterministic Qwen inference runner; judge execution is not included.

The optional `telemetry` group enables inference-only footprint measurement.
`evaluation_has_a_footprint.telemetry` keeps NVML as the primary GPU-attributed
energy source and CodeCarbon as a secondary broader-scope cross-check. It does
not select a carbon location automatically: callers choose the explicit accepted
campaign profile or provide a carbon intensity. Water outputs are labeled
**ESTIMATED** and do not treat CodeCarbon WUE/water values of zero as physical
zero.
