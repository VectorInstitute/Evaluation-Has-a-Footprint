"""Integrity checks for the curated aggregate-results release."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGGREGATES = ROOT / "results" / "aggregates"
CHECKSUMS = AGGREGATES / "SHA256SUMS"
EXPECTED_ARTIFACTS = {
    "cluster_bootstrap_interventions.csv",
    "cluster_structure_report.csv",
    "final_76_campaign.csv",
    "final_76_campaign_provenance.json",
    "frozen_vs_monte_carlo.csv",
    "int4_bbq_context_polarity.csv",
    "int4_category_deltas.csv",
    "int4_demographic_group_deltas.csv",
    "intervention_deltas.csv",
    "monte_carlo_model_ranking_stability.csv",
    "monte_carlo_provenance.json",
    "monte_carlo_sampling_errors.csv",
    "monte_carlo_tolerance_rates.csv",
}


def test_curated_aggregate_checksums_match_exact_file_bytes() -> None:
    """Require each released aggregate to match its published SHA-256 digest."""
    entries: dict[str, str] = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        assert len(digest) == 64
        assert filename == Path(filename).name
        assert filename not in entries
        entries[filename] = digest

    assert set(entries) == EXPECTED_ARTIFACTS
    assert {path.name for path in AGGREGATES.iterdir() if path.is_file()} == EXPECTED_ARTIFACTS | {"SHA256SUMS"}
    for filename, expected_digest in entries.items():
        assert (AGGREGATES / filename).is_file()
        actual_digest = hashlib.sha256((AGGREGATES / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest
