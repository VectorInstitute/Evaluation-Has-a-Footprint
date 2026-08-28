"""Checks that this repository does not ship aggregate result tables."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGGREGATES = ROOT / "results" / "aggregates"
RESULT_SUFFIXES = {".csv", ".json"}


def test_no_aggregate_result_tables_are_shipped() -> None:
    """The public repo is runner/code only; result matrices stay out of tree."""
    if not AGGREGATES.exists():
        return
    leaked = [
        path.name
        for path in AGGREGATES.iterdir()
        if path.is_file() and (path.suffix in RESULT_SUFFIXES or path.name == "SHA256SUMS")
    ]
    assert leaked == []
