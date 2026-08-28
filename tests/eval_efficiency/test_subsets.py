"""Synthetic coverage for public M4 membership generation and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from eval_efficiency.subsets import (
    REPLICATE_TARGETS,
    TARGETS,
    _hash,
    build_manifest,
    select_manifest_records,
    validate_manifest,
)


def _bbq_rows() -> list[dict[str, str]]:
    return [
        {"category": f"c{category}", "case_id": f"case{case}", "id": f"{category}-{case}-{row}"}
        for category in range(11)
        for case in range(45 + (category < 5))
        for row in range(4)
    ]


def _bbqv_rows() -> list[dict[str, str]]:
    return [
        {
            "category": f"c{category}",
            "unique_question_id": f"q{category}-{scenario}",
            "id": f"{category}-{scenario}-{polarity}",
            "question_polarity": polarity,
            "cross_id": polarity,
        }
        for category in range(9)
        for scenario in range(111)
        for polarity in ("neg", "nonneg")
    ]


@pytest.mark.parametrize(("dataset", "factory"), [("bbq", _bbq_rows), ("bbq_v", _bbqv_rows)])
def test_generation_is_deterministic_and_valid(dataset: str, factory: Callable[[], list[dict[str, str]]]) -> None:
    rows = factory()
    first = build_manifest(rows, dataset, "a" * 64)
    assert first == build_manifest(rows, dataset, "a" * 64)
    validate_manifest(first, dataset, "a" * 64)
    assert set(first["subsets"]) == {str(target) for target in TARGETS[dataset]}
    for target in REPLICATE_TARGETS[dataset]:
        assert set(first["subsets"][str(target)]["replicates"]) == {"1", "2", "3", "4", "5"}


def test_manifest_tampering_and_unknown_ids_fail_closed() -> None:
    rows = _bbq_rows()
    manifest = build_manifest(rows, "bbq", "a" * 64)
    manifest["subsets"]["1000"]["membership_fingerprint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        validate_manifest(manifest, "bbq", "a" * 64)
    manifest = build_manifest(rows, "bbq", "a" * 64)
    manifest["subsets"]["1000"]["selected_unit_ids"][0] = "unknown"
    with pytest.raises(ValueError):
        select_manifest_records(rows, manifest, "bbq", "a" * 64, 1000)


def test_sha256_ranking_changes_with_seed() -> None:
    assert _hash("bbq", "c0", "case0") == _hash("bbq", "c0", "case0")
    assert _hash("bbq", "c0", "case0") != _hash("bbq", "c0", "case0", seed="different")


@pytest.mark.parametrize("name,dataset", [("bbq_m4_subsets.json", "bbq"), ("bbq_v_m4_subsets.json", "bbq_v")])
def test_public_manifest_schema(name: str, dataset: str) -> None:
    path = Path(__file__).parents[2] / "data/manifests" / name
    validate_manifest(json.loads(path.read_text(encoding="utf-8")), dataset)
