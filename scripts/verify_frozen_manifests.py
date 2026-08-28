#!/usr/bin/env python3
"""Verify metadata-only frozen manifests against separately prepared CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into a list of string-valued dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def require_columns(rows: list[dict[str, str]], fields: Iterable[str], *, name: str) -> None:
    """Raise when a CSV lacks an expected public manifest column."""
    missing = set(fields) - set(rows[0])
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def verify_bbq(sample_path: Path) -> None:
    """Verify the BBQ public manifest against a prepared upstream sample."""
    manifest = read_csv(MANIFEST_DIR / "bbq_frozen_ids.csv")
    sample = read_csv(sample_path)
    fields = ("category", "example_id", "case_id", "question_index", "context_condition", "question_polarity")
    require_columns(manifest, fields + ("natural_unit_id",), name="BBQ manifest")
    require_columns(sample, fields, name="BBQ sample")
    expected = [tuple(row[field] for field in fields) for row in manifest]
    observed = [tuple(row[field] for field in fields) for row in sample]
    if observed != expected:
        raise ValueError("BBQ prepared rows do not exactly match the frozen manifest order and identifiers")
    if len(expected) != 2000 or len(set(expected)) != 2000:
        raise ValueError("BBQ manifest must contain exactly 2,000 unique rows")
    units: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        expected_unit = f"{row['category']}:{row['case_id']}"
        if row["natural_unit_id"] != expected_unit:
            raise ValueError(f"BBQ malformed natural unit ID: {row['natural_unit_id']!r}")
        units[expected_unit].append(row)
    required_cells = {("ambig", "neg"), ("ambig", "nonneg"), ("disambig", "neg"), ("disambig", "nonneg")}
    if len(units) != 500 or any(
        len(rows) != 4 or {(row["context_condition"], row["question_polarity"]) for row in rows} != required_cells
        for rows in units.values()
    ):
        raise ValueError("BBQ manifest does not contain 500 complete four-row cases")
    expected_counts = {
        "Age": 180,
        "Disability_status": 180,
        "Gender_identity": 184,
        "Nationality": 180,
        "Physical_appearance": 180,
        "Race_ethnicity": 184,
        "Race_x_SES": 184,
        "Race_x_gender": 184,
        "Religion": 180,
        "SES": 184,
        "Sexual_orientation": 180,
    }
    if dict(Counter(row["category"] for row in manifest)) != expected_counts:
        raise ValueError("BBQ category row counts differ from the accepted frozen sample")
    print("BBQ: 2,000 rows, 500 complete cases, exact frozen membership verified")


def verify_bbq_v(sample_path: Path) -> None:
    """Verify the BBQ-V public manifest against a prepared upstream sample."""
    manifest = read_csv(MANIFEST_DIR / "bbq_v_frozen_ids.csv")
    sample = read_csv(sample_path)
    fields = ("id", "cross_id", "unique_question_id", "category", "question_polarity", "image_sha256")
    require_columns(manifest, fields + ("natural_unit_id",), name="BBQ-V manifest")
    require_columns(sample, fields, name="BBQ-V sample")
    expected = [tuple(row[field] for field in fields) for row in manifest]
    observed = [tuple(row[field] for field in fields) for row in sample]
    if observed != expected:
        raise ValueError("BBQ-V prepared rows do not exactly match the frozen manifest order and identifiers")
    if len(expected) != 1998 or len({row["id"] for row in manifest}) != 1998:
        raise ValueError("BBQ-V manifest must contain exactly 1,998 unique source rows")
    scenarios: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["natural_unit_id"] != row["unique_question_id"]:
            raise ValueError(f"BBQ-V malformed natural unit ID: {row['natural_unit_id']!r}")
        scenarios[row["unique_question_id"]].append(row)
    if len(scenarios) != 389 or any(
        {row["question_polarity"] for row in rows} != {"neg", "nonneg"} or len({row["cross_id"] for row in rows}) != 2
        for rows in scenarios.values()
    ):
        raise ValueError("BBQ-V manifest does not contain 389 complete scenarios")
    if len({row["image_sha256"] for row in manifest}) != 999:
        raise ValueError("BBQ-V manifest does not contain the expected 999 source image hashes")
    if set(Counter(row["category"] for row in manifest).values()) != {222}:
        raise ValueError("BBQ-V category row counts differ from the accepted frozen sample")
    print("BBQ-V: 1,998 rows, 389 complete scenarios, 999 image hashes, exact frozen membership verified")


def main() -> None:
    """Parse source-sample paths and validate both frozen manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbq-sample", type=Path, required=True)
    parser.add_argument("--bbq-v-sample", type=Path, required=True)
    args = parser.parse_args()
    verify_bbq(args.bbq_sample)
    verify_bbq_v(args.bbq_v_sample)


if __name__ == "__main__":
    main()
