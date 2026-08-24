"""Fail-closed loading, validation, and selection of prepared frozen inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_SPECS = {
    "bbq": {
        "fields": {
            "category",
            "case_id",
            "example_id",
            "question_index",
            "context_condition",
            "question_polarity",
            "context",
            "question",
            "ans0",
            "ans1",
            "ans2",
            "label",
            "answer_info",
            "subcategory",
            "stereotyped_groups",
            "target_loc",
            "label_type",
            "known_stereotyped_groups",
            "known_stereotyped_race",
            "known_stereotyped_var2",
        },
        "fingerprint_fields": (
            "category",
            "case_id",
            "example_id",
            "question_index",
            "context_condition",
            "question_polarity",
            "label",
            "target_loc",
        ),
    },
    "bbq_v": {
        "fields": {
            "id",
            "cross_id",
            "unique_question_id",
            "category",
            "subcategory",
            "stereotyped_groups",
            "question_polarity",
            "context",
            "question",
            "ans0",
            "ans1",
            "ans2",
            "label",
            "gold_answer",
            "image_sha256",
            "image_file",
        },
        "fingerprint_fields": ("id", "cross_id", "unique_question_id", "image_sha256", "label"),
    },
}
FINGERPRINT_FIELDS = {name: spec["fingerprint_fields"] for name, spec in DATASET_SPECS.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_fingerprint(unit_ids: Iterable[str]) -> str:
    """Return the canonical SHA-256 of sorted membership IDs."""
    return hashlib.sha256("".join(f"{item}\n" for item in sorted(unit_ids)).encode()).hexdigest()


def dataset_fingerprint(
    dataset: str, rows: Iterable[dict[str, str]], *, category_order: Iterable[str] | None = None
) -> str:
    """Return the accepted-campaign frozen-sample fingerprint."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    values = list(rows)
    if dataset == "bbq":
        if category_order is None:
            raise ValueError("BBQ fingerprinting requires category_order")
        rank = {value: index for index, value in enumerate(category_order)}
        try:
            ordered = sorted(values, key=lambda row: (rank[row["category"]], int(row["example_id"])))
        except KeyError as error:
            raise ValueError(f"Unknown BBQ category in category_order: {error.args[0]}") from error
    else:
        ordered = sorted(values, key=lambda row: row["id"])
    fields = FINGERPRINT_FIELDS[dataset]
    try:
        body = "".join(",".join(row[field] for field in fields) + "\n" for row in ordered)
    except KeyError as error:
        raise ValueError(f"Missing fingerprint field: {error.args[0]}") from error
    return hashlib.sha256(body.encode()).hexdigest()


def unit_id(dataset: str, row: dict[str, str]) -> str:
    """Return the indivisible frozen-subset unit identifier."""
    if dataset == "bbq":
        return f"{row['category']}:{row['case_id']}"
    if dataset == "bbq_v":
        return row["unique_question_id"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _require_columns(dataset: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Frozen sample is empty")
    missing = set(DATASET_SPECS[dataset]["fields"]) - set(rows[0])
    if missing:
        raise ValueError(f"Frozen {dataset} sample is missing required columns: {sorted(missing)}")


def _validate_bbq_units(groups: dict[str, list[dict[str, str]]]) -> None:
    expected = {("ambig", "neg"), ("ambig", "nonneg"), ("disambig", "neg"), ("disambig", "nonneg")}
    for key, group in groups.items():
        cells = {(row["context_condition"], row["question_polarity"]) for row in group}
        if len(group) != 4 or cells != expected:
            raise ValueError(f"Frozen BBQ sample has incomplete case: {key}")


def _validate_bbqv_units(groups: dict[str, list[dict[str, str]]]) -> None:
    for key, group in groups.items():
        if {row["question_polarity"] for row in group} != {"neg", "nonneg"} or len(
            {row["cross_id"] for row in group}
        ) != 2:
            raise ValueError(f"Frozen BBQ-V sample has incomplete scenario: {key}")


def _validate_bbqv_images(rows: list[dict[str, str]], images_dir: Path) -> None:
    root = images_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"BBQ-V image directory is not a directory: {images_dir}")
    hash_files: dict[str, str] = {}
    verified: dict[str, str] = {}
    for row in rows:
        filename = row["image_file"]
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".jpg":
            raise ValueError(f"Unsafe BBQ-V image_file: {filename!r}")
        previous = hash_files.setdefault(row["image_sha256"], filename)
        if previous != filename:
            raise ValueError("One BBQ-V image hash maps to multiple local filenames")
        image = images_dir / filename
        if not image.is_file():
            raise ValueError(f"Missing frozen BBQ-V image: {filename}")
        resolved = image.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"BBQ-V image resolves outside prepared image directory: {filename}")
        actual = verified.setdefault(filename, _sha256_file(resolved))
        if row["image_sha256"] and actual != row["image_sha256"]:
            raise ValueError(f"Frozen BBQ-V image hash mismatch: {filename}")


def validate_rows(dataset: str, rows: Iterable[dict[str, str]], *, images_dir: Path | None = None) -> None:
    """Validate the final campaign's complete natural units and local images."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    values = list(rows)
    _require_columns(dataset, values)
    identifiers = (
        {(row["category"], row["example_id"]) for row in values} if dataset == "bbq" else {row["id"] for row in values}
    )
    if len(identifiers) != len(values):
        raise ValueError(f"Frozen {dataset} sample has duplicate item identifiers")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in values:
        groups[unit_id(dataset, row)].append(row)
    if dataset == "bbq":
        _validate_bbq_units(groups)
        return
    _validate_bbqv_units(groups)
    if images_dir is None:
        return
    _validate_bbqv_images(values, images_dir)


def load_prepared_dataset(path: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a caller-prepared ``sample.csv`` and immutable sampling provenance."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    sample_path = path / "sample.csv" if path.is_dir() else path
    root = sample_path.parent
    provenance_path = root / "sampling.json"
    if not sample_path.is_file() or not provenance_path.is_file():
        raise ValueError("Prepared data requires sample.csv and sampling.json in the same directory")
    with sample_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    images_dir = root / "images" if dataset == "bbq_v" else None
    validate_rows(dataset, rows, images_dir=images_dir)
    category_order = provenance.get("sampling", {}).get("category_order") if dataset == "bbq" else None
    fingerprint = dataset_fingerprint(dataset, rows, category_order=category_order)
    expected = provenance.get("validation", {}).get("sample_fingerprint", {}).get("sha256")
    if not expected or fingerprint != expected:
        raise ValueError("Prepared frozen dataset fingerprint does not match sampling.json")
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = dict(row)
        if dataset == "bbq":
            record["sample_id"] = f"{row['category']}:{row['example_id']}"
            record["group_id"] = unit_id(dataset, row)
            record["gold_answer"] = row[f"ans{row['label']}"]
        else:
            assert images_dir is not None
            record["sample_id"] = row["id"]
            record["group_id"] = unit_id(dataset, row)
            record["image_path"] = str((images_dir / row["image_file"]).resolve(strict=True))
        records.append(record)
    return records, {"sample_fingerprint": fingerprint, "provenance": provenance}


def select_frozen_subset(
    rows: Iterable[dict[str, str]],
    manifest: dict[str, Any],
    *,
    dataset: str,
    source_fingerprint: str,
    target_rows: int,
    replicate: int = 1,
) -> tuple[list[dict[str, str]], str]:
    """Verify and select one recorded membership without resampling."""
    values = list(rows)
    validate_rows(dataset, values)
    if (
        manifest.get("dataset") != dataset
        or manifest.get("provenance", {}).get("source_sample_fingerprint") != source_fingerprint
    ):
        raise ValueError("Membership does not match the frozen dataset")
    entry = manifest.get("subsets", {}).get(str(target_rows))
    if not isinstance(entry, dict):
        raise ValueError("Unknown target_rows")
    selected = entry.get("replicates", {}).get(str(replicate), entry if replicate == 1 else None)
    if not isinstance(selected, dict):
        raise ValueError("Unknown replicate")
    ids = selected.get("selected_unit_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("Invalid membership IDs")
    actual = membership_fingerprint(ids)
    if selected.get("membership_fingerprint", {}).get("sha256") != actual:
        raise ValueError("Membership fingerprint mismatch")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in values:
        grouped[unit_id(dataset, row)].append(row)
    unknown = set(ids) - set(grouped)
    if unknown:
        raise ValueError(f"Unknown membership ID: {sorted(unknown)[0]}")
    chosen = [row for identifier in ids for row in grouped[identifier]]
    if len(chosen) != target_rows:
        raise ValueError("Membership row count mismatch")
    return sorted(chosen, key=lambda row: row.get("sample_id", row.get("id", row.get("example_id", "")))), actual


def load_membership(path: Path) -> dict[str, Any]:
    """Load a JSON membership document without assuming a project layout."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Membership document must be an object")
    return value
