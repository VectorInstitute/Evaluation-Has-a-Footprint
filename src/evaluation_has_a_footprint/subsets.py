"""Freeze deterministic nested M4 membership IDs without duplicating frozen rows."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache
from typing import Any


NAMESPACE = "evaluation-footprint-m4-membership-v1"
SEED = "20260818"
TARGETS = {
    "bbq": (2000, 1500, 1000, 748, 500, 248),
    "bbq_v": (1998, 1494, 1008, 756, 504, 252),
}

# Additional deterministic, distinct memberships (replicates 2-5) are generated
# only for the ~50% and ~12-13% row counts, per the finalized 19-run
# design. Replicate 1 always equals the pre-existing canonical M4 curve member
# for that target, unchanged, produced by the default SEED.
REPLICATE_COUNT = 5
REPLICATE_TARGETS = {"bbq": (1000, 248), "bbq_v": (1008, 252)}
TARGET_TO_PER_CATEGORY_INCREMENT = {1494: 166, 1008: 112, 756: 84, 504: 56, 252: 28}


def _replicate_seed(replicate: int) -> str:
    if replicate == 1:
        return SEED
    return f"{SEED}-rep{replicate}"


def _hash(dataset: str, category: str, unit_id: str, seed: str = SEED) -> str:
    text = f"{NAMESPACE}|{seed}|{dataset}|{category}|{unit_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint(unit_ids: list[str]) -> str:
    return hashlib.sha256("".join(f"{unit_id}\n" for unit_id in sorted(unit_ids)).encode("utf-8")).hexdigest()


def _groups(
    records: list[dict[str, Any]],
    dataset: str,
    seed: str = SEED,
) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    key_name = "case_id" if dataset == "bbq" else "unique_question_id"
    for record in records:
        grouped[(record["category"], record[key_name])].append(record)
    expected_size = 4 if dataset == "bbq" else None
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (category, source_id), rows in grouped.items():
        if expected_size is not None and len(rows) != expected_size:
            raise RuntimeError(f"Incomplete BBQ case: {category}:{source_id}")
        if dataset == "bbq_v" and (
            len({row["question_polarity"] for row in rows}) != 2 or len({row["cross_id"] for row in rows}) != 2
        ):
            raise RuntimeError(f"Incomplete BBQ-V scenario: {source_id}")
        unit_id = f"{category}:{source_id}" if dataset == "bbq" else source_id
        by_category[category].append((unit_id, rows))
    return {
        category: sorted(units, key=lambda item: (_hash(dataset, category, item[0], seed), item[0]))
        for category, units in sorted(by_category.items())
    }


def _balanced_case_selection(
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]],
    target_rows: int,
    seed: str = SEED,
) -> set[str]:
    if target_rows == 2000:
        return {unit_id for units in by_category.values() for unit_id, _ in units}
    if target_rows % 4:
        raise RuntimeError(f"BBQ target is not complete-case aligned: {target_rows}")
    target_cases = target_rows // 4
    categories = sorted(
        by_category,
        key=lambda category: (_hash("bbq", "category", category, seed), category),
    )
    base, remainder = divmod(target_cases, len(categories))
    selected: set[str] = set()
    for index, category in enumerate(categories):
        quota = base + (index < remainder)
        if quota > len(by_category[category]):
            raise RuntimeError(f"BBQ category quota exceeds availability: {category}")
        selected.update(unit_id for unit_id, _ in by_category[category][:quota])
    return selected


def _layered_scenario_selection(
    units: list[tuple[str, list[dict[str, Any]]]],
) -> dict[int, set[str]]:
    """Return exact nested per-category selections for 28, 56, 84, 112, and 166 rows."""
    weights = [(unit_id, len(rows)) for unit_id, rows in units]
    if any(weight % 2 for _, weight in weights):
        raise RuntimeError("BBQ-V scenario sizes must be even for the balanced targets.")
    increments = (28, 28, 28, 28, 54)
    scaled_weights = tuple((unit_id, weight // 2) for unit_id, weight in weights)
    scaled_needs = tuple(value // 2 for value in increments)
    suffix = [0] * (len(scaled_weights) + 1)
    for index in range(len(scaled_weights) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + scaled_weights[index][1]

    @lru_cache(maxsize=None)
    def assign(index: int, needs: tuple[int, ...]) -> tuple[int | None, ...] | None:
        if not any(needs):
            return (None,) * (len(scaled_weights) - index)
        if index == len(scaled_weights) or suffix[index] < sum(needs):
            return None
        unit_id, weight = scaled_weights[index]
        for layer_index, needed in enumerate(needs):
            if weight > needed:
                continue
            next_needs = list(needs)
            next_needs[layer_index] -= weight
            result = assign(index + 1, tuple(next_needs))
            if result is not None:
                return (layer_index,) + result
        result = assign(index + 1, needs)
        return (None,) + result if result is not None else None

    assignments = assign(0, scaled_needs)
    if assignments is None:
        raise RuntimeError("No exact layered BBQ-V scenario selection was found.")
    layers: list[set[str]] = [set() for _ in increments]
    for (unit_id, _), layer_index in zip(scaled_weights, assignments, strict=True):
        if layer_index is not None:
            layers[layer_index].add(unit_id)
    selections: dict[int, set[str]] = {}
    selected: set[str] = set()
    for target, layer in zip((28, 56, 84, 112, 166), layers, strict=True):
        selected.update(layer)
        rows = sum(weight for unit_id, weight in weights if unit_id in selected)
        if rows != target:
            raise RuntimeError(f"Layered scenario selection produced {rows}, expected {target}.")
        selections[target] = set(selected)
    return selections


def _membership(  # noqa: PLR0917
    dataset: str,
    target_rows: int,
    selected_ids: set[str],
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]],
    source_fingerprint: str,
    parent_subset_id: str | None,
    *,
    replicate: int = 1,
    seed: str = SEED,
) -> dict[str, Any]:
    all_units = {unit_id: (category, len(rows)) for category, units in by_category.items() for unit_id, rows in units}
    unknown = selected_ids - set(all_units)
    if unknown:
        raise RuntimeError(f"Unknown membership IDs, first: {sorted(unknown)[0]}")
    actual_rows = sum(all_units[unit_id][1] for unit_id in selected_ids)
    category_rows = {
        category: sum(
            rows
            for unit_id, (item_category, rows) in all_units.items()
            if item_category == category and unit_id in selected_ids
        )
        for category in sorted(by_category)
    }
    if actual_rows != target_rows:
        raise RuntimeError(f"{dataset} membership has {actual_rows} rows, expected {target_rows}.")
    return {
        "schema_version": 1,
        "subset_id": f"m4_{target_rows}",
        "dataset": dataset,
        "membership_unit": "(category, case_id)" if dataset == "bbq" else "(category, unique_question_id)",
        "target_rows": target_rows,
        "replicate": replicate,
        "selected_unit_ids": sorted(selected_ids),
        "provenance": {
            "source_sample_fingerprint": source_fingerprint,
            "selection_namespace": NAMESPACE,
            "seed": seed,
            "ranking": "SHA-256(namespace|seed|dataset|category|unit_id), ascending digest then unit_id",
        },
        "validation": {
            "actual_rows": actual_rows,
            "complete_units": len(selected_ids),
            "category_rows": category_rows,
            "membership_fingerprint": {
                "algorithm": "SHA-256 over sorted selected_unit_ids, one UTF-8 ID plus newline per line",
                "sha256": _fingerprint(sorted(selected_ids)),
            },
            "parent_subset_id": parent_subset_id,
        },
    }


def _replicate_selection(
    dataset: str,
    target_rows: int,
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]],
    records: list[dict[str, Any]],
    seed: str,
) -> set[str]:
    """Recompute one seeded selection for a single (non-nested) replicate target."""
    if dataset == "bbq":
        return _balanced_case_selection(by_category, target_rows, seed=seed)
    seeded_by_category = _groups(records, dataset, seed=seed)
    per_category_rows = TARGET_TO_PER_CATEGORY_INCREMENT[target_rows]
    per_category = {
        category: _layered_scenario_selection(units)[per_category_rows]
        for category, units in seeded_by_category.items()
    }
    return set().union(*(per_category[category] for category in sorted(seeded_by_category)))


def _build_replicate_memberships(
    records: list[dict[str, Any]],
    dataset: str,
    source_fingerprint: str,
    canonical: dict[int, dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Build replicates 2--5; replicate 1 reuses the canonical membership."""
    by_category = _groups(records, dataset)
    result: dict[int, dict[int, dict[str, Any]]] = {}
    for target_rows in REPLICATE_TARGETS[dataset]:
        by_replicate: dict[int, dict[str, Any]] = {1: canonical[target_rows]}
        seen_id_sets = {frozenset(canonical[target_rows]["selected_unit_ids"])}
        for replicate in range(2, REPLICATE_COUNT + 1):
            seed = _replicate_seed(replicate)
            selected_ids = _replicate_selection(dataset, target_rows, by_category, records, seed)
            if frozenset(selected_ids) in seen_id_sets:
                raise RuntimeError(f"{dataset} {target_rows} replicate {replicate} duplicates an earlier replicate.")
            seen_id_sets.add(frozenset(selected_ids))
            by_replicate[replicate] = _membership(
                dataset,
                target_rows,
                selected_ids,
                by_category,
                source_fingerprint,
                parent_subset_id=None,
                replicate=replicate,
                seed=seed,
            )
        result[target_rows] = by_replicate
    return result


def _build_memberships(records: list[dict[str, Any]], dataset: str, source_fingerprint: str) -> list[dict[str, Any]]:
    by_category = _groups(records, dataset)
    if dataset == "bbq":
        selected_by_target = {target: _balanced_case_selection(by_category, target) for target in TARGETS[dataset]}
    else:
        per_category = {category: _layered_scenario_selection(units) for category, units in by_category.items()}
        selected_by_target = {
            1998: {unit_id for units in by_category.values() for unit_id, _ in units},
        }
        for target, per_category_rows in TARGET_TO_PER_CATEGORY_INCREMENT.items():
            selected_by_target[target] = set().union(
                *(per_category[category][per_category_rows] for category in sorted(by_category))
            )
    memberships: list[dict[str, Any]] = []
    parent_ids: set[str] | None = None
    parent_name: str | None = None
    for target in TARGETS[dataset]:
        selected_ids = selected_by_target[target]
        if parent_ids is not None and not selected_ids <= parent_ids:
            raise RuntimeError(f"{dataset} {target} is not nested in {parent_name}.")
        memberships.append(
            _membership(
                dataset,
                target,
                selected_ids,
                by_category,
                source_fingerprint,
                parent_name,
            )
        )
        parent_ids, parent_name = selected_ids, f"m4_{target}"
    return memberships


def _replicate_entry(membership: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": membership["provenance"]["seed"],
        "selected_unit_ids": membership["selected_unit_ids"],
        **membership["validation"],
    }


def _consolidated(
    dataset: str,
    memberships: list[dict[str, Any]],
    replicates: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    first = memberships[0]
    subsets: dict[str, Any] = {}
    for membership in memberships:
        entry = {
            "selected_unit_ids": membership["selected_unit_ids"],
            **membership["validation"],
        }
        target = membership["target_rows"]
        if target in replicates:
            entry["replicates"] = {
                str(replicate): _replicate_entry(replicate_membership)
                for replicate, replicate_membership in sorted(replicates[target].items())
            }
        subsets[str(target)] = entry
    return {
        "schema_version": 1,
        "dataset": dataset,
        "membership_unit": first["membership_unit"],
        "provenance": first["provenance"],
        "subsets": subsets,
    }


def build_manifest(records: list[dict[str, Any]], dataset: str, source_fingerprint: str) -> dict[str, Any]:
    """Recreate the final campaign's deterministic M4 manifest from prepared rows."""
    if dataset not in TARGETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    canonical = _build_memberships(records, dataset, source_fingerprint)
    by_target = {membership["target_rows"]: membership for membership in canonical}
    return _consolidated(
        dataset, canonical, _build_replicate_memberships(records, dataset, source_fingerprint, by_target)
    )


def validate_manifest(  # noqa: PLR0912
    manifest: dict[str, Any], dataset: str, source_fingerprint: str | None = None
) -> None:
    """Fail closed when an M4 manifest is structurally corrupt or tampered with."""
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != dataset:
        raise ValueError("Unsupported M4 manifest schema or dataset")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("selection_namespace") != NAMESPACE:
        raise ValueError("Invalid M4 provenance")
    if source_fingerprint is not None and provenance.get("source_sample_fingerprint") != source_fingerprint:
        raise ValueError("M4 source fingerprint mismatch")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict) or set(subsets) != {str(value) for value in TARGETS[dataset]}:
        raise ValueError("Unexpected M4 target rows")
    parent_ids: set[str] | None = None
    parent_name: str | None = None
    for target in TARGETS[dataset]:
        entry = subsets[str(target)]
        ids = entry.get("selected_unit_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Invalid or duplicate membership IDs")
        validation = entry
        if validation.get("actual_rows") != target or validation.get("complete_units") != len(ids):
            raise ValueError("M4 row or unit count mismatch")
        if sum(validation.get("category_rows", {}).values()) != target:
            raise ValueError("M4 category counts do not sum to target rows")
        if validation.get("membership_fingerprint", {}).get("sha256") != _fingerprint(ids):
            raise ValueError("M4 membership fingerprint mismatch")
        if validation.get("parent_subset_id") != parent_name or (parent_ids is not None and not set(ids) < parent_ids):
            raise ValueError("M4 parent subset relationship mismatch")
        replicates = entry.get("replicates", {})
        expected_replicates = (
            {str(value) for value in range(1, REPLICATE_COUNT + 1)} if target in REPLICATE_TARGETS[dataset] else set()
        )
        if set(replicates) != expected_replicates:
            raise ValueError("Unexpected M4 replicate set")
        for replicate, item in replicates.items():
            replicate_ids = item.get("selected_unit_ids")
            if not isinstance(replicate_ids, list) or len(replicate_ids) != len(set(replicate_ids)):
                raise ValueError("Invalid M4 replicate IDs")
            if item.get("actual_rows") != target or item.get("complete_units") != len(replicate_ids):
                raise ValueError("M4 replicate count mismatch")
            if item.get("membership_fingerprint", {}).get("sha256") != _fingerprint(replicate_ids):
                raise ValueError("M4 replicate fingerprint mismatch")
            if replicate == "1" and replicate_ids != ids:
                raise ValueError("M4 replicate 1 must equal canonical membership")
        parent_ids, parent_name = set(ids), f"m4_{target}"


def select_manifest_records(  # noqa: PLR0917
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    dataset: str,
    source_fingerprint: str,
    target_rows: int,
    replicate: int = 1,
) -> list[dict[str, Any]]:
    """Validate and expand one membership against complete prepared natural units."""
    validate_manifest(manifest, dataset, source_fingerprint)
    entry = manifest["subsets"].get(str(target_rows))
    if not isinstance(entry, dict):
        raise ValueError("Unknown M4 target rows")
    selected = entry.get("replicates", {}).get(str(replicate), entry if replicate == 1 else None)
    if not isinstance(selected, dict):
        raise ValueError("Unknown M4 replicate")
    groups = {unit_id: rows for units in _groups(records, dataset).values() for unit_id, rows in units}
    ids = selected["selected_unit_ids"]
    unknown = set(ids) - set(groups)
    if unknown:
        raise ValueError(f"Unknown M4 membership ID: {sorted(unknown)[0]}")
    expanded = [row for unit_id in ids for row in groups[unit_id]]
    if len(expanded) != target_rows:
        raise ValueError("M4 row expansion does not match target rows")
    return expanded
