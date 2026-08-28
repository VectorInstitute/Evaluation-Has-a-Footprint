"""Ground-truth MCQ metric helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def summary(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return production-equivalent accuracy counts."""
    rows = list(predictions)
    scored = [row for row in rows if isinstance(row.get("correct"), bool)]
    correct = sum(row["correct"] for row in scored)
    return {
        "attempted": len(rows),
        "scored": len(scored),
        "parse_failures": sum(row.get("status") == "parse_failure" for row in rows),
        "inference_failures": sum(row.get("status") == "inference_failure" for row in rows),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
    }


def breakdown(
    predictions: Iterable[dict[str, Any]], metadata: dict[str, dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Summarize predictions by explicit benchmark-native fields."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["sample_id"] not in metadata:
            raise ValueError("Prediction references unknown frozen sample_id")
        groups[tuple(str(metadata[row["sample_id"]][field]) for field in fields)].append(row)
    return [{**dict(zip(fields, key, strict=True)), **summary(rows)} for key, rows in sorted(groups.items())]
