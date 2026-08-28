"""Effective, fail-closed profiles for the accepted Qwen and Gemma campaigns."""

from __future__ import annotations

from dataclasses import dataclass


DATASETS = ("bbq", "bbq_v")
CONDITIONS = ("M0", "M1", "M2", "M3", "M4", "M5a", "M5b")
M5_MEMBERSHIPS = {"bbq": (1000, 1), "bbq_v": (1008, 1)}
M4_TARGET_ROWS = {"bbq": frozenset({1500, 1000, 748, 500, 248}), "bbq_v": frozenset({1494, 1008, 756, 504, 252})}
QWEN3_BBQV_MAX_PIXELS = 1_536 * 1_536


@dataclass(frozen=True)
class Condition:
    """One resolved, scientifically supported campaign configuration."""

    name: str
    model_key: str
    dataset: str
    dtype: str
    quantization: str
    batch_size: int
    uses_subset: bool
    max_pixels: int | None
    subset_rows: int | None = None
    replicate: int | None = None


_BASE = {
    "M0": ("bfloat16", "none", False),
    "M1": ("bfloat16", "none", False),
    "M2": ("bfloat16", "int8", False),
    "M3": ("bfloat16", "int4", False),
    "M4": ("bfloat16", "none", True),
    "M5a": ("bfloat16", "none", True),
    "M5b": ("bfloat16", "int8", True),
}


def _batch_size(model_key: str, dataset: str, condition: str) -> int:
    if condition not in {"M1", "M5a"}:
        return 1
    if model_key == "qwen25_vl_7b":
        return 8
    if model_key == "qwen3_vl_30b_a3b":
        return 2 if dataset == "bbq_v" else 8
    if model_key == "gemma4_12b":
        return 4
    raise ValueError(f"Unsupported model: {model_key}")


def resolve_condition(
    name: str,
    *,
    model_key: str,
    dataset: str,
    batch_size: int | None = None,
    subset_rows: int | None = None,
    replicate: int = 1,
    max_pixels_override: int | None = None,
) -> Condition:
    """Resolve an accepted profile and reject unrecorded condition combinations."""
    if name not in _BASE or dataset not in DATASETS:
        raise ValueError("Unknown condition or dataset")
    if model_key not in {"qwen25_vl_7b", "qwen3_vl_30b_a3b", "gemma4_12b"}:
        raise ValueError(f"Unsupported model: {model_key}")
    if max_pixels_override is not None and model_key == "gemma4_12b":
        raise ValueError("max_pixels_override is not qualified for Gemma")
    dtype, quantization, uses_subset = _BASE[name]
    accepted_batch = _batch_size(model_key, dataset, name)
    if batch_size is not None and batch_size != accepted_batch:
        raise ValueError(f"{model_key}/{dataset}/{name} requires accepted batch size {accepted_batch}")
    if uses_subset != (subset_rows is not None):
        raise ValueError("Subset rows are required only for M4/M5")
    if not uses_subset and replicate != 1:
        raise ValueError("Replicate is valid only for M4/M5")
    if name in {"M5a", "M5b"} and (subset_rows, replicate) != M5_MEMBERSHIPS[dataset]:
        raise ValueError(f"{name} must reuse the accepted 50% {dataset} membership, replicate 1")
    if name == "M4" and subset_rows not in M4_TARGET_ROWS[dataset]:
        raise ValueError(f"M4 must use an accepted frozen {dataset} target size")
    if max_pixels_override is not None and (dataset != "bbq_v" or max_pixels_override < 1):
        raise ValueError("max_pixels_override must be positive and is valid only for BBQ-V")
    profile_max_pixels = QWEN3_BBQV_MAX_PIXELS if (model_key, dataset) == ("qwen3_vl_30b_a3b", "bbq_v") else None
    return Condition(
        name=name,
        model_key=model_key,
        dataset=dataset,
        dtype=dtype,
        quantization=quantization,
        batch_size=accepted_batch,
        uses_subset=uses_subset,
        max_pixels=max_pixels_override if max_pixels_override is not None else profile_max_pixels,
        subset_rows=subset_rows,
        replicate=replicate if uses_subset else None,
    )
