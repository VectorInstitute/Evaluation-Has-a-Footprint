"""Infrastructure-free execution of a caller-prepared frozen evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .conditions import Condition
from .datasets import load_membership, load_prepared_dataset, select_frozen_subset
from .inference import (
    SMOKE_MAX_NEW_TOKENS,
    SMOKE_PROMPT_SHA256,
    SMOKE_PROMPT_VERSION,
    build_messages,
    generate,
    load_model,
    prepare_inputs,
)
from .metrics import breakdown, summary
from .model_registry import MODELS
from .parsing import recover_maintenance
from .telemetry import start_measurement, stop_measurement


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_stable_json(value) + "\n", encoding="utf-8")


def _runtime_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "accelerate", "bitsandbytes", "qwen-vl-utils", "torchvision", "Pillow")
    values: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def _predict(
    bundle: dict[str, Any], records: list[dict[str, Any]], dataset: str, condition: Condition
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(records), condition.batch_size):
        batch = records[start : start + condition.batch_size]
        started = time.perf_counter()
        try:
            raw_outputs = generate(
                bundle,
                prepare_inputs(
                    bundle, [build_messages(record, dataset) for record in batch], max_pixels=condition.max_pixels
                ),
            )
            if len(raw_outputs) != len(batch):
                raise RuntimeError("Generation output count differs from batch size")
        except Exception as error:
            latency = time.perf_counter() - started
            for record in batch:
                predictions.append(
                    {
                        "sample_id": record["sample_id"],
                        "group_id": record["group_id"],
                        "input_ref": {
                            "category": record["category"],
                            "subcategory": record.get("subcategory"),
                            "image_sha256": record.get("image_sha256"),
                        },
                        "output": {"raw": None, "answer": None, "reasoning": None},
                        "correct": None,
                        "status": "inference_failure",
                        "error": type(error).__name__,
                        "batch_size": len(batch),
                        "batch_latency_seconds": round(latency, 6),
                    }
                )
            continue
        latency = time.perf_counter() - started
        for record, raw in zip(batch, raw_outputs, strict=True):
            parsed = recover_maintenance(raw, (record["ans0"], record["ans1"], record["ans2"]))
            answer = parsed.answer
            predictions.append(
                {
                    "sample_id": record["sample_id"],
                    "group_id": record["group_id"],
                    "input_ref": {
                        "category": record["category"],
                        "subcategory": record.get("subcategory"),
                        "image_sha256": record.get("image_sha256"),
                    },
                    "output": {"raw": raw, "answer": answer, "reasoning": parsed.reasoning},
                    "correct": answer == "ABC"[int(record["label"])] if answer is not None else None,
                    "status": "success" if parsed.error is None else "parse_failure",
                    "error": parsed.error,
                    "parser_method": parsed.method,
                    "batch_size": len(batch),
                    "batch_latency_seconds": round(latency, 6),
                }
            )
    return predictions


def run_prepared_evaluation(
    *,
    model_key: str,
    dataset: str,
    condition: Condition,
    prepared_dataset: Path,
    output: Path,
    membership_path: Path | None = None,
    telemetry: bool = False,
    telemetry_device_index: int | None = None,
    carbon_profile: str | None = None,
    carbon_intensity_g_per_kwh: float | None = None,
) -> dict[str, Any]:
    """Run one accepted campaign profile without downloads, schedulers, or tracking."""
    required_modality = "text" if dataset == "bbq" else "image_text"
    if required_modality not in MODELS[model_key]["modalities"]:
        raise ValueError(f"{model_key} does not support {dataset}")
    records, dataset_info = load_prepared_dataset(prepared_dataset, dataset)
    membership_sha256: str | None = None
    if condition.uses_subset:
        if membership_path is None:
            raise ValueError("M4/M5 requires an explicit frozen membership file")
        records, membership_sha256 = select_frozen_subset(
            records,
            load_membership(membership_path),
            dataset=dataset,
            source_fingerprint=dataset_info["sample_fingerprint"],
            target_rows=condition.subset_rows or 0,
            replicate=condition.replicate or 1,
        )
    elif membership_path is not None:
        raise ValueError("Membership is valid only for M4/M5")
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_model(model_key, quantization=condition.quantization)
    telemetry_handle: dict[str, Any] | None = None
    footprint: dict[str, Any] | None = None
    if telemetry:
        telemetry_handle = start_measurement(
            device_index=telemetry_device_index,
            carbon_profile=carbon_profile,
            carbon_intensity_g_per_kwh=carbon_intensity_g_per_kwh,
        )
    try:
        predictions = _predict(bundle, records, dataset, condition)
    finally:
        if telemetry_handle is not None:
            footprint = stop_measurement(telemetry_handle, evaluated_items=len(records))
    metadata = {record["sample_id"]: record for record in records}
    metrics = {
        "schema_version": "public-metrics-v1",
        "source": {"dataset_fingerprint": dataset_info["sample_fingerprint"]},
        "prediction_fingerprint": hashlib.sha256(
            "".join(_stable_json(row) + "\n" for row in predictions).encode()
        ).hexdigest(),
        "gt": summary(predictions),
    }
    breakdowns: dict[str, Any] = {
        "category": breakdown(predictions, metadata, ("category",)),
        "question_polarity": breakdown(predictions, metadata, ("question_polarity",)),
    }
    if dataset == "bbq":
        breakdowns["context_condition"] = breakdown(predictions, metadata, ("context_condition",))
        breakdowns["category_x_context_condition"] = breakdown(predictions, metadata, ("category", "context_condition"))
    metrics["breakdowns"] = breakdowns
    run = {
        "schema_version": "public-run-v1",
        "model": {"key": model_key, "hf_id": MODELS[model_key]["hf_id"], "revision": MODELS[model_key]["revision"]},
        "dataset": {
            "name": dataset,
            "sample_fingerprint": dataset_info["sample_fingerprint"],
            "evaluated_rows": len(records),
        },
        "condition": {
            "name": condition.name,
            "batch_size": condition.batch_size,
            "dtype": condition.dtype,
            "quantization": condition.quantization,
            "subset_rows": condition.subset_rows,
            "replicate": condition.replicate,
            "max_pixels": condition.max_pixels,
        },
        "generation": {
            "prompt_version": SMOKE_PROMPT_VERSION,
            "prompt_sha256": SMOKE_PROMPT_SHA256,
            "do_sample": False,
            "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
        },
        "membership_fingerprint": membership_sha256,
        "software": _runtime_versions(),
        "quantization_verification": bundle["quantization"],
    }
    if footprint is not None:
        run["telemetry"] = {
            "enabled": True,
            "measurement_status": footprint["measurement_status"],
            "footprint_artifact": "footprint.json",
        }
    _write_json(output / "run.json", run)
    (output / "predictions.jsonl").write_text(
        "".join(_stable_json(row) + "\n" for row in predictions), encoding="utf-8"
    )
    _write_json(output / "metrics.json", metrics)
    if footprint is not None:
        _write_json(output / "footprint.json", footprint)
    result: dict[str, Any] = {"run": run, "metrics": metrics, "predictions": predictions}
    if footprint is not None:
        result["footprint"] = footprint
    return result
