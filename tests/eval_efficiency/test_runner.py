"""Offline tests of the public runner artifact path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_efficiency.conditions import resolve_condition
from eval_efficiency.runner import _scientific_prediction_fingerprint, run_prepared_evaluation


def _records() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "age:0",
            "group_id": "age:c1",
            "category": "age",
            "subcategory": "sub",
            "question_polarity": "neg",
            "context_condition": "ambig",
            "context": "context",
            "question": "question",
            "ans0": "one",
            "ans1": "two",
            "ans2": "three",
            "label": "0",
        }
    ]


def test_prediction_fingerprint_excludes_runtime_metadata() -> None:
    predictions = [
        {
            "sample_id": "age:0",
            "output": {"answer": "A", "raw": "first", "reasoning": "x"},
            "correct": True,
            "status": "success",
            "batch_latency_seconds": 0.1,
        },
        {
            "sample_id": "age:1",
            "output": {"answer": None, "raw": None, "reasoning": None},
            "correct": None,
            "status": "parse_failure",
            "batch_latency_seconds": 0.2,
        },
    ]
    changed_runtime = [{**row, "batch_latency_seconds": 99.0} for row in predictions]
    assert _scientific_prediction_fingerprint(predictions) == _scientific_prediction_fingerprint(changed_runtime)
    assert (
        _scientific_prediction_fingerprint(predictions)
        == "b0e12e782eb61e69722ef15bbacc2ae111245aeb75db1395193666f2c5f3a644"
    )


def test_runner_writes_portable_artifacts(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "eval_efficiency.runner.load_prepared_dataset",
        lambda *_: (_records(), {"sample_fingerprint": "dataset-sha"}),
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.load_model", lambda *_args, **_kwargs: {"quantization": {"verified": True}}
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.prepare_inputs", lambda *_args, **_kwargs: {"input_ids": object()}
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.generate", lambda *_args, **_kwargs: ['{"answer":"A","reasoning":"x"}']
    )
    condition = resolve_condition("M0", model_key="qwen25_vl_7b", dataset="bbq")
    result = run_prepared_evaluation(
        model_key="qwen25_vl_7b", dataset="bbq", condition=condition, prepared_dataset=tmp_path, output=tmp_path / "run"
    )
    assert result["metrics"]["gt"]["accuracy"] == 1.0
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["model"]["revision"] == "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    assert run["generation"]["do_sample"] is False
    assert (tmp_path / "run" / "predictions.jsonl").is_file()
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert not (tmp_path / "run" / "footprint.json").exists()


def test_runner_writes_footprint_only_when_telemetry_enabled(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "eval_efficiency.runner.load_prepared_dataset",
        lambda *_: (_records(), {"sample_fingerprint": "dataset-sha"}),
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.load_model", lambda *_args, **_kwargs: {"quantization": {"verified": True}}
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.prepare_inputs", lambda *_args, **_kwargs: {"input_ids": object()}
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.generate", lambda *_args, **_kwargs: ['{"answer":"A","reasoning":"x"}']
    )
    events: list[str] = []
    monkeypatch.setattr(
        "eval_efficiency.runner.start_measurement", lambda **_kwargs: events.append("start") or {}
    )
    monkeypatch.setattr(
        "eval_efficiency.runner.stop_measurement",
        lambda _handle, **_kwargs: (
            events.append("stop")
            or {"measurement_status": "measured", "boundary": "inference_only", "schema_version": "public-footprint-v1"}
        ),
    )
    condition = resolve_condition("M0", model_key="qwen25_vl_7b", dataset="bbq")
    run_prepared_evaluation(
        model_key="qwen25_vl_7b",
        dataset="bbq",
        condition=condition,
        prepared_dataset=tmp_path,
        output=tmp_path / "telemetry-run",
        telemetry=True,
        carbon_profile="accepted-campaign",
    )
    assert events == ["start", "stop"]
    footprint = json.loads((tmp_path / "telemetry-run" / "footprint.json").read_text())
    assert footprint["boundary"] == "inference_only"
