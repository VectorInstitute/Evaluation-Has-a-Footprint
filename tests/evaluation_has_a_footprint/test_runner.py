"""Offline tests of the public runner artifact path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation_has_a_footprint.conditions import resolve_condition
from evaluation_has_a_footprint.runner import run_prepared_evaluation


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


def test_runner_writes_portable_artifacts(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.load_prepared_dataset",
        lambda *_: (_records(), {"sample_fingerprint": "dataset-sha"}),
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.load_model", lambda *_args, **_kwargs: {"quantization": {"verified": True}}
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.prepare_inputs", lambda *_args, **_kwargs: {"input_ids": object()}
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.generate", lambda *_args, **_kwargs: ['{"answer":"A","reasoning":"x"}']
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
        "evaluation_has_a_footprint.runner.load_prepared_dataset",
        lambda *_: (_records(), {"sample_fingerprint": "dataset-sha"}),
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.load_model", lambda *_args, **_kwargs: {"quantization": {"verified": True}}
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.prepare_inputs", lambda *_args, **_kwargs: {"input_ids": object()}
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.generate", lambda *_args, **_kwargs: ['{"answer":"A","reasoning":"x"}']
    )
    events: list[str] = []
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.start_measurement", lambda **_kwargs: events.append("start") or {}
    )
    monkeypatch.setattr(
        "evaluation_has_a_footprint.runner.stop_measurement",
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
