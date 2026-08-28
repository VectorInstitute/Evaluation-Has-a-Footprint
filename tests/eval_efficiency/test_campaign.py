"""Offline parity tests for accepted-campaign public behavior."""

from __future__ import annotations

import hashlib

import pytest

from evaluation_has_a_footprint.conditions import QWEN3_BBQV_MAX_PIXELS, resolve_condition
from evaluation_has_a_footprint.inference import SMOKE_PROMPT, SMOKE_PROMPT_SHA256, build_messages
from evaluation_has_a_footprint.parsing import parse_primary, recover_maintenance


@pytest.mark.parametrize(
    ("model", "dataset", "condition", "expected"),
    [
        (
            model,
            dataset,
            condition,
            8
            if condition in {"M1", "M5a"} and not (model == "qwen3_vl_30b_a3b" and dataset == "bbq_v")
            else 2
            if condition in {"M1", "M5a"}
            else 1,
        )
        for model in ("qwen25_vl_7b", "qwen3_vl_30b_a3b")
        for dataset in ("bbq", "bbq_v")
        for condition in ("M0", "M1", "M2", "M3")
    ],
)
def test_effective_full_condition_matrix(model: str, dataset: str, condition: str, expected: int) -> None:
    assert resolve_condition(condition, model_key=model, dataset=dataset).batch_size == expected


@pytest.mark.parametrize("model", ("qwen25_vl_7b", "qwen3_vl_30b_a3b"))
@pytest.mark.parametrize("dataset,rows", (("bbq", 1000), ("bbq_v", 1008)))
@pytest.mark.parametrize("condition", ("M4", "M5a", "M5b"))
def test_effective_subset_condition_matrix(model: str, dataset: str, rows: int, condition: str) -> None:
    profile = resolve_condition(condition, model_key=model, dataset=dataset, subset_rows=rows)
    expected = (
        8
        if condition == "M5a" and not (model == "qwen3_vl_30b_a3b" and dataset == "bbq_v")
        else 2
        if condition == "M5a"
        else 1
    )
    assert profile.batch_size == expected


@pytest.mark.parametrize("dataset", ("bbq", "bbq_v"))
def test_gemma4_uses_accepted_batch_four(dataset: str) -> None:
    rows = 1000 if dataset == "bbq" else 1008
    assert resolve_condition("M0", model_key="gemma4_12b", dataset=dataset).batch_size == 1
    assert resolve_condition("M1", model_key="gemma4_12b", dataset=dataset).batch_size == 4
    profile = resolve_condition("M5a", model_key="gemma4_12b", dataset=dataset, subset_rows=rows)
    assert profile.batch_size == 4


def test_gemma4_rejects_pixel_override() -> None:
    with pytest.raises(ValueError, match="not qualified for Gemma"):
        resolve_condition("M0", model_key="gemma4_12b", dataset="bbq_v", max_pixels_override=1024)


@pytest.mark.parametrize("condition", ("M5a", "M5b"))
def test_m5_reuses_exact_replica_one(condition: str) -> None:
    profile = resolve_condition(condition, model_key="qwen25_vl_7b", dataset="bbq", subset_rows=1000)
    assert (profile.subset_rows, profile.replicate) == (1000, 1)
    with pytest.raises(ValueError):
        resolve_condition(condition, model_key="qwen25_vl_7b", dataset="bbq", subset_rows=1000, replicate=2)


def test_pixel_profile_is_model_dataset_specific() -> None:
    assert resolve_condition("M0", model_key="qwen3_vl_30b_a3b", dataset="bbq_v").max_pixels == QWEN3_BBQV_MAX_PIXELS
    assert resolve_condition("M0", model_key="qwen25_vl_7b", dataset="bbq_v").max_pixels is None
    assert resolve_condition("M0", model_key="qwen3_vl_30b_a3b", dataset="bbq").max_pixels is None


@pytest.mark.parametrize(
    "raw,answer",
    [
        ("A. reason", "A"),
        ("B. reason", "B"),
        ("C. reason", "C"),
        ('{"answer":"A","reasoning":"r"}', "A"),
        ("Answer: B because", "B"),
    ],
)
def test_final_primary_parser_accepts_only_supported_abc_forms(raw: str, answer: str) -> None:
    assert parse_primary(raw).answer == answer


@pytest.mark.parametrize(
    "raw", ["D. reason", "d. reason", "Answer: D", "I think A", '{"answer":"D","reasoning":"r"}', "", None]
)
def test_final_primary_parser_rejects_unsupported_forms(raw: str | None) -> None:
    assert parse_primary(raw).answer is None


def test_maintenance_recovery_is_constrained() -> None:
    assert recover_maintenance('{"answer":"B"', ("one", "two", "three")).answer == "B"
    assert recover_maintenance('{"answer":"two"}', ("one", "two", "three")).answer == "B"
    assert recover_maintenance("The answer is B", ("one", "two", "three")).answer is None


def test_message_modalities_and_prompt_hash() -> None:
    row = {"context": "c", "question": "q", "ans0": "a", "ans1": "b", "ans2": "c", "image_path": "/tmp/image.jpg"}
    assert build_messages(row, "bbq")[0]["content"][0]["type"] == "text"
    assert build_messages(row, "bbq_v")[0]["content"][0]["type"] == "image"
    assert SMOKE_PROMPT_SHA256 == "b1da1b08a03abe5847519a84bc07c5aeb42993abeec82adb86f2a4674871d0cf"
    assert hashlib.sha256(SMOKE_PROMPT.encode()).hexdigest() == SMOKE_PROMPT_SHA256
