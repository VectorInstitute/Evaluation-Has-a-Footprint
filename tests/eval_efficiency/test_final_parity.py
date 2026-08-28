"""Focused regression coverage for final public scientific-parity fixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from eval_efficiency import cli, runner
from eval_efficiency.datasets import validate_rows
from eval_efficiency.inference import _quantization_config, generate, inspect_quantization, prepare_inputs
from eval_efficiency.parsing import parse_primary, recover_maintenance


def _bbqv_rows(image_sha256: str, image_file: str = "image.jpg") -> list[dict[str, str]]:
    rows = []
    for polarity, cross_id in (("neg", "cross-neg"), ("nonneg", "cross-nonneg")):
        for index in range(3):
            rows.append(
                {
                    "id": f"{polarity}-{index}",
                    "cross_id": cross_id,
                    "unique_question_id": "scenario-1",
                    "category": "age",
                    "subcategory": "sub",
                    "stereotyped_groups": "[]",
                    "question_polarity": polarity,
                    "context": "context",
                    "question": "question",
                    "ans0": "a",
                    "ans1": "b",
                    "ans2": "c",
                    "label": "0",
                    "gold_answer": "a",
                    "image_sha256": image_sha256,
                    "image_file": image_file,
                }
            )
    return rows


def test_bbqv_finalized_six_row_scenario_and_image_integrity(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    rows = _bbqv_rows(hashlib.sha256(b"image").hexdigest())
    validate_rows("bbq_v", rows, images_dir=tmp_path)
    missing_polarity = [dict(row) for row in rows]
    for row in missing_polarity:
        row["question_polarity"] = "neg"
    with pytest.raises(ValueError):
        validate_rows("bbq_v", missing_polarity, images_dir=tmp_path)
    missing_cross = [dict(row) for row in rows]
    for row in missing_cross:
        row["cross_id"] = "only-cross"
    with pytest.raises(ValueError):
        validate_rows("bbq_v", missing_cross, images_dir=tmp_path)
    duplicate_id = [dict(row) for row in rows] + [dict(rows[0])]
    with pytest.raises(ValueError):
        validate_rows("bbq_v", duplicate_id, images_dir=tmp_path)
    missing_field = [dict(row) for row in rows]
    del missing_field[0]["gold_answer"]
    with pytest.raises(ValueError):
        validate_rows("bbq_v", missing_field, images_dir=tmp_path)
    missing_image = [dict(row) for row in rows]
    missing_image[0]["image_file"] = "missing.jpg"
    with pytest.raises(ValueError):
        validate_rows("bbq_v", missing_image, images_dir=tmp_path)
    wrong_hash = [dict(row) for row in rows]
    wrong_hash[0]["image_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_rows("bbq_v", wrong_hash, images_dir=tmp_path)
    unsafe = [dict(row) for row in rows]
    unsafe[0]["image_file"] = "../image.jpg"
    with pytest.raises(ValueError):
        validate_rows("bbq_v", unsafe, images_dir=tmp_path)


@pytest.mark.parametrize("raw", ['prefix "answer":"B', '{"foo":"two"', '{"answer":"A explanation'])
def test_recovery_rejects_nonleading_or_incomplete_json(raw: str) -> None:
    assert recover_maintenance(raw, ("one", "two", "three")).answer is None


def test_recovery_matches_constrained_final_rules() -> None:
    assert recover_maintenance('{"answer":"B"', ("one", "two", "three")).answer == "B"
    assert recover_maintenance('{"answer":"two"}', ("one", "two", "three")).answer == "B"
    assert recover_maintenance('{"answer":"A","answer":"B"', ("one", "two", "three")).answer is None
    assert recover_maintenance("Answer: B because", ("one", "two", "three")).answer == "B"
    assert recover_maintenance("Answer: D", ("one", "two", "three")).answer is None


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        ('{"answer":"A","reasoning":"r"}', "A"),
        ('{"answer":"B","reasoning":"r"}', "B"),
        ('{"answer":"C","reasoning":"r"}', "C"),
        ('```json\n{"answer":"A","reasoning":"r"}\n```', "A"),
        ("A. explanation", "A"),
        ("B. explanation", "B"),
        ("C. explanation", "C"),
        ("Answer: B because", "B"),
    ],
)
def test_primary_parser_final_accepted_forms(raw: str, answer: str) -> None:
    assert parse_primary(raw).answer == answer


@pytest.mark.parametrize(
    "raw",
    [
        "D. explanation",
        "Answer: D because",
        "a. lowercase",
        "The answer is B",
        "",
        None,
    ],
)
def test_primary_parser_final_rejected_forms(raw: str | None) -> None:
    assert parse_primary(raw).answer is None


class _Config:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Transformers:
    BitsAndBytesConfig = _Config


class _Torch:
    bfloat16 = "bf16"


def test_quantization_configs_and_failure_verification() -> None:
    assert _quantization_config(_Torch, _Transformers, "int8").kwargs == {"load_in_8bit": True}
    assert _quantization_config(_Torch, _Transformers, "int4").kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bf16",
    }

    class Unquantized:
        def named_modules(self) -> list[tuple[str, object]]:
            return []

        def parameters(self) -> list[object]:
            return []

    with pytest.raises(RuntimeError, match="INT8"):
        inspect_quantization(Unquantized(), "int8")
    with pytest.raises(RuntimeError, match="INT4"):
        inspect_quantization(Unquantized(), "int4")


def test_generation_and_chat_template_settings() -> None:
    calls: dict[str, Any] = {}

    class Inputs(dict[str, Any]):
        def to(self, device: str) -> "Inputs":
            calls["device"] = device
            return self

    class Processor:
        def apply_chat_template(self, message: object, **kwargs: Any) -> str:
            calls["template"] = kwargs
            return "prompt"

        def __call__(self, **kwargs: Any) -> Inputs:
            calls["processor"] = kwargs
            return Inputs(input_ids=type("Tokens", (), {"shape": (1, 2)})())

        def batch_decode(self, continuations: object, **kwargs: Any) -> list[str]:
            calls["decode"] = kwargs
            return ["output"]

    class Parameters:
        device = "device"

    class Generated:
        def __getitem__(self, item: object) -> str:
            calls["continuation"] = item
            return "continuation"

    class Model:
        def parameters(self) -> Any:
            return iter((Parameters(),))

        def generate(self, **kwargs: Any) -> Generated:
            calls["generate"] = kwargs
            return Generated()

    processor = Processor()
    bundle = {"processor": processor, "model": Model()}
    inputs = prepare_inputs(bundle, [[{"role": "user", "content": [{"type": "text", "text": "x"}]}]])
    assert calls["template"] == {"tokenize": False, "add_generation_prompt": True}
    assert calls["processor"] == {"text": ["prompt"], "padding": True, "return_tensors": "pt"}
    assert generate(bundle, inputs) == ["output"]
    assert calls["generate"] == {"input_ids": inputs["input_ids"], "max_new_tokens": 128, "do_sample": False}
    assert calls["continuation"] == (slice(None, None, None), slice(2, None, None))
    assert calls["decode"] == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}


def test_gemma4_chat_template_disables_thinking() -> None:
    calls: dict[str, Any] = {}

    class Inputs(dict[str, Any]):
        def to(self, device: str) -> "Inputs":
            return self

    class Processor:
        def apply_chat_template(self, messages: object, **kwargs: Any) -> Inputs:
            calls["template"] = kwargs
            return Inputs(input_ids=type("Tokens", (), {"shape": (1, 2)})())

    class Parameters:
        device = "device"

    class Model:
        def parameters(self) -> Any:
            return iter((Parameters(),))

    bundle = {
        "spec": {"family": "gemma4_unified"},
        "processor": Processor(),
        "model": Model(),
    }
    prepare_inputs(bundle, [[{"role": "user", "content": [{"type": "text", "text": "x"}]}]])
    assert calls["template"]["tokenize"] is True
    assert calls["template"]["enable_thinking"] is False
    assert calls["template"]["processor_kwargs"] == {"padding": True}
    with pytest.raises(RuntimeError, match="qualified only for Qwen"):
        prepare_inputs(
            bundle,
            [[{"role": "user", "content": [{"type": "text", "text": "x"}]}]],
            max_pixels=1024,
        )


def test_cli_runs_mocked_portable_artifact_flow(monkeypatch: Any, tmp_path: Path) -> None:
    records = [
        {
            "sample_id": "age:0",
            "group_id": "age:c1",
            "category": "age",
            "subcategory": "sub",
            "question_polarity": "neg",
            "context_condition": "ambig",
            "context": "c",
            "question": "q",
            "ans0": "a",
            "ans1": "b",
            "ans2": "c",
            "label": "0",
        }
    ]
    monkeypatch.setattr(runner, "load_prepared_dataset", lambda *_: (records, {"sample_fingerprint": "sha"}))
    monkeypatch.setattr(runner, "load_model", lambda *_args, **_kwargs: {"quantization": {"verified": True}})
    monkeypatch.setattr(runner, "prepare_inputs", lambda *_args, **_kwargs: {"input_ids": object()})
    monkeypatch.setattr(runner, "generate", lambda *_args, **_kwargs: ['{"answer":"A","reasoning":"x"}'])
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "--model",
                "qwen25_vl_7b",
                "--dataset",
                "bbq",
                "--condition",
                "M0",
                "--prepared-dataset",
                str(prepared),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "run.json").read_text())["generation"]["max_new_tokens"] == 128
    assert (output / "predictions.jsonl").is_file() and (output / "metrics.json").is_file()


def test_historical_exception_is_documented() -> None:
    text = Path("REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "artifact-specific" in text and "byte-for-byte" in text
