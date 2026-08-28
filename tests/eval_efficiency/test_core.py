"""Synthetic tests for portable public scientific utilities."""

import pytest

from eval_efficiency.datasets import (
    dataset_fingerprint,
    validate_rows,
)
from eval_efficiency.metrics import summary
from eval_efficiency.model_registry import MODELS
from eval_efficiency.parsing import parse_answer
from eval_efficiency.prompts import (
    BIAS_PRESENT_THRESHOLD,
    BIAS_SCORE_JUDGE_PROMPT,
    RELEVANCY_SCORE_JUDGE_PROMPT,
    bias_present,
    prompt_sha256,
)


def bbq_rows() -> list[dict[str, str]]:
    cells = (("ambig", "neg"), ("ambig", "nonneg"), ("disambig", "neg"), ("disambig", "nonneg"))
    return [
        {
            "category": "age",
            "case_id": "c1",
            "example_id": str(i),
            "question_index": "0",
            "context_condition": c,
            "question_polarity": p,
            "label": "0",
            "target_loc": "0",
            "context": "context",
            "question": "question",
            "ans0": "a",
            "ans1": "b",
            "ans2": "c",
            "answer_info": "{}",
            "subcategory": "sub",
            "stereotyped_groups": "[]",
            "label_type": "type",
            "known_stereotyped_groups": "[]",
            "known_stereotyped_race": "",
            "known_stereotyped_var2": "",
        }
        for i, (c, p) in enumerate(cells)
    ]


def test_dataset_membership_and_metrics() -> None:
    rows = bbq_rows()
    fingerprint = dataset_fingerprint("bbq", rows, category_order=["age"])
    validate_rows("bbq", rows)
    rows[0]["label"] = "1"
    assert dataset_fingerprint("bbq", rows, category_order=["age"]) != fingerprint
    with pytest.raises(ValueError):
        validate_rows("bbq", bbq_rows()[:-1])
    assert summary([{"correct": True}, {"correct": False}])["accuracy"] == 0.5


def test_parser_prompts_and_models() -> None:
    assert parse_answer('{"answer":"A","reasoning":"x"}') == ("A", "x", None)
    assert parse_answer("C. Cannot answer") == ("C", "Cannot answer", None)
    assert parse_answer("I think A is correct")[0] is None
    assert prompt_sha256(BIAS_SCORE_JUDGE_PROMPT) == "1947415bd6cb445e5cdd07d45c4a9a9cdbc2d952963a28fc6e1fc0fc0303d8dd"
    assert (
        prompt_sha256(RELEVANCY_SCORE_JUDGE_PROMPT)
        == "61391f03d5d96f7be23d2c7b059e1991d1239741f61f08be6bac9ec5d23e93d0"
    )
    assert BIAS_SCORE_JUDGE_PROMPT and RELEVANCY_SCORE_JUDGE_PROMPT
    assert BIAS_PRESENT_THRESHOLD == 1
    assert bias_present(None) is None and bias_present(1) is False and bias_present(2) is True
    assert len({item["hf_id"] for item in MODELS.values()}) == len(MODELS)
    assert MODELS["gemma4_12b"]["hf_id"] == "google/gemma-4-12B-it"
    assert MODELS["gemma4_12b"]["revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
