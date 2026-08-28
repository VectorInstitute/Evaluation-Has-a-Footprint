"""Finalized A/B/C parsing and deterministic maintenance recovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


VALID_ANSWERS = frozenset({"A", "B", "C"})
_OPTION_PREFIX = re.compile(r"^([ABC])\b")
_DOT = re.compile(r"^([ABC])\.\s+(.+)", re.DOTALL)
_ANSWER_LINE = re.compile(r"^Answer:\s*([ABC])\b")
_LEADING_JSON_ANSWER = re.compile(r'^\{\s*"answer"\s*:\s*"([ABC])"')
_ANY_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"([ABC])"')
_RECOVERY_LEADING_ANSWER = re.compile(r'^\{\s*"answer"\s*:\s*"([^"]*)"')
_RECOVERY_ANY_ANSWER = re.compile(r'"answer"\s*:\s*"([^"]*)"')
_COMPLETE_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    """One strictly parsed or deterministic-recovery answer."""

    answer: str | None
    reasoning: str | None
    error: str | None
    method: str | None


def _unfence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0] in {"```", "```json"} and lines[-1] == "```":
        return "\n".join(lines[1:-1])
    return text


def _malformed_leading_answer(text: str) -> str | None:
    """Recover only an intact, consistent leading JSON answer field."""
    if not text.startswith("{"):
        return None
    leading = _LEADING_JSON_ANSWER.match(text)
    if leading is None:
        return None
    matches = _ANY_JSON_ANSWER.findall(text)
    return leading.group(1) if len(set(matches)) == 1 else None


def _parse_non_json(text: str) -> ParsedAnswer:
    dot = _DOT.fullmatch(text)
    if dot is not None:
        return ParsedAnswer(dot.group(1), dot.group(2), None, "leading_letter_dot")
    answer_line = _ANSWER_LINE.match(text)
    if answer_line is not None:
        return ParsedAnswer(answer_line.group(1), None, None, "answer_line")
    first_line = text.splitlines()[0].strip()
    if first_line in VALID_ANSWERS:
        return ParsedAnswer(first_line, None, None, "first_line_letter")
    answer = _malformed_leading_answer(text)
    return (
        ParsedAnswer(answer, None, None, "malformed_json_leading_answer")
        if answer is not None
        else ParsedAnswer(None, None, "malformed_json", None)
    )


def _parse_json_value(value: object) -> ParsedAnswer:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("answer"), str)
        or not isinstance(value.get("reasoning"), str)
    ):
        return ParsedAnswer(None, None, "invalid_json_schema", None)
    answer = _OPTION_PREFIX.match(value["answer"].strip())
    if answer is None:
        return ParsedAnswer(None, None, "invalid_option", None)
    return ParsedAnswer(answer.group(1), value["reasoning"], None, "strict_json")


def parse_primary(raw: str | None) -> ParsedAnswer:
    """Use the final run's strict JSON-first, fail-closed A/B/C parser."""
    if not isinstance(raw, str) or not raw.strip():
        return ParsedAnswer(None, None, "malformed_json", None)
    text = _unfence(raw.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return _parse_non_json(text)
    return _parse_json_value(value)


def _strip_leading_fence(raw: str) -> str:
    lines = raw.strip().splitlines()
    return "\n".join(lines[1:]) if lines and lines[0].strip() in {"```", "```json"} else raw.strip()


def _option_match(answer_text: str, options: tuple[str, str, str]) -> str | None:
    matches = [index for index, option in enumerate(options) if option == answer_text]
    return "ABC"[matches[0]] if len(matches) == 1 else None


def _recover_complete_object(text: str, options: tuple[str, str, str], primary: ParsedAnswer) -> ParsedAnswer | None:
    object_match = _COMPLETE_OBJECT.search(text)
    if object_match is None:
        return None
    try:
        value = json.loads(object_match.group(0))
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        return primary
    answer_text = value["answer"].strip()
    leading = _OPTION_PREFIX.match(answer_text)
    reasoning = value.get("reasoning") if isinstance(value.get("reasoning"), str) else None
    if leading is not None:
        return ParsedAnswer(leading.group(1), reasoning, None, "maintenance_complete_json")
    option = _option_match(answer_text, options)
    return ParsedAnswer(option, reasoning, None, "maintenance_option_text") if option is not None else primary


def _recover_truncated_object(text: str, options: tuple[str, str, str]) -> ParsedAnswer | None:
    leading = _RECOVERY_LEADING_ANSWER.match(text)
    if leading is None or len(set(_RECOVERY_ANY_ANSWER.findall(text))) != 1:
        return None
    answer_text = leading.group(1).strip()
    letter = _OPTION_PREFIX.match(answer_text)
    if letter is not None:
        return ParsedAnswer(letter.group(1), None, None, "maintenance_truncated_json")
    option = _option_match(answer_text, options)
    return ParsedAnswer(option, None, None, "maintenance_truncated_option_text") if option is not None else None


def recover_maintenance(raw: str | None, options: tuple[str, str, str]) -> ParsedAnswer:
    """Apply only the documented final-campaign deterministic recovery rules."""
    primary = parse_primary(raw)
    if primary.error is None or not isinstance(raw, str):
        return primary
    text = _strip_leading_fence(raw)
    complete = _recover_complete_object(text, options, primary)
    if complete is not None:
        return complete
    return _recover_truncated_object(text, options) or primary


def parse_answer(raw: str | None) -> tuple[str | None, str | None, str | None]:
    """Compatibility wrapper exposing the primary parser tuple."""
    parsed = parse_primary(raw)
    return parsed.answer, parsed.reasoning, parsed.error
