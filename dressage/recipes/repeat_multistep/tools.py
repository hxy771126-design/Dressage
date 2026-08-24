"""Shared helpers for the repeat multi-step recipe."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """\
You are a deterministic copy engine.

Follow the current repeat instruction exactly. Output only the requested text,
without explanations, delimiters, or additional formatting.
"""

REPEAT_INSTRUCTION = """\
Repeat your immediately preceding assistant response exactly.
Output only that response. Do not add, remove, or change any character.
"""


def extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        return {"role": "assistant", "content": ""}
    return dict(message)
