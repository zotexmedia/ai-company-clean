"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    Normalize company names to core business identifiers. Output JSON only.

    Rules:
    1. Convert ALL CAPS to Title Case (except IBM, AT&T, NASA)
    2. Remove legal suffixes: LLC, Inc, Corp, Ltd, LP, PC, etc.
    3. Remove generic terms after names: "and Associates", "Law Firm", "Dental Studio", "Services", "Group"
    4. Keep "The" prefix if present
    5. Keep integral brand words: "Systems" in "Cisco Systems", "International" in brand names
    6. Extract personal names: "John Smith and Associates" → "John Smith"

    Return: {"canonical": "Brand Name", "canonical_with_article": "Brand Name", "article_policy": "none|optional|official", "is_new": false, "confidence": 0.95, "reason": "brief explanation"}
    """
).strip()

USER_TEMPLATE = dedent(
    """
    RAW: "{raw_name}"
    
    Normalize this company name. Return JSON only.
    """
).strip()

FEW_SHOTS: List[Tuple[str, dict]] = []


# Convert Python booleans in FEW_SHOTS to JSON-friendly strings when serializing
for idx, (raw, payload) in enumerate(FEW_SHOTS):
    FEW_SHOTS[idx] = (raw, json.loads(json.dumps(payload)))


def build_user_message(raw_name: str, retry_suffix: str | None = None) -> str:
    message = USER_TEMPLATE.format(raw_name=raw_name)
    if retry_suffix:
        message = f"{message}\n\n{retry_suffix.strip()}"
    return message


def few_shot_messages() -> Iterable[dict]:
    for raw, payload in FEW_SHOTS:
        yield {
            "role": "user",
            "content": build_user_message(raw),
        }
        yield {
            "role": "assistant",
            "content": json.dumps(payload, separators=(",", ":")),
        }


def build_conversation(raw_name: str, retry_suffix: str | None = None) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *few_shot_messages(),
        {
            "role": "user",
            "content": build_user_message(raw_name, retry_suffix=retry_suffix),
        },
    ]
