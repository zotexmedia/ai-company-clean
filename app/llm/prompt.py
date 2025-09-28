"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    Rules (apply in order):
    1) Strip legal suffixes: LLC, L.L.C., Inc, Incorporated, Ltd, Limited, LLP, L.L.P., PLLC, Corp, Corporation.
    2) Strip generic designators unless they are the only head word: Company, Co., Group, Firm, Associates, Partners, Holdings, Services, Solutions, Supplies, Enterprises.
    3) Keep exactly one industry head noun if present: Law, Dental, Packaging, Mortgage, Insurance, Accounting, Construction, Medical, Veterinary, Bookkeeping, Realty, Chiropractic, Exteriors, Painting, Automotive, Banking, Finance, Advisors, Consulting.
       • "Hess Law Firm" → canonical: "Hess Law"
       • "AmeriHome Mortgage" → "AmeriHome Mortgage"
    4) Domains: if a TLD/extension (.com, .net, .org, .expert, etc.) is present, remove only the extension and keep brand camel case intact.
       • "PerformanceCulture.Expert" → "PerformanceCulture"
    5) Locations/qualifiers: remove "of <City/State>", "at <Place>", city/state codes, "USA".
    6) Names ending with a generic designator:
       • canonical: drop the designator ("Wellington Group" → "Wellington")
       • sentence: prepend "the" and keep the designator ("Wellington Group" → "the Wellington Group")
    7) Preserve obvious stylization, punctuation, and symbols that are part of the brand (CamelCase, &, apostrophes, numerals). Collapse extra spaces.
    8) Do not invent words. If unsure, prefer keeping the industry head noun rather than removing it.

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
