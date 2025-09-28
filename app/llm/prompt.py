"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    Normalize company names to core business identifiers. Output JSON only.

    Rules (apply in order):
    1) Remove any corporate/legal suffixes or professional entity designators (e.g., Inc, LLC, Ltd, LLP, PLLC, PC, Corp, PLC, GmbH, S.A., BV, NV, Oy, etc.), including punctuated or localized variants. Treat this list as non-exhaustive and remove equivalent forms by reasoning.
    2) Remove location/qualifiers and parentheticals: phrases like "of <City/State>", "at <Place>", trailing state/country codes, "USA", and any text in parentheses.
    3) Domains: if a TLD/extension (.com, .net, .org, .expert, etc.) is present, remove only the extension; do not split internal CamelCase or brand tokens.
    4) Keep the core brand and keep exactly one meaningful industry head noun if present (e.g., Law, Dental, Packaging, Mortgage, Insurance, Accounting, Consulting, Advisors, Realty, Chiropractic, Exteriors, Painting, Automotive, Banking, Finance, Wealth Management, Asset Management, Investment Management, Report). Prefer keeping the head noun over removing it if uncertain.
    5) Designators like Group, Partners, Firm, Associates, Company/Co.:
       • If they are integral to how the brand is known and no other head noun remains (e.g., Wellington Group, Martin Group), keep them.
       • If they appear as trailing fluff after a stronger head noun (e.g., "Hess Law Firm"), drop the designator and keep the head noun ("Hess Law").
    6) Article rule (sentence readability): if the final word is Group, Partners, Firm, Associates, or Company/Co. and it is kept per rule 5, prepend "the" unless the name already begins with "The".
    7) Cleanup: preserve obvious stylization (CamelCase, numerals, &), normalize commas/spaces, and avoid adding or inventing words.

    Return: {"canonical": "Brand Name", "canonical_with_article": "Brand Name", "article_policy": "none|optional|official", "is_new": false}
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
