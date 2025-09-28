"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    Normalize company names to their core business identifiers. Return only valid JSON.

    ## Pattern Recognition (evaluate all patterns, then apply highest priority match):

    ### PRIORITY 1: Personal Professional Practices
    Pattern: [Personal Name] + [Professional Designator] + [Any Legal Suffix]
    - Designators: & Associates, Partners, Law Firm, Law Office, Law Group
    - Action: Extract ONLY the personal name
    - Examples: 
      - "John Smith & Associates LLC" → "John Smith"
      - "Mary Chen Law Firm, PC" → "Mary Chen"

    ### PRIORITY 2: Standard Business Names
    Pattern: [Brand] + [Industry Noun] + [Generic Designator] + [Legal Suffix]
    - Action: Keep brand + industry noun, remove designators and suffixes
    - Examples:
      - "Acme Software Solutions Inc" → "Acme Software"
      - "Henderson Dental Associates PLLC" → "Henderson Dental"

    ### PRIORITY 3: Branded Groups/Companies
    Pattern: [Brand] + [Group/Partners/Company] + [Legal Suffix]
    - Action: Keep brand + designator (when no industry noun exists)
    - Examples:
      - "The Wellington Group Inc" → "The Wellington Group"
      - "Anderson Partners LLC" → "Anderson Partners"

    ### PRIORITY 4: Simple Names
    Pattern: [Brand] + [Legal Suffix only]
    - Action: Remove legal suffix only
    - Examples:
      - "Microsoft Corporation" → "Microsoft"
      - "Google LLC" → "Google"

    ## Universal Cleanup Rules (apply to final result):
    - Remove geographic modifiers: "of [Place]", state codes, parentheticals
    - Remove domain extensions: .com, .net, .org, etc.
    - Preserve existing "The" but never add it
    - Maintain original capitalization and ampersands

    Output format: {"canonical": "[normalized name]"}
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
