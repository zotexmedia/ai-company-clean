"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    You normalize English company names.
    Ignore legal suffixes and locations. Keep meaningful brand words and industry terms if they distinguish brands.
    Only return valid JSON that matches the provided JSON Schema.
    """
).strip()

USER_TEMPLATE = dedent(
    """
    RAW: "{raw_name}"
    CONTEXT:
    - Language is always English.
    - Ignore legal suffixes: inc, incorporated, co, corp, ltd, llc, llp, plc, gmbh, srl, s.a., bv, nv, ab, oy, as, kk, pty ltd, sdn bhd.
    - Ignore location qualifiers: city, state, country, "HQ", directional suffixes, zip codes.
    - Keep industry words if they differentiate sibling brands, for example "Acme Cleaning" vs "Acme Pest".
    Return only the canonical form used most commonly by customers.
    """
).strip()

# 14 diverse examples mirroring high-signal edge cases. Kept short so they travel with the app.
FEW_SHOTS: List[Tuple[str, dict]] = [
    ("ACME FACILITY SERVICES, INC.", {"canonical": "Acme Facility Services", "is_new": False, "confidence": 0.95, "reason": "Identical to existing brand"}),
    ("Acme Cleaning LLC - Dallas", {"canonical": "Acme Cleaning", "is_new": False, "confidence": 0.92, "reason": "Same brand location branch"}),
    ("The Sterling Group Holdings, LLC", {"canonical": "Sterling Group", "is_new": False, "confidence": 0.88, "reason": "Drop legal tail"}),
    ("BrightWave Solar Ltd (Austin HQ)", {"canonical": "BrightWave Solar", "is_new": False, "confidence": 0.9, "reason": "Remove HQ location"}),
    ("Northstar Pest Control of Tampa", {"canonical": "Northstar Pest Control", "is_new": False, "confidence": 0.87, "reason": "Ignore city"}),
    ("Blue Horizon Marine Services Pty Ltd", {"canonical": "Blue Horizon Marine Services", "is_new": False, "confidence": 0.9, "reason": "Drop Pty Ltd suffix"}),
    ("Acme Facility Care (West Region)", {"canonical": "Acme Facility Care", "is_new": False, "confidence": 0.84, "reason": "Regional label ignored"}),
    ("NovaTech Robotics GmbH", {"canonical": "NovaTech Robotics", "is_new": False, "confidence": 0.93, "reason": "GmbH legal form ignored"}),
    ("Atlas Industrial Cleaning & Co.", {"canonical": "Atlas Industrial Cleaning", "is_new": False, "confidence": 0.89, "reason": "Trailing Co removed"}),
    ("GreenLeaf Landscaping - Phoenix East", {"canonical": "GreenLeaf Landscaping", "is_new": False, "confidence": 0.86, "reason": "Discard directional"}),
    ("Precision Analytics (New)", {"canonical": "Precision Analytics", "is_new": False, "confidence": 0.8, "reason": "Marketing tag removed"}),
    ("Summit Bio Labs, Inc.", {"canonical": "Summit Bio Labs", "is_new": False, "confidence": 0.9, "reason": "Strip Inc"}),
    ("Coastal Pest & Lawn LLC", {"canonical": "Coastal Pest & Lawn", "is_new": False, "confidence": 0.88, "reason": "Keep industry differentiator"}),
    ("Valor Security Solutions (US)", {"canonical": "Valor Security Solutions", "is_new": False, "confidence": 0.85, "reason": "Country qualifier ignored"}),
]


def build_user_message(raw_name: str, retry_suffix: str | None = None) -> str:
    """Render the user template for a raw company name."""
    message = USER_TEMPLATE.format(raw_name=raw_name)
    if retry_suffix:
        message = f"{message}\n\n{retry_suffix.strip()}"
    return message


def few_shot_messages() -> Iterable[dict]:
    """Yield interleaved user/assistant messages for the example pairs."""
    for raw, payload in FEW_SHOTS:
        yield {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_message(raw)}
            ],
        }
        yield {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": json.dumps(payload, separators=(",", ":"))
                }
            ],
        }


def build_conversation(raw_name: str, retry_suffix: str | None = None) -> List[dict]:
    """Construct the message list for a single normalization call."""
    messages: List[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *few_shot_messages(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_message(raw_name, retry_suffix=retry_suffix)}
            ],
        },
    ]
    return messages
