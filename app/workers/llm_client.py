"""Thin wrapper around the OpenAI Responses API with Structured Outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI

from app.llm.prompt import build_conversation

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "llm" / "structured_schema.json"


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


class LLMCallError(RuntimeError):
    pass


def _extract_structured(message: Any) -> Dict[str, Any]:
    """Best-effort extraction of JSON payload from SDK response."""
    content = getattr(message, "content", None)
    if not content:
        raise LLMCallError("Message missing content")
    
    # For structured outputs, content should be a string with JSON
    if isinstance(content, str):
        return json.loads(content)
    
    # Fallback for other content formats
    if hasattr(content, "text"):
        return json.loads(content.text)
    
    raise LLMCallError("Unable to extract JSON from response message")


def normalize_batch_gpt4o_mini(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call OpenAI once per item (placeholder until multi-input batching is GA)."""
    schema = load_schema()
    results: List[Dict[str, Any]] = []

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "CompanyCanon",
            "schema": schema,
            "strict": True,
        },
    }

    for item in items:
        retry_suffix = item.get("retry_suffix") if isinstance(item, dict) else None
        conversation = build_conversation(item["raw_name"], retry_suffix=retry_suffix)
        logger.debug("Invoking OpenAI for id=%s", item["id"])
        rsp = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            temperature=0,
            response_format=response_format,
        )
        if not rsp.choices:
            raise LLMCallError("Response missing choices")
        payload = _extract_structured(rsp.choices[0].message)
        results.append(
            {
                "id": item["id"],
                "raw_name": item["raw_name"],
                "payload": payload,
                "usage": getattr(rsp, "usage", None),
            }
        )
    return results


__all__ = ["normalize_batch_gpt4o_mini", "LLMCallError", "get_client", "load_schema"]
