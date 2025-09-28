"""Thin wrapper around the OpenAI Responses API with Structured Outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence
import concurrent.futures
import threading

from openai import OpenAI

from app.llm.prompt import build_conversation

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-5"  # GPT-5: better quality for ultra-simple prompts
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


def _process_single_item(item: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item through the LLM."""
    retry_suffix = item.get("retry_suffix") if isinstance(item, dict) else None
    conversation = build_conversation(item["raw_name"], retry_suffix=retry_suffix)
    logger.debug("Invoking OpenAI for id=%s", item["id"])
    
    if MODEL_NAME.startswith("gpt-5"):
        # Use Responses API for GPT-5 models
        text_format = {
            "type": "json_schema",
            "name": "CompanyCanon",
            "schema": schema,
            "strict": True,
        }
        
        rsp = get_client().responses.create(
            model=MODEL_NAME,
            input=conversation,
            text={"format": text_format},
        )
        
        # Parse response from Responses API
        if not rsp.output:
            raise LLMCallError("Response missing output")
            
        # Find the message with JSON content
        for output_item in rsp.output:
            if output_item.type == "message" and output_item.role == "assistant":
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        payload = json.loads(content_item.text)
                        return {
                            "id": item["id"],
                            "raw_name": item["raw_name"],
                            "payload": payload,
                            "usage": getattr(rsp, "usage", None),
                        }
        
        raise LLMCallError("No valid JSON content found in response")
    else:
        # Use Chat Completions API for GPT-4 models
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "CompanyCanon",
                "schema": schema,
                "strict": True,
            },
        }
        
        rsp = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            temperature=0,
            response_format=response_format,
        )
        
        if not rsp.choices:
            raise LLMCallError("Response missing choices")
        
        payload = _extract_structured(rsp.choices[0].message)
        
        return {
            "id": item["id"],
            "raw_name": item["raw_name"],
            "payload": payload,
            "usage": getattr(rsp, "usage", None),
        }

def normalize_batch_gpt4o_mini(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call OpenAI with concurrent processing for better performance."""
    schema = load_schema()
    
    # For small batches, process sequentially to avoid overhead
    if len(items) <= 5:
        results = []
        for item in items:
            results.append(_process_single_item(item, schema))
        return results
    
    # For larger batches, use concurrent processing with conservative limits
    max_workers = min(5, len(items))  # Conservative limit to avoid rate limiting
    results = [None] * len(items)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all items
        future_to_index = {
            executor.submit(_process_single_item, item, schema): i
            for i, item in enumerate(items)
        }
        
        # Collect results in original order
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.error("Error processing item %d (%s): %s", index, items[index]["raw_name"], str(e))
                # Create error result that will be handled by retry logic
                results[index] = {
                    "id": items[index]["id"],
                    "raw_name": items[index]["raw_name"],
                    "payload": None,
                    "usage": None,
                }
    
    return [r for r in results if r is not None]


__all__ = ["normalize_batch_gpt4o_mini", "LLMCallError", "get_client", "load_schema"]
