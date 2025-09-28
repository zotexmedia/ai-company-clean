"""Fetch completed Batch API outputs and merge to Postgres."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from app.llm.postprocess import apply_guardrails
from app.stores.db import upsert_alias_result


def download_batch_results(batch_id: str, destination: Path) -> Path:
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status not in {"completed", "failed", "cancelled"}:
        raise RuntimeError(f"Batch {batch_id} not ready (status={batch.status})")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} missing output file reference")

    file_content = client.files.content(batch.output_file_id)
    destination.write_bytes(file_content.read())
    return destination


def ingest_results(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if "response" not in entry:
                continue
            raw = entry.get("input", {}).get("RAW", "")
            payload = entry["response"].get("output", [{}])[0].get("content", [{}])[0].get("text")
            if not payload:
                continue
            result = apply_guardrails(raw, json.loads(payload))
            upsert_alias_result(raw, result)


__all__ = ["download_batch_results", "ingest_results"]
