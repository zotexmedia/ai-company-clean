"""Submit overnight batch jobs to the OpenAI Batch API."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from app.llm.prompt import build_conversation
from app.workers.llm_client import load_schema

BATCH_MODEL = "gpt-4o-mini"


def build_jsonl(records: Iterable[str], destination: Path) -> Path:
    schema = load_schema()
    with destination.open("w", encoding="utf-8") as handle:
        for idx, raw in enumerate(records):
            request = {
                "custom_id": f"norm-{idx}",
                "response": {
                    "model": BATCH_MODEL,
                    "input": build_conversation(raw),
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "CompanyCanon",
                            "schema": schema,
                        },
                    },
                },
            }
            handle.write(json.dumps(request))
            handle.write("\n")
    return destination


def submit_batch(records: Iterable[str]) -> str:
    client = OpenAI()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "payload.jsonl"
        build_jsonl(records, tmp_path)
        with tmp_path.open("rb") as fh:
            file = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="responses",
            completion_window="24h",
        )
        return batch.id


__all__ = ["submit_batch"]
