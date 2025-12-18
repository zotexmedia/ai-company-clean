"""Worker pipeline for company-name normalization."""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from jsonschema import Draft7Validator, ValidationError

from app.api.schemas import CanonicalResult, NormalizeRecord, NormalizeResponseItem
from app.llm.postprocess import GuardrailResult, apply_guardrails, redact_for_logging
from app.stores.cache import cache_get, cache_set, exact_cache_key, near_dupe_lookup
from app.stores.db import JobRun, JobStatus, get_job, record_job, set_job_status, upsert_alias_result
from app.workers.llm_client import LLMCallError, normalize_batch_gpt4o_mini, load_schema

logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))  # Optimized for GPT-5 Mini rate limits
INVALID_SUFFIX = "Invalid JSON, return valid JSON only."


@dataclass
class PendingItem:
    id: str
    raw_name: str
    source: str
    cache_key: str
    retry_suffix: Optional[str] = None


def batched(iterable: Sequence[NormalizeRecord], size: int) -> Iterable[Sequence[NormalizeRecord]]:
    for start in range(0, len(iterable), size):
        yield iterable[start:start + size]


class NormalizationService:
    def __init__(self, batch_size: int = BATCH_SIZE) -> None:
        self.batch_size = batch_size
        self.validator = Draft7Validator(load_schema())

    async def process_online(self, records: Sequence[NormalizeRecord]) -> Tuple[List[NormalizeResponseItem], List[str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_records, records, None)

    def _process_records(
        self,
        records: Sequence[NormalizeRecord],
        job: Optional[JobRun],
    ) -> Tuple[List[NormalizeResponseItem], List[str]]:
        responses: List[NormalizeResponseItem] = []
        errors: List[str] = []

        for chunk in batched(records, self.batch_size):
            chunk_responses, chunk_errors = self._process_chunk(chunk, job)
            responses.extend(chunk_responses)
            errors.extend(chunk_errors)
        return responses, errors

    def _process_chunk(
        self,
        chunk: Sequence[NormalizeRecord],
        job: Optional[JobRun],
    ) -> Tuple[List[NormalizeResponseItem], List[str]]:
        outputs: List[NormalizeResponseItem] = []
        errors: List[str] = []
        pending: List[PendingItem] = []
        lookup_by_id: Dict[str, NormalizeRecord] = {}
        cache_key_by_id: Dict[str, str] = {}

        # Cache metrics
        metrics = {"cache_hits": 0, "near_dupe_hits": 0, "llm_calls": 0, "batch_dedup": 0}

        # Track seen cache keys within this batch for deduplication
        seen_in_batch: Dict[str, str] = {}  # cache_key -> first record's id
        deferred_duplicates: List[Tuple[NormalizeRecord, str]] = []  # (record, cache_key)

        for record in chunk:
            cache_key = exact_cache_key(record.raw_name)

            # Check if duplicate within same batch (dedup before cache check)
            if cache_key in seen_in_batch:
                metrics["batch_dedup"] += 1
                deferred_duplicates.append((record, cache_key))
                continue

            cached_payload = cache_get(cache_key)
            if cached_payload:
                metrics["cache_hits"] += 1
                guard = apply_guardrails(record.raw_name, cached_payload)
                outputs.append(self._to_response(record, guard, cached=True))
                seen_in_batch[cache_key] = record.id
                # upsert_alias_result(record.raw_name, guard, record.source, getattr(job, "id", None))  # Disabled for performance
                continue

            near_payload = near_dupe_lookup(record.raw_name)
            if near_payload:
                metrics["near_dupe_hits"] += 1
                guard = apply_guardrails(record.raw_name, near_payload)
                cache_set(cache_key, near_payload)
                outputs.append(self._to_response(record, guard, cached=True))
                seen_in_batch[cache_key] = record.id
                # upsert_alias_result(record.raw_name, guard, record.source, getattr(job, "id", None))  # Disabled for performance
                continue

            # Mark this cache_key as seen (will be processed by LLM)
            seen_in_batch[cache_key] = record.id
            pending.append(PendingItem(
                id=record.id,
                raw_name=record.raw_name,
                source=record.source,
                cache_key=cache_key,
            ))
            lookup_by_id[record.id] = record
            cache_key_by_id[record.id] = cache_key

        # Track payloads by cache_key for deferred duplicate resolution
        payload_by_cache_key: Dict[str, Dict] = {}

        if pending:
            metrics["llm_calls"] += len(pending)
            successes, failed = self._call_llm_with_retry(pending)
            for item_id, payload in successes.items():
                record = lookup_by_id[item_id]
                guard = apply_guardrails(record.raw_name, payload)
                cache_set(cache_key_by_id[item_id], payload)
                payload_by_cache_key[cache_key_by_id[item_id]] = payload
                # upsert_alias_result(record.raw_name, guard, record.source, getattr(job, "id", None))  # Disabled for performance
                outputs.append(self._to_response(record, guard, cached=False))

            for item_id, error in failed.items():
                record = lookup_by_id[item_id]
                outputs.append(
                    NormalizeResponseItem(
                        id=record.id,
                        raw_name=record.raw_name,
                        cached=False,
                        error=error,
                    )
                )
                errors.append(f"{record.id}:{error}")

        # Process deferred duplicates - reuse results from first occurrence
        for record, cache_key in deferred_duplicates:
            # Try to get from LLM results first, then from cache
            payload = payload_by_cache_key.get(cache_key) or cache_get(cache_key)
            if payload:
                guard = apply_guardrails(record.raw_name, payload)
                outputs.append(self._to_response(record, guard, cached=True))
            else:
                # First occurrence failed, propagate error
                outputs.append(
                    NormalizeResponseItem(
                        id=record.id,
                        raw_name=record.raw_name,
                        cached=False,
                        error="duplicate_of_failed_item",
                    )
                )
                errors.append(f"{record.id}:duplicate_of_failed_item")

        # Log cache metrics
        logger.info(
            "Batch complete: %d cache hits, %d near-dupe hits, %d LLM calls, %d batch dedup",
            metrics["cache_hits"], metrics["near_dupe_hits"],
            metrics["llm_calls"], metrics["batch_dedup"]
        )

        return outputs, errors

    def _call_llm_with_retry(self, pending: List[PendingItem]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        successes: Dict[str, Dict] = {}
        failures: Dict[str, str] = {}
        retry_queue: List[PendingItem] = pending

        for attempt in range(2):
            if not retry_queue:
                break
            try:
                llm_rows = normalize_batch_gpt4o_mini([
                    {
                        "id": item.id,
                        "raw_name": item.raw_name,
                        "retry_suffix": item.retry_suffix,
                    }
                    for item in retry_queue
                ])
            except Exception as exc:  # pragma: no cover - network failure path
                logger.exception("LLM call failed for %d items", len(retry_queue))
                for item in retry_queue:
                    failures[item.id] = "llm_call_failed"
                break

            response_by_id = {row.get("id"): row for row in llm_rows}
            next_retry: List[PendingItem] = []
            for pending_item in retry_queue:
                row = response_by_id.get(pending_item.id)
                if not row:
                    failures[pending_item.id] = "missing_response"
                    continue
                payload = row.get("payload")
                if not payload:
                    failures[pending_item.id] = "llm_call_failed"
                    continue
                try:
                    self.validator.validate(payload)
                except ValidationError as err:
                    logger.warning("Invalid JSON schema for id %s (attempt %s): %s. Raw payload: %s", 
                                 pending_item.id, attempt + 1, err, payload)
                    if attempt == 0:
                        next_retry.append(PendingItem(
                            id=pending_item.id,
                            raw_name=pending_item.raw_name,
                            source=pending_item.source,
                            cache_key=pending_item.cache_key,
                            retry_suffix=INVALID_SUFFIX,
                        ))
                    else:
                        failures[pending_item.id] = "invalid_output"
                    continue
                successes[pending_item.id] = payload

            retry_queue = next_retry

        # Any items still outstanding after retries are failures.
        for item in retry_queue:
            failures[item.id] = "invalid_output"
        return successes, failures

    def _to_response(self, record: NormalizeRecord, guard: GuardrailResult, cached: bool) -> NormalizeResponseItem:
        result = CanonicalResult(
            canonical=guard.canonical,
            canonical_with_article=guard.canonical_with_article,
            article_policy=guard.article_policy,
            is_new=guard.is_new,
            confidence=guard.confidence,
            reason=guard.reason,
            key_form=guard.key_form,
            display_form=guard.display_form,
            flags=list(guard.flags),
        )
        return NormalizeResponseItem(
            id=record.id,
            raw_name=record.raw_name,
            cached=cached,
            result=result,
        )

    def process_job(self, job: JobRun, records: Sequence[NormalizeRecord]) -> Tuple[List[NormalizeResponseItem], List[str]]:
        logger.info("Processing job %s with %d records", job.id, len(records))
        set_job_status(job.id, JobStatus.running)
        try:
            results, errors = self._process_records(records, job)
            status = JobStatus.done if not errors else JobStatus.partial
            set_job_status(job.id, status)
            return results, errors
        except Exception as exc:  # pragma: no cover - catastrophic path
            logger.exception("Job %s failed: %s", job.id, exc)
            set_job_status(job.id, JobStatus.failed)
            raise


    def _load_records_from_csv(self, path: str, source: str) -> List[NormalizeRecord]:
        records: List[NormalizeRecord] = []
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = NormalizeRecord(
                    id=row.get("id") or str(uuid.uuid4()),
                    raw_name=row.get("raw_name") or row.get("company") or "",
                    source=source,
                    country_hint=row.get("country_hint"),
                )
                records.append(record)
        return records


service = NormalizationService()

try:  # pragma: no cover - optional dependency
    from rq import Queue
except ImportError:  # pragma: no cover
    Queue = None  # type: ignore


def get_queue():  # pragma: no cover - simple accessor
    if Queue is None:
        raise RuntimeError("rq is not installed; cannot enqueue jobs")
    from app.stores.cache import get_client

    return Queue("normalize", connection=get_client())


def enqueue_job(upload_key: Optional[str], signed_url: Optional[str], source: str) -> JobRun:
    job = JobRun(status=JobStatus.queued, input_count=0)
    record_job(job)
    if Queue is None:
        logger.warning("rq not installed; job %s recorded but not enqueued", job.id)
        return job

    payload = {"upload_key": upload_key, "signed_url": signed_url, "source": source}
    get_queue().enqueue(run_ingest_job, str(job.id), payload, job_timeout=60 * 60)
    return job


def run_ingest_job(job_id: str, ingest: Dict[str, Optional[str]]) -> None:
    job = get_job(uuid.UUID(job_id))
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    set_job_status(job.id, JobStatus.running)

    upload_key = ingest.get("upload_key")
    signed_url = ingest.get("signed_url")
    source = ingest.get("source") or "csv"

    if upload_key and os.path.exists(upload_key):
        records = service._load_records_from_csv(upload_key, source)
    elif signed_url:
        raise NotImplementedError("Signed URL ingestion requires external downloader")
    else:
        raise RuntimeError("No ingest target supplied")

    service.process_job(job, records)


__all__ = [
    "NormalizationService",
    "service",
    "enqueue_job",
    "run_ingest_job",
]
