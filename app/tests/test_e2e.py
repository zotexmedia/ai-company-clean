from __future__ import annotations

import itertools
from typing import Dict, List

import pytest

from app.api.schemas import NormalizeRecord
from app.tests.fixtures import FIXTURE_ROWS
from app.workers import normalize_worker as nw


@pytest.fixture
def memory_cache(monkeypatch):
    store: Dict[str, Dict] = {}

    monkeypatch.setattr(nw, "cache_get", store.get)
    monkeypatch.setattr(nw, "cache_set", lambda key, value: store.__setitem__(key, value))
    monkeypatch.setattr(nw, "near_dupe_lookup", lambda raw: None)
    return store


@pytest.fixture
def noop_db(monkeypatch):
    monkeypatch.setattr(nw, "upsert_alias_result", lambda raw, guard, source, job_id=None: None)


def build_records(count: int) -> List[NormalizeRecord]:
    rows = itertools.islice(FIXTURE_ROWS, count)
    return [
        NormalizeRecord(id=row["id"], raw_name=row["raw_name"], source="csv", country_hint="US")
        for row in rows
    ]


def test_retry_invalid_output(memory_cache, noop_db, monkeypatch):
    service = nw.NormalizationService(batch_size=2)
    records = build_records(1)

    call_counter = {"count": 0}

    def fake_llm(items):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return [
                {
                    "id": items[0]["id"],
                    "raw_name": items[0]["raw_name"],
                    "payload": {
                        "canonical": "",
                        "canonical_with_article": "",
                        "article_policy": "none",
                        "is_new": False,
                        "confidence": 0.9,
                        "reason": None,
                    },
                }
            ]
        return [
            {
                "id": items[0]["id"],
                "raw_name": items[0]["raw_name"],
                "payload": {
                    "canonical": "Acme Facility Services",
                    "canonical_with_article": "Acme Facility Services",
                    "article_policy": "none",
                    "is_new": False,
                    "confidence": 0.92,
                    "reason": "retry success",
                },
            }
        ]

    monkeypatch.setattr(nw, "normalize_batch_gpt4o_mini", fake_llm)

    results, errors = service._process_records(records, None)
    assert not errors
    assert call_counter["count"] == 2
    assert results[0].result.canonical == "Acme Facility Services"
    assert (
        results[0].result.canonical_with_article
        == "Acme Facility Services"
    )
    assert results[0].result.article_policy == "none"
    assert results[0].result.confidence == pytest.approx(0.92)
    assert results[0].result.flags == []


def test_guardrail_low_overlap(memory_cache, noop_db, monkeypatch):
    service = nw.NormalizationService(batch_size=2)
    records = build_records(1)

    def fake_llm(items):
        return [
            {
                "id": items[0]["id"],
                "raw_name": items[0]["raw_name"],
                "payload": {
                    "canonical": "Completely Different Brand",
                    "canonical_with_article": "Completely Different Brand",
                    "article_policy": "none",
                    "is_new": True,
                    "confidence": 0.95,
                    "reason": "",
                },
            }
        ]

    monkeypatch.setattr(nw, "normalize_batch_gpt4o_mini", fake_llm)

    results, errors = service._process_records(records, None)
    assert not errors
    assert results[0].result.confidence == pytest.approx(0.5)
    assert "low_token_overlap" in results[0].result.flags


def test_cache_hit_bypasses_llm(memory_cache, noop_db, monkeypatch):
    service = nw.NormalizationService(batch_size=2)
    records = build_records(1)

    cached_payload = {
        "canonical": "Cached Company",
        "canonical_with_article": "Cached Company",
        "article_policy": "none",
        "is_new": False,
        "confidence": 0.88,
        "reason": "memoized",
    }
    key = nw.exact_cache_key(records[0].raw_name)
    memory_cache[key] = cached_payload

    monkeypatch.setattr(nw, "normalize_batch_gpt4o_mini", lambda items: pytest.fail("LLM should not be called"))

    results, errors = service._process_records(records, None)
    assert not errors
    assert results[0].cached is True
    assert results[0].result.canonical == "Cached Company"
    assert results[0].result.canonical_with_article == "Cached Company"
    assert results[0].result.article_policy == "none"
