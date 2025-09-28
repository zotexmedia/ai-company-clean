"""Redis helpers for exact- and near-duplicate caching."""

from __future__ import annotations

import json
import os
import hashlib
from functools import lru_cache
from typing import Any, Dict, Optional

import redis

from app.llm.postprocess import min_clean
from app.stores import ann

TTL_SECONDS = 60 * 60 * 24 * 365
CACHE_VERSION = "v1"


@lru_cache(maxsize=1)
def get_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def exact_cache_key(raw_name: str) -> str:
    cleaned = min_clean(raw_name)
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
    return f"norm:{CACHE_VERSION}:{digest}"


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    payload = get_client().get(key)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def cache_set(key: str, value: Dict[str, Any], ttl: int = TTL_SECONDS) -> None:
    get_client().setex(key, ttl, json.dumps(value))


def near_dupe_lookup(raw_name: str, threshold: float = 0.92) -> Optional[Dict[str, Any]]:
    """Optional near-duplicate lookup via ANN index."""
    index = ann.get_index()
    if not index:
        return None
    candidate = index.query(raw_name, threshold=threshold)
    if not candidate:
        return None
    cached = cache_get(candidate.cache_key)
    if cached:
        return cached
    if candidate.payload:
        cache_set(candidate.cache_key, candidate.payload)
        return candidate.payload
    return None


def warm_cache(entries: Dict[str, Dict[str, Any]]) -> None:
    """Warm Tier A cache from historic alias map on startup."""
    for raw_name, payload in entries.items():
        cache_set(exact_cache_key(raw_name), payload)


__all__ = [
    "cache_get",
    "cache_set",
    "exact_cache_key",
    "get_client",
    "near_dupe_lookup",
    "warm_cache",
]
