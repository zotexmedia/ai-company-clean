"""Redis helpers for exact- and near-duplicate caching."""

from __future__ import annotations

import json
import os
import hashlib
import time
from functools import lru_cache
from typing import Any, Dict, Optional

import redis

from app.llm.postprocess import min_clean
from app.stores import ann

TTL_SECONDS = 60 * 60 * 24 * 365
CACHE_VERSION = "v16"  # Ultra-restrictive article rule - only simple generic corporate names

# Fallback in-memory cache when Redis is unavailable
_memory_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
_redis_available = None


def clear_memory_cache() -> None:
    """Clear the in-memory cache fallback."""
    global _memory_cache
    _memory_cache.clear()


@lru_cache(maxsize=1)
def get_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def exact_cache_key(raw_name: str) -> str:
    cleaned = min_clean(raw_name)
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
    return f"norm:{CACHE_VERSION}:{digest}"


def _is_redis_available() -> bool:
    """Check if Redis is available, cache result for performance."""
    global _redis_available
    if _redis_available is not None:
        return _redis_available
    
    try:
        get_client().ping()
        _redis_available = True
        return True
    except Exception:
        _redis_available = False
        return False

def cache_get(key: str) -> Optional[Dict[str, Any]]:
    # Try Redis first
    if _is_redis_available():
        try:
            payload = get_client().get(key)
            if not payload:
                return None
            return json.loads(payload)
        except (redis.RedisError, json.JSONDecodeError, ConnectionError):
            # Redis failed, fall back to memory cache
            pass
    
    # Use memory cache as fallback
    if key in _memory_cache:
        value, expiry = _memory_cache[key]
        if time.time() < expiry:
            return value
        else:
            del _memory_cache[key]
    
    return None


def cache_set(key: str, value: Dict[str, Any], ttl: int = TTL_SECONDS) -> None:
    # Try Redis first
    if _is_redis_available():
        try:
            get_client().setex(key, ttl, json.dumps(value))
            return
        except (redis.RedisError, ConnectionError):
            # Redis failed, fall back to memory cache
            pass
    
    # Use memory cache as fallback
    expiry = time.time() + ttl
    _memory_cache[key] = (value, expiry)
    
    # Clean up expired entries occasionally (keep memory usage reasonable)
    if len(_memory_cache) > 1000:
        current_time = time.time()
        expired_keys = [k for k, (_, exp) in _memory_cache.items() if current_time >= exp]
        for k in expired_keys:
            del _memory_cache[k]


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
    "clear_memory_cache",
    "exact_cache_key",
    "get_client",
    "near_dupe_lookup",
    "warm_cache",
]
