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
CACHE_VERSION = "v21"  # Added smart capitalization for ALL CAPS company names

# Fallback in-memory cache when Redis is unavailable
_memory_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
_redis_available = None
_redis_checked_at = 0.0
# Re-probe Redis on this cadence so a transient failure at boot (cold start, brief
# blip) does NOT disable the shared cache for the whole process lifetime.
REDIS_RECHECK_SECONDS = float(os.getenv("REDIS_RECHECK_SECONDS", "30"))


def clear_memory_cache() -> None:
    """Clear the in-memory cache fallback."""
    global _memory_cache
    _memory_cache.clear()


@lru_cache(maxsize=1)
def get_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Bounded timeouts + pooling so a hung/slow Redis fails fast to the memory
    # fallback instead of blocking a threadpool worker up to the OS TCP timeout.
    return redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "2")),
        socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT", "2")),
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=True,
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "24")),
    )


def exact_cache_key(raw_name: str) -> str:
    cleaned = min_clean(raw_name)
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
    return f"norm:{CACHE_VERSION}:{digest}"


def _is_redis_available() -> bool:
    """Check if Redis is available. Result is cached for REDIS_RECHECK_SECONDS and
    then re-probed — so a transient failure never latches Redis off permanently
    (the previous bug: one failed ping disabled Redis for the whole process)."""
    global _redis_available, _redis_checked_at
    now = time.time()
    if _redis_available is not None and (now - _redis_checked_at) < REDIS_RECHECK_SECONDS:
        return _redis_available

    try:
        get_client().ping()
        _redis_available = True
    except Exception:
        _redis_available = False
    _redis_checked_at = now
    return _redis_available

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


def cache_get_many(keys: list[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Batch fetch: one Redis MGET round-trip for the whole set instead of N
    sequential GETs. Returns {key: payload_or_None}. Falls back to the in-memory
    cache (per key) if Redis is unavailable, matching cache_get() semantics."""
    result: Dict[str, Optional[Dict[str, Any]]] = {k: None for k in keys}
    if not keys:
        return result

    unresolved = list(dict.fromkeys(keys))  # de-dup, preserve order

    if _is_redis_available():
        try:
            values = get_client().mget(unresolved)
            for k, payload in zip(unresolved, values):
                if not payload:
                    continue
                try:
                    result[k] = json.loads(payload)
                except json.JSONDecodeError:
                    result[k] = None
            return result
        except (redis.RedisError, ConnectionError):
            # Redis failed mid-call → fall through to memory cache
            pass

    now = time.time()
    for k in unresolved:
        entry = _memory_cache.get(k)
        if not entry:
            continue
        value, expiry = entry
        if now < expiry:
            result[k] = value
        else:
            del _memory_cache[k]
    return result


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
    "cache_get_many",
    "cache_set",
    "clear_memory_cache",
    "exact_cache_key",
    "get_client",
    "near_dupe_lookup",
    "warm_cache",
]
