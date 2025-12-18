"""Optional approximate-nearest-neighbor helpers for near-duplicate reuse."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

from app.llm.postprocess import min_clean

logger = logging.getLogger(__name__)


@dataclass
class NearDupeMatch:
    raw_name: str
    score: float
    cache_key: str
    payload: Optional[Dict]


class BaseANNIndex:
    def query(self, raw_name: str, threshold: float) -> Optional[NearDupeMatch]:  # pragma: no cover - interface
        raise NotImplementedError


class NullIndex(BaseANNIndex):
    def query(self, raw_name: str, threshold: float) -> Optional[NearDupeMatch]:
        return None


class PgTrigramIndex(BaseANNIndex):
    """PostgreSQL trigram similarity-based near-duplicate lookup."""

    def __init__(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg is required for PostgreSQL index") from exc
        self._pg = psycopg
        self._dsn = os.getenv("POSTGRES_DSN")
        if not self._dsn:
            raise RuntimeError("POSTGRES_DSN not configured")

    def query(self, raw_name: str, threshold: float) -> Optional[NearDupeMatch]:
        """Find similar company name using trigram similarity."""
        # Convert threshold from 0.92 scale to trigram scale (0.4-0.6 is good for company names)
        trgm_threshold = max(0.3, threshold - 0.5)  # 0.92 -> 0.42

        lookup_sql = """
            SELECT raw_name, score, payload
            FROM company_normalizer.find_similar($1, $2)
        """
        try:
            with self._pg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(lookup_sql, (raw_name, trgm_threshold))
                    row = cur.fetchone()
                    if not row:
                        return None
                    raw, score, payload = row
                    # Lazy import to avoid circular dependency with cache.py
                    from app.stores.cache import exact_cache_key
                    cache_key = exact_cache_key(raw)
                    return NearDupeMatch(raw_name=raw, score=score, cache_key=cache_key, payload=payload)
        except Exception as exc:
            logger.warning("Trigram lookup failed: %s", exc)
            return None

    def add_alias(self, raw_name: str, canonical: str, payload: Dict) -> None:
        """Add a company name alias to the database for future lookups."""
        cleaned = min_clean(raw_name)
        add_sql = "SELECT company_normalizer.add_alias($1, $2, $3, $4)"
        try:
            with self._pg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    import json
                    cur.execute(add_sql, (raw_name, cleaned, canonical, json.dumps(payload)))
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to add alias: %s", exc)


# Keep PgVectorIndex as alias for backwards compatibility
PgVectorIndex = PgTrigramIndex


class FaissIndex(BaseANNIndex):
    def __init__(self):  # pragma: no cover - heavy dependency
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss is required for FAISS index") from exc
        self.faiss = faiss
        self.index = None  # Load from disk on demand.

    def query(self, raw_name: str, threshold: float) -> Optional[NearDupeMatch]:
        if not self.index:
            logger.warning("FAISS index not loaded; returning None")
            return None
        # Placeholder: call your embedding encoder + FAISS search.
        return None


@lru_cache(maxsize=1)
def get_index() -> BaseANNIndex:
    backend = os.getenv("ANN_BACKEND", "none").lower()
    try:
        if backend == "pgvector":
            return PgVectorIndex()
        if backend == "faiss":
            return FaissIndex()
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Failed to initialize ANN backend %s: %s", backend, exc)
    return NullIndex()


__all__ = ["get_index", "NearDupeMatch"]
