"""Heuristic post-processing and guardrails for normalization outputs."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

LEGAL_SUFFIX_PATTERNS = [
    r'incorporated', r'inc\.?',
    r'llc', r'l\.l\.c\.?',
    r'company', r'co\.?',
    r'corp\.?', r'corporation',
    r'limited', r'ltd\.?', r'plc',
    r'gmbh', r's\.a\.?', r'bv', r'b\.v\.?', r'ag',
    r'dds', r'dmd',
    r'p\.c\.?', r'p\s*c', r'pc',
    r'p\.a\.?', r'p\s*a', r'pa',
    r'llp', r'l\.l\.p\.?',
    r'm\.d\.?', r'm\s*d', r'md'
]

SUFFIX_RE = re.compile(r"\s+(?:" + "|".join(LEGAL_SUFFIX_PATTERNS) + r")\s*$", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^\w\s&-]")
MULTISPACE_RE = re.compile(r"\s{2,}")
APOSTROPHE_RE = re.compile(r"[’']")
STOPWORDS = {'of', 'and', 'the', 'for', 'in', 'on', 'at', 'with', 'to', 'from', 'by'}
STATE_CODES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky',
    'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd',
    'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
}
ACRONYM_WHITELIST = {'usa', 'bsn', 'ibm', 'uams', 'uapb', 'mri', 'ct', 'ltb'}


def min_clean(value: str) -> str:
    """Deterministic normalization for cache keys (NFKC, trim, dedupe spaces)."""
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip('"\'()[]{}.,;:')
    return normalized.lower()


def _smart_case(word: str, first: bool) -> str:
    low = word.lower()
    if low in STOPWORDS:
        return word.capitalize() if first else low
    if len(word) == 2 and low in STATE_CODES:
        return word.upper()
    if word.isupper() and low in ACRONYM_WHITELIST:
        return word
    return "-".join(part.capitalize() for part in word.split('-'))


def clean_company_name(name: str) -> str:
    if not name:
        return ""
    name = APOSTROPHE_RE.sub("", name)
    name = NON_ALNUM_RE.sub(" ", name)
    name = MULTISPACE_RE.sub(" ", name).strip()
    name = SUFFIX_RE.sub("", name).strip()
    if not re.search(r"&\s+co\.?$", name, re.IGNORECASE):
        name = re.sub(r"\s+co\.?$", "", name, flags=re.IGNORECASE)
    tokens = name.split()
    styled = [_smart_case(tok, idx == 0) for idx, tok in enumerate(tokens)]
    if styled and styled[-1] == '&':
        styled.append('Co')
    return " ".join(styled)


def tokenize(value: str) -> List[str]:
    return [tok for tok in re.split(r"[^\w&]+", value.lower()) if tok]


def token_overlap(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@dataclass
class GuardrailResult:
    canonical: str
    is_new: bool
    confidence: float
    reason: Optional[str]
    key_form: str
    display_form: str
    raw_reason: Optional[str]
    flags: Tuple[str, ...]


def apply_guardrails(raw_name: str, payload: Dict[str, object]) -> GuardrailResult:
    """Enforce post-processing rules and generate join/display helpers."""
    canonical = str(payload.get("canonical", "")).strip()
    is_new = bool(payload.get("is_new", False))
    confidence = float(payload.get("confidence", 0.0))
    reason = payload.get("reason")
    flags: List[str] = []

    if not canonical:
        canonical = clean_company_name(raw_name)
        confidence = 0.0
        flags.append("empty_canonical")

    overlap = token_overlap(raw_name, canonical)
    if overlap < 0.3:
        confidence = min(confidence, 0.5)
        flags.append("low_token_overlap")

    cleaned_display = clean_company_name(canonical)
    key_form = min_clean(canonical)

    if not cleaned_display:
        cleaned_display = clean_company_name(raw_name)
        flags.append("fallback_display")

    return GuardrailResult(
        canonical=canonical,
        is_new=is_new,
        confidence=confidence,
        reason=reason if isinstance(reason, str) and reason.strip() else None,
        key_form=key_form,
        display_form=cleaned_display,
        raw_reason=reason if isinstance(reason, str) else None,
        flags=tuple(flags),
    )


def redact_for_logging(raw_name: str) -> str:
    """Return a lightly redacted form for observability."""
    tokens = raw_name.split()
    if len(tokens) <= 1:
        return tokens[0][:3] + "***" if tokens else ""
    return f"{tokens[0]} ... {tokens[-1]}"


__all__ = [
    "apply_guardrails",
    "clean_company_name",
    "GuardrailResult",
    "min_clean",
    "redact_for_logging",
]
