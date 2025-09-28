"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    You normalize English company names into a single canonical brand name for email marketing and professional communication.

    Output strict JSON only, matching the provided schema. Do not include prose, markdown, or explanations outside the JSON.

    Goals
    1. Preserve the true brand for professional email communication.
    2. Ignore legal suffixes and administrative location tails when they are not part of the brand.
    3. Apply the definite article "the" correctly so the canonical reads naturally in email greetings and business correspondence.
    4. Use proper business capitalization that looks professional in emails.

    General rules
    - Language is always English.
    - **CRITICAL CAPITALIZATION RULE**: ALWAYS convert ALL CAPS company names to proper Title Case for professional appearance. Only exceptions are established acronyms like "IBM", "AT&T", "NASA".
    - Examples of required conversions:
      • "ROCKERT DENTAL STUDIO" → "Rockert Dental Studio"  
      • "GENESIS INTEGRATIVE MEDICINE" → "Genesis Integrative Medicine"
      • "BRIX GROUP" → "Brix Group"
    - Remove surrounding quotes, emojis, and trailing punctuation.
    - **Industry word rules**: Keep the core industry descriptor, remove generic qualifiers:
      • "Law Firm" → "Law" (keep industry, remove generic "Firm")
      • "Bookkeeping Services" → "Bookkeeping" (keep industry, remove generic "Services")  
      • "Dental Studio" → "Dental" (keep industry, remove generic "Studio")
      • "Pet Boutique" → keep both (when both words are specific)
      • "Accounting Services" → "Accounting" (keep industry, remove "Services")
    - **Article-based exception**: When "The" precedes the name, keep the full firm name:
      • "Hess Law Firm" → "Hess Law" (remove generic "Firm")
      • "The Hess Law Firm" → "The Hess Law Firm" (official brand name with article)
    - Remove generic business descriptors: "and Associates", "Firm", "Services", "Studio", "Company", "Group" (unless part of official brand with "The")
    - Keep industry-specific words: "Law", "Dental", "Medical", "Bookkeeping", "Accounting", "Pet", etc.
    - Keep numbers and alphanumerics that are part of the brand ("Studio 54", "3M").
    - Replace "&" with "and" unless the ampersand is integral to the brand (e.g., "AT&T" stays "AT&T").
    - Remove store/unit markers: "#12", "Suite 300", "Unit B", "Store 145".
    - Drop administrative markers: "HQ", "Headquarters", "Main Office", "Branch", "Warehouse", "Plant 2".
    - Prefer the d/b/a brand when present (use the portion after "dba" / "DBA").

    Legal suffixes to ignore when not part of the brand:
    inc, incorporated, co, company, corp, corporation, ltd, limited, llc, l.l.c., llp, l.l.p., plc, gmbh, srl, s.a., bv, nv, ab, oy, as, kk, pty ltd, sdn bhd, pbc, pc, lp, llc-pc

    Common non-brand tails to remove:
    accounts payable/receivable, billing, collections, fulfillment, corporate office, division, department, warehouse, plant, location, site, campus

    Location handling
    - Keep location tokens when integral to the brand or conventional usage:
      - Location token at the start followed by a generic brand noun: "The Dallas Group", "Dallas Dental Clinic", "Boston Market".
      - Multi-word brands where location is fused into the name: "New York Life", "Arizona Beverages".
      - No delimiter suggests an address tail (no comma, dash, or parentheses after the brand).
    - Remove location tails that are administrative or appended: "Acme Cleaning, Dallas", "Acme Cleaning - Dallas", "Acme Cleaning (Dallas)".
    - Franchise/store descriptors: keep only the national brand if followed by store numbers or cities, e.g., "Walmart Store 145 - Plano" -> "Walmart".
    - If the only distinctive tokens are location plus an industry noun, keep both (e.g., "Dallas Dental Clinic").

    Definite article policy
    Return two strings:
    - canonical: the authoritative display name. Include "the" only when it is required for natural brand reading or is an official part of the brand.
    - canonical_with_article: a grammatically natural form with "the" prefixed when appropriate (lowercase "the" unless the official brand capitalizes it). If "the" is not natural or not used for that brand, canonical_with_article must equal canonical.

    Article categories:
    - required – Professional service patterns that sound natural in email context: "The Law Office of [Name]", "The Office of [Title]", "The [Place] Group", "The University of [Place]", "The City of [Name]". Use when it sounds natural in "Dear [Company Name]" emails.
    - official – the brand itself includes "The" as part of the official name ("The Home Depot", "The North Face", "The Ohio State University"). Keep the capitalized "The".
    - optional – add lowercase "the" only in canonical_with_article to improve readability (e.g., "Dallas Group" -> canonical_with_article "the Dallas Group"), but do not include it in canonical.
    - none – most commercial brands and standalone universities where "the" is neither official nor helpful for email communication.

    Ambiguity & safety checks
    - Do not collapse distinct brands; keep disambiguating tokens when needed.
    - Do not output empty or single-letter canonicals.
    - If the result would be only a location with no brand word, keep the best brand reading and lower confidence accordingly.

    Confidence guidelines
    - 0.95-1.00: exact brand form or minor legal/location cleanup.
    - 0.85-0.94: clear brand with small uncertainty.
    - 0.70-0.84: ambiguous; reasonable choice made.
    - <0.70: uncertain; consider human review.

    Return strict JSON only.
    """
).strip()

USER_TEMPLATE = dedent(
    """
    RAW: "{raw_name}"

    Guidelines to apply:
    • Follow the rules and the definite article policy exactly.
    • **MANDATORY**: Convert ALL CAPS words to Title Case (except established acronyms like IBM, AT&T)
    • Return only valid JSON that matches the schema.
    • Always include the "reason" field with a brief explanation of what was kept or removed.
    • Your "canonical" field must NEVER contain ALL CAPS words unless they are established acronyms.
    """
).strip()

FEW_SHOTS: List[Tuple[str, dict]] = [
    (
        "University of Texas at Dallas",
        {
            "canonical": "University of Texas at Dallas",
            "canonical_with_article": "the University of Texas at Dallas",
            "article_policy": "required",
            "is_new": False,
            "confidence": 0.98,
            "reason": "University of [Place] pattern requires article only in grammatical contexts"
        },
    ),
    (
        "Bank of America, Dallas Branch",
        {
            "canonical": "Bank of America",
            "canonical_with_article": "Bank of America",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Removed branch descriptor; brand does not use 'the'"
        },
    ),
    (
        "The Home Depot Store 145 - Plano",
        {
            "canonical": "The Home Depot",
            "canonical_with_article": "The Home Depot",
            "article_policy": "official",
            "is_new": False,
            "confidence": 0.97,
            "reason": "Official brand includes capitalized 'The'"
        },
    ),
    (
        "The Dallas Group",
        {
            "canonical": "Dallas Group",
            "canonical_with_article": "the Dallas Group",
            "article_policy": "optional",
            "is_new": False,
            "confidence": 0.93,
            "reason": "Article optional for readability"
        },
    ),
    (
        "Dallas Dental Clinic",
        {
            "canonical": "Dallas Dental Clinic",
            "canonical_with_article": "the Dallas Dental Clinic",
            "article_policy": "optional",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Location + industry noun reads naturally with article"
        },
    ),
    (
        "Boston Scientific Inc.",
        {
            "canonical": "Boston Scientific",
            "canonical_with_article": "Boston Scientific",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.98,
            "reason": "Known brand; no article"
        },
    ),
    (
        "New York Life Insurance Company",
        {
            "canonical": "New York Life",
            "canonical_with_article": "New York Life",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.98,
            "reason": "Normalized to common brand form"
        },
    ),
    (
        "Subway #123 Dallas",
        {
            "canonical": "Subway",
            "canonical_with_article": "Subway",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Removed store/city tail"
        },
    ),
    (
        "Acme LLC dba Metro Facility Care",
        {
            "canonical": "Metro Facility Care",
            "canonical_with_article": "Metro Facility Care",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Preferred d/b/a brand"
        },
    ),
    (
        "AT&T Corp.",
        {
            "canonical": "AT&T",
            "canonical_with_article": "AT&T",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.98,
            "reason": "Brand keeps ampersand; legal suffix removed"
        },
    ),
    (
        "Acme Cleaning (Dallas, TX 75201)",
        {
            "canonical": "Acme Cleaning",
            "canonical_with_article": "Acme Cleaning",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.92,
            "reason": "Removed address tail"
        },
    ),
    (
        "Augustana University",
        {
            "canonical": "Augustana University",
            "canonical_with_article": "Augustana University",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Most universities do not use 'the' in normal usage"
        },
    ),
    (
        "GENESIS INTEGRATIVE MEDICINE LLC",
        {
            "canonical": "Genesis Integrative Medicine",
            "canonical_with_article": "Genesis Integrative Medicine",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Converted ALL CAPS to Title Case; removed LLC suffix"
        },
    ),
    (
        "The Law Office of Damon M. Fisch, PC",
        {
            "canonical": "The Law Office of Damon M. Fisch",
            "canonical_with_article": "The Law Office of Damon M. Fisch",
            "article_policy": "required",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Professional service pattern requires 'The' for natural email communication"
        },
    ),
    (
        "ROCKERT DENTAL STUDIO, INC.",
        {
            "canonical": "Rockert Dental",
            "canonical_with_article": "Rockert Dental",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Converted ALL CAPS to Title Case; kept 'Dental', removed 'Studio' and Inc suffix"
        },
    ),
    (
        "Ferrari Bookkeeping Services LLC",
        {
            "canonical": "Ferrari Bookkeeping",
            "canonical_with_article": "Ferrari Bookkeeping",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Kept industry word 'Bookkeeping'; removed generic 'Services' and LLC suffix"
        },
    ),
    (
        "Legacy and Life Law Firm, PC",
        {
            "canonical": "Legacy and Life Law",
            "canonical_with_article": "Legacy and Life Law",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Kept industry word 'Law'; removed generic 'Firm' and PC suffix"
        },
    ),
    (
        "Two Bostons Pet Boutique",
        {
            "canonical": "Two Bostons",
            "canonical_with_article": "Two Bostons",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.94,
            "reason": "Core brand name; removed generic business descriptor 'Pet Boutique'"
        },
    ),
    (
        "Robert Slayton and Associates",
        {
            "canonical": "Robert Slayton",
            "canonical_with_article": "Robert Slayton",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Personal brand name; removed generic 'and Associates'"
        },
    ),
    (
        "DECO Accounting Services Inc",
        {
            "canonical": "DECO Accounting",
            "canonical_with_article": "DECO Accounting",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.95,
            "reason": "Kept industry word 'Accounting'; removed generic 'Services' and Inc"
        },
    ),
    (
        "BRIX GROUP LLC",
        {
            "canonical": "Brix Group",
            "canonical_with_article": "Brix Group",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Converted ALL CAPS to Title Case; kept 'Group' as part of brand name; removed LLC"
        },
    ),
    (
        "Hess Law Firm",
        {
            "canonical": "Hess Law",
            "canonical_with_article": "Hess Law",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Kept industry word 'Law'; removed generic 'Firm' (no article prefix)"
        },
    ),
    (
        "The Hess Law Firm",
        {
            "canonical": "The Hess Law Firm",
            "canonical_with_article": "The Hess Law Firm",
            "article_policy": "official",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Official brand name with 'The' prefix; kept full 'Law Firm' as part of brand identity"
        },
    ),
]


# Convert Python booleans in FEW_SHOTS to JSON-friendly strings when serializing
for idx, (raw, payload) in enumerate(FEW_SHOTS):
    FEW_SHOTS[idx] = (raw, json.loads(json.dumps(payload)))


def build_user_message(raw_name: str, retry_suffix: str | None = None) -> str:
    message = USER_TEMPLATE.format(raw_name=raw_name)
    if retry_suffix:
        message = f"{message}\n\n{retry_suffix.strip()}"
    return message


def few_shot_messages() -> Iterable[dict]:
    for raw, payload in FEW_SHOTS:
        yield {
            "role": "user",
            "content": build_user_message(raw),
        }
        yield {
            "role": "assistant",
            "content": json.dumps(payload, separators=(",", ":")),
        }


def build_conversation(raw_name: str, retry_suffix: str | None = None) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *few_shot_messages(),
        {
            "role": "user",
            "content": build_user_message(raw_name, retry_suffix=retry_suffix),
        },
    ]
