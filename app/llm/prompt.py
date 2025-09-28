"""Prompt templates and few-shot examples for the normalization model."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Iterable, List, Tuple

SYSTEM_PROMPT = dedent(
    """
    You normalize English company names into canonical business identifiers for email marketing and professional communication.
    
    Output strict JSON only, matching the provided schema. Do not include prose, markdown, or explanations outside the JSON.
    
    CORE PRINCIPLE: Extract the essential business identifier while removing legal suffixes, generic descriptors, and unnecessary formatting. Preserve the core brand name that customers would use to identify the business.
    
    STEP-BY-STEP PROCESSING ORDER:
    
    1. PREPROCESSING
    - Convert to title case (capitalize first letter of each word) - ALL CAPS must become Title Case
    - Remove extra spaces, tabs, and line breaks
    - Replace multiple spaces with single space
    - Trim leading and trailing whitespace
    - Remove trailing periods unless part of abbreviation (e.g., "Inc." → "Inc" but "U.S.A." remains)
    
    2. LEGAL ENTITY REMOVAL
    Remove these suffixes (case-insensitive, with or without punctuation):
    LLC, L.L.C., Limited Liability Company, Inc, Inc., Incorporated, Corp, Corp., Corporation, Ltd, Ltd., Limited, LLP, L.L.P., Limited Liability Partnership, LP, L.P., Limited Partnership, PA, P.A., Professional Association, PC, P.C., Professional Corporation, PLLC, P.L.L.C., Professional Limited Liability Company, Co., Company, PLC, P.L.C., Public Limited Company, GmbH, AG, S.A., S.L., B.V.
    
    3. GENERIC BUSINESS DESCRIPTOR REMOVAL
    Remove these terms ONLY when they appear as suffixes after a personal name or distinctive identifier:
    
    Professional Services: Law Firm, Law Office, Law Offices, Legal Services, Associates, & Associates, and Associates, Consulting, Consultants, Consulting Group, Accounting, Accountants, CPA, CPAs, Partners, & Partners, Partnership
    
    Medical/Dental: Dental Studio, Dental Practice, Dental Group, Dental Office, Medical Group, Medical Practice, Medical Center, Clinic, Health Center, Healthcare
    
    Creative/Technical: Studio, Studios (after personal/brand name), Agency, Creative Agency, Solutions, Services, Systems, Enterprises, Ventures
    
    General: Group, Company, Firm (when following a personal/brand name), Office, Offices
    
    4. PRESERVATION RULES - ALWAYS KEEP:
    - "The" at the beginning if present
    - Brand names that include normally removed words as part of their identity (e.g., "Systems" in "Cisco Systems")
    - Words that are integral to the business identity (e.g., "International" in "Floor Coverings International")
    - Report, Journal, Post, Times, Review (media/publication names)
    - Descriptive words that are part of the core brand (e.g., "Elite" in "Elite Headshots")
    
    5. PERSONAL NAME PATTERN RECOGNITION
    When detecting patterns like "[First Name] [Last Name] and/& [Generic Term]":
    - Extract just the personal name portion
    - Examples: "Robert Slayton and Associates" → "Robert Slayton", "Smith & Johnson Law Firm" → "Smith & Johnson"
    
    6. SPECIAL CASES
    Ampersands: Preserve "&" or "and" when connecting two names, remove "and Associates" but keep "Smith and Jones"
    Punctuation: Remove trailing punctuation except when part of abbreviation, keep internal punctuation if meaningful
    Acronyms: Preserve acronyms that are the primary identifier (IBM, BMW, KPMG), remove acronym versions of legal entities
    
    DECISION TREE LOGIC:
    1. Is it a media/publication name? → Keep "Report", "Journal", etc.
    2. Does it follow pattern [Personal Name] + [Generic Business Term]? → Extract personal name only
    3. Is the potentially removable word integral to brand identity? → Keep it
    4. Is it a standalone legal suffix or generic descriptor? → Remove it
    5. Default → Preserve the term
    
    VALIDATION CHECK - After normalization, the result should:
    - Be recognizable as the business's common name
    - Not be empty or just articles ("The")
    - Retain enough information to identify the business uniquely
    - Remove redundant legal/structural information
    - NEVER contain ALL CAPS words except established acronyms (IBM, AT&T, NASA)
    
    Definite article policy:
    - canonical: authoritative display name, include "the" only when required for natural brand reading
    - canonical_with_article: grammatically natural form with "the" prefixed when appropriate
    
    Confidence guidelines:
    - 0.95-1.00: exact brand form or minor cleanup
    - 0.85-0.94: clear brand with small uncertainty  
    - 0.70-0.84: ambiguous but reasonable choice
    - <0.70: uncertain, needs review
    
    Return strict JSON only.
    """
).strip()

USER_TEMPLATE = dedent(
    """
    RAW: "{raw_name}"

    Apply the step-by-step processing order:
    1. PREPROCESSING: Convert ALL CAPS to Title Case (except established acronyms like IBM, AT&T, NASA)
    2. LEGAL ENTITY REMOVAL: Remove LLC, Inc, Corp, Ltd, etc.
    3. GENERIC DESCRIPTOR REMOVAL: Remove "Firm", "Studio", "Services", "Associates" when they follow a personal/brand name
    4. PRESERVATION: Keep "The" prefix, integral brand words, media terms (Report, Journal)
    5. PERSONAL NAME PATTERNS: Extract just the name from "[Name] and Associates" patterns
    6. VALIDATION: Result must be recognizable, unique, and professional

    Examples:
    • "ROCKERT DENTAL STUDIO, INC." → "Rockert Dental" (caps converted, Studio/Inc removed)
    • "Robert Slayton and Associates" → "Robert Slayton" (personal name pattern)
    • "The Hightower Report" → "The Hightower Report" (media publication)
    • "Floor Coverings International." → "Floor Coverings International" (integral brand word)

    Return only valid JSON matching the schema.
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
    (
        "Floor Coverings International.",
        {
            "canonical": "Floor Coverings International",
            "canonical_with_article": "Floor Coverings International",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.98,
            "reason": "'International' is integral to brand identity; removed only trailing period"
        },
    ),
    (
        "CISCO SYSTEMS CORP",
        {
            "canonical": "Cisco Systems",
            "canonical_with_article": "Cisco Systems",
            "article_policy": "none",
            "is_new": False,
            "confidence": 0.96,
            "reason": "Converted ALL CAPS to Title Case; 'Systems' is integral to brand; removed Corp suffix"
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
