"""Pydantic schemas for API input/output."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field, conlist, model_validator


class NormalizeRecord(BaseModel):
    id: str = Field(..., description="Client-supplied primary key")
    raw_name: str = Field(..., min_length=1)
    source: Literal["gmaps", "apollo", "csv"]
    country_hint: Optional[str] = Field(None, min_length=2, max_length=2)


class CanonicalResult(BaseModel):
    canonical: str
    is_new: bool
    confidence: float = Field(..., ge=0, le=1)
    reason: Optional[str] = None
    key_form: str
    display_form: str
    flags: List[str] = []


class NormalizeResponseItem(BaseModel):
    id: str
    raw_name: str
    cached: bool = False
    result: Optional[CanonicalResult] = None
    error: Optional[str] = None


class NormalizeRequest(BaseModel):
    records: conlist(NormalizeRecord, min_length=1, max_length=256)


class NormalizeResponse(BaseModel):
    results: List[NormalizeResponseItem]
    errors: List[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    partial = "partial"


class JobCreateRequest(BaseModel):
    upload_key: Optional[str] = None
    signed_url: Optional[AnyHttpUrl] = None
    source: Literal["csv", "apollo", "gmaps"] = "csv"

    @model_validator(mode="after")
    def ensure_ingest_target(cls, values: "JobCreateRequest") -> "JobCreateRequest":
        if not values.upload_key and not values.signed_url:
            raise ValueError("Either upload_key or signed_url is required")
        return values


class JobResource(BaseModel):
    id: UUID
    status: JobStatus
    input_count: int
    success_count: int
    error_count: int
    created_at: dt.datetime
    updated_at: dt.datetime
    result_path: Optional[str] = None


class JobCreateResponse(BaseModel):
    job: JobResource


__all__ = [
    "CanonicalResult",
    "JobCreateRequest",
    "JobCreateResponse",
    "JobResource",
    "JobStatus",
    "NormalizeRecord",
    "NormalizeRequest",
    "NormalizeResponse",
    "NormalizeResponseItem",
]
