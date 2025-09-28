"""FastAPI entrypoint for the company name normalization service."""

from __future__ import annotations

import logging
from typing import Tuple
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.api.schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobResource,
    JobStatus as ApiJobStatus,
    NormalizeRequest,
    NormalizeResponse,
)
from app.stores.db import JobRun, JobStatus as DbJobStatus, get_job
from app.workers.normalize_worker import enqueue_job, service

logger = logging.getLogger(__name__)

app = FastAPI(title="Company Name Cleaner", version="1.0.0")


@app.get("/")
def root() -> dict:
    return {"message": "Company Name Cleaner API", "version": "1.0.0", "status": "running"}


def _job_to_resource(job: JobRun) -> JobResource:
    db_job = get_job(job.id)
    if db_job is None:
        raise RuntimeError(f"Job {job.id} vanished after creation")
    return JobResource(
        id=db_job.id,
        status=ApiJobStatus(db_job.status.value),
        input_count=db_job.input_count,
        success_count=db_job.success_count,
        error_count=db_job.error_count,
        created_at=db_job.created_at,
        updated_at=db_job.updated_at,
        result_path=db_job.result_path,
    )


@app.post("/normalize", response_model=NormalizeResponse)
async def normalize(req: NormalizeRequest) -> NormalizeResponse:
    results, errors = await service.process_online(req.records)
    return NormalizeResponse(results=results, errors=errors)


@app.post("/jobs", response_model=JobCreateResponse)
def create_job(payload: JobCreateRequest, background: BackgroundTasks) -> JobCreateResponse:
    job = enqueue_job(payload.upload_key, str(payload.signed_url) if payload.signed_url else None, payload.source)
    logger.info("Queued job %s from payload=%s", job.id, payload.dict())
    return JobCreateResponse(job=_job_to_resource(job))


@app.get("/jobs/{job_id}", response_model=JobResource)
def get_job_status(job_id: UUID) -> JobResource:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_resource(job)


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


__all__ = ["app"]
