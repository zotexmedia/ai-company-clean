"""Postgres models and persistence helpers."""

from __future__ import annotations

import datetime as dt
import enum
import os
import uuid
from contextlib import contextmanager
from typing import Dict, Iterable, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from app.llm.postprocess import GuardrailResult

Base = declarative_base()


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    partial = "partial"


class CanonicalCompany(Base):
    __tablename__ = "canonical_company"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name = Column(Text, unique=True, nullable=False)
    key_form = Column(Text, unique=True, nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_seen = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    confidence_avg = Column(Float, nullable=False, default=0.0)
    aliases_count = Column(Integer, nullable=False, default=0)

    aliases = relationship("Alias", back_populates="canonical")


class Alias(Base):
    __tablename__ = "alias"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alias_name = Column(Text, nullable=False)
    canonical_id = Column(UUID(as_uuid=True), ForeignKey("canonical_company.id"), nullable=False)
    source = Column(String(32), nullable=True)
    first_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_seen = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    confidence_last = Column(Float, nullable=False, default=0.0)
    details = Column(JSON, nullable=True)

    canonical = relationship("CanonicalCompany", back_populates="aliases")


class JobRun(Base):
    __tablename__ = "job_run"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.queued)
    input_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    result_path = Column(Text, nullable=True)


DATABASE_URL = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/company_cleaner")

# Add logging to help debug connection issues
import logging
logger = logging.getLogger(__name__)
logger.info("Database URL configured: %s", DATABASE_URL.replace(DATABASE_URL.split("@")[0].split("//")[1], "***") if "@" in DATABASE_URL else "local")

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)


@contextmanager
def session_scope() -> Iterable[Session]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_models() -> None:
    Base.metadata.create_all(engine)


def upsert_alias_result(
    raw_name: str,
    result: GuardrailResult,
    source: Optional[str] = None,
    job_id: Optional[uuid.UUID] = None,
) -> None:
    """Persist canonical + alias rows, maintaining aggregates."""
    try:
        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        with session_scope() as session:
            canonical = session.execute(
                select(CanonicalCompany).where(CanonicalCompany.key_form == result.key_form)
            ).scalar_one_or_none()

            if not canonical:
                canonical = CanonicalCompany(
                    canonical_name=result.display_form,
                    key_form=result.key_form,
                    first_seen=now,
                    last_seen=now,
                    confidence_avg=result.confidence,
                    aliases_count=1,
                )
                session.add(canonical)
                session.flush()
            else:
                # Online EMA update for confidence average.
                new_avg = (
                    (canonical.confidence_avg * canonical.aliases_count + result.confidence)
                    / max(canonical.aliases_count + 1, 1)
                )
                canonical.confidence_avg = new_avg
                canonical.aliases_count += 1
                canonical.last_seen = now

            alias = session.execute(
                select(Alias).where(Alias.alias_name == raw_name, Alias.canonical_id == canonical.id)
            ).scalar_one_or_none()

            if alias:
                alias.last_seen = now
                alias.confidence_last = result.confidence
                alias.details = {
                    "flags": list(result.flags),
                    "reason": result.raw_reason,
                    "canonical_with_article": result.canonical_with_article,
                    "article_policy": result.article_policy,
                }
            else:
                alias = Alias(
                    alias_name=raw_name,
                    canonical_id=canonical.id,
                    source=source,
                    confidence_last=result.confidence,
                    details={
                        "flags": list(result.flags),
                        "reason": result.raw_reason,
                        "canonical_with_article": result.canonical_with_article,
                        "article_policy": result.article_policy,
                    },
                )
                session.add(alias)

            if job_id:
                increment_job_progress(session, job_id, success_delta=1)
    except Exception as e:
        # Database unavailable - log and continue without persistence
        logger.warning("Database unavailable, skipping persistence for '%s': %s", raw_name, str(e))
        logger.debug("Full error details: %s", e, exc_info=True)


def record_job(job: JobRun) -> uuid.UUID:
    with session_scope() as session:
        session.add(job)
        session.flush()
        return job.id


def increment_job_progress(session: Session, job_id: uuid.UUID, success_delta: int = 0, error_delta: int = 0) -> None:
    stmt = (
        update(JobRun)
        .where(JobRun.id == job_id)
        .values(
            success_count=JobRun.success_count + success_delta,
            error_count=JobRun.error_count + error_delta,
            updated_at=func.now(),
        )
    )
    session.execute(stmt)


def set_job_status(job_id: uuid.UUID, status: JobStatus, result_path: Optional[str] = None) -> None:
    with session_scope() as session:
        stmt = (
            update(JobRun)
            .where(JobRun.id == job_id)
            .values(status=status, updated_at=func.now(), result_path=result_path)
        )
        session.execute(stmt)


def get_job(job_id: uuid.UUID) -> Optional[JobRun]:
    with session_scope() as session:
        return session.get(JobRun, job_id)


__all__ = [
    "Alias",
    "CanonicalCompany",
    "JobRun",
    "JobStatus",
    "SessionLocal",
    "get_job",
    "increment_job_progress",
    "init_models",
    "record_job",
    "set_job_status",
    "upsert_alias_result",
]
