"""SSA listings, criteria, summaries, evidence linkages, reviewer feedback."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class SsaListing(Base):
    """The canonical SSA Blue Book listing.

    `rule_json` holds the structured criterion tree which is decomposed
    into atomic criteria when the listing is loaded for a case.
    """

    __tablename__ = "ssa_listings"
    __table_args__ = (UniqueConstraint("code", name="uq_ssa_listing_code"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    code: Mapped[str] = mapped_column(String(32), index=True)  # e.g. "12.04"
    title: Mapped[str] = mapped_column(String(512))
    body_system: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(Text)
    rule_json: Mapped[dict] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(32), default="2024.1")

    summary_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024), nullable=True
    )
    # Populated by the listings_embed worker after seed. Used by
    # identify_candidate_listings to triage which listings to analyze for
    # a case. NULL means "embedding stale, recompute on next worker pass."

    criteria: Mapped[list["SsaListingCriterion"]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )
    synonyms: Mapped[list["SsaListingSynonym"]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )


class SsaListingCriterion(Base):
    """Atomic, testable criterion derived from `SsaListing.rule_json`.

    Criteria form a tree. Internal nodes have `logic_operator` of AND/OR
    and no `criterion_text`. Leaves carry the testable text.
    """

    __tablename__ = "ssa_listing_criteria"
    __table_args__ = (
        Index("ix_criteria_listing", "listing_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    listing_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ssa_listings.id", ondelete="CASCADE"),
        index=True,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ssa_listing_criteria.id", ondelete="CASCADE"),
        nullable=True,
    )
    path: Mapped[str] = mapped_column(String(255))
    # e.g. "A", "A.1", "B.2.a"; useful for display + grouping

    logic_operator: Mapped[str | None] = mapped_column(String(8))  # "AND"/"OR"/None
    criterion_text: Mapped[str | None] = mapped_column(Text)
    is_leaf: Mapped[bool] = mapped_column(Boolean, default=True)
    duration_months_required: Mapped[int | None] = mapped_column(Integer)

    # Optional clinical hints to drive distilled-keyword query generation.
    keywords: Mapped[list | None] = mapped_column(JSON)

    listing: Mapped["SsaListing"] = relationship(back_populates="criteria")
    parent: Mapped["SsaListingCriterion | None"] = relationship(
        remote_side="SsaListingCriterion.id"
    )


class SsaListingSynonym(Base):
    """Per-listing synonym map for the synonym-variant retrieval query.

    Each row is one (canonical term, variant) pair scoped to a single listing.
    Per-listing scoping prevents cross-domain contamination — "compression" in
    1.15 means nerve-root compression, in 4.11 it might mean a compression
    stocking.

    The retrieval layer reads all rows for a listing once at the start of
    case analysis and builds an in-memory dict; we don't query this table
    per-criterion at runtime.
    """

    __tablename__ = "ssa_listing_synonyms"
    __table_args__ = (
        UniqueConstraint(
            "listing_id", "canonical", "variant",
            name="uq_listing_canonical_variant",
        ),
        Index("idx_synonyms_listing", "listing_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    listing_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ssa_listings.id", ondelete="CASCADE"),
        index=True,
    )
    canonical: Mapped[str] = mapped_column(String(128))
    variant: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    listing: Mapped["SsaListing"] = relationship(back_populates="synonyms")


class AiCaseSummary(Base):
    """One snapshot of the case-level analysis.

    Each run of the pipeline creates a new row so we keep history.
    Body holds the structured JSON returned by the consolidator.
    """

    __tablename__ = "ai_case_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        index=True,
    )
    model: Mapped[str] = mapped_column(String(128))
    body: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ListingEvidence(Base):
    """Per-case, per-criterion analysis result.

    This is the joined table that bundles a criterion's status, the
    chunks that support / contradict it, and the structured citations.
    `body` mirrors the LLM's structured output for replay/audit.
    """

    __tablename__ = "listing_evidence"
    __table_args__ = (
        UniqueConstraint(
            "case_id", "criterion_id", "summary_id", name="uq_listing_evidence_unit"
        ),
        Index("ix_listing_evidence_case", "case_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        index=True,
    )
    summary_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ai_case_summaries.id", ondelete="CASCADE"),
        index=True,
    )
    listing_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ssa_listings.id", ondelete="CASCADE"),
        index=True,
    )
    criterion_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("ssa_listing_criteria.id", ondelete="CASCADE"),
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32))
    # "MET" | "NOT_MET" | "PARTIALLY_MET" | "INSUFFICIENT_EVIDENCE"

    strength: Mapped[float | None] = mapped_column(Float)  # 0..1
    body: Mapped[dict] = mapped_column(JSON)
    # body schema -> see app/schemas/criterion.py CriterionAnalysis


class ReviewerFeedback(Base):
    """Reviewer notes / overrides on a criterion or listing."""

    __tablename__ = "reviewer_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        index=True,
    )
    target_type: Mapped[str] = mapped_column(String(32))  # "criterion" | "listing"
    target_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True))
    reviewer_id: Mapped[str] = mapped_column(String(128))
    rating: Mapped[str | None] = mapped_column(String(32))  # agree/disagree/unclear
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
