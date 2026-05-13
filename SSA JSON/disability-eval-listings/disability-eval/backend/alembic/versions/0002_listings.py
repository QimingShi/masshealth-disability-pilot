"""listing synonyms + embedding column

Revision ID: 0002_listings
Revises: 0001_initial
Create Date: 2026-04-29

The 0001_initial migration already created ssa_listings and ssa_listing_criteria.
This migration adds:
  - ssa_listing_synonyms: one row per (listing, canonical, variant) for the
    synonym-variant retrieval query. Per-listing scoped intentionally so
    "compression" in 1.15 (nerve root) doesn't pollute matches in 4.11
    (compression stocking).
  - ssa_listings.summary_embedding: pgvector column populated by the
    listings_embed worker, used by identify_candidate_listings for triage.

Keywords on each leaf criterion are stored on ssa_listing_criteria.keywords
(JSON column, already exists from 0001). Storing keywords as a JSON array on
the criterion row is correct here — we only ever read them per-leaf at
retrieval time, never query "find all listings with keyword X" in production.

See data/listings/SCHEMA.md for the full design rationale.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID


# revision identifiers, used by Alembic.
revision = "0002_listings"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pg_trgm enables ILIKE/similarity indexes on synonym text
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ----------------------------------------------------------------
    #  Add summary_embedding column to ssa_listings
    # ----------------------------------------------------------------
    op.execute("ALTER TABLE ssa_listings ADD COLUMN summary_embedding vector(1024)")
    op.execute(
        """
        CREATE INDEX idx_ssa_listings_summary_embed
        ON ssa_listings USING ivfflat (summary_embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # ----------------------------------------------------------------
    #  ssa_listing_synonyms — per-listing scoped synonym map
    # ----------------------------------------------------------------
    op.create_table(
        "ssa_listing_synonyms",
        sa.Column(
            "id", PGUUID(as_uuid=True), primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "listing_id", PGUUID(as_uuid=True),
            sa.ForeignKey("ssa_listings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("canonical", sa.String(128), nullable=False),
        sa.Column("variant", sa.String(256), nullable=False),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "listing_id", "canonical", "variant", name="uq_listing_canonical_variant"
        ),
    )
    op.create_index("idx_synonyms_listing", "ssa_listing_synonyms", ["listing_id"])
    op.execute(
        """
        CREATE INDEX idx_synonyms_canonical_trgm
        ON ssa_listing_synonyms USING gin (canonical gin_trgm_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_synonyms_variant_trgm
        ON ssa_listing_synonyms USING gin (variant gin_trgm_ops)
        """
    )


def downgrade() -> None:
    op.drop_table("ssa_listing_synonyms")
    op.execute("DROP INDEX IF EXISTS idx_ssa_listings_summary_embed")
    op.execute("ALTER TABLE ssa_listings DROP COLUMN IF EXISTS summary_embedding")
