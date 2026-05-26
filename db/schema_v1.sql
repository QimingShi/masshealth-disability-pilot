-- ============================================================================
-- SSA listings schema, v1 — no pgvector, no pg_trgm dependencies
-- ============================================================================
-- Run this in DBeaver as user `shiq` on database `expedite`.
-- Creates three tables that hold the SSA Blue Book listings, criterion trees,
-- and per-listing synonym maps. Designed so `SSA JSON/load_all_listings.py`
-- runs against it unchanged.
--
-- Two columns are intentionally placeholder shapes:
--   ssa_listings.summary_embedding  BYTEA     ← will become vector(384) once
--                                                pgvector is enabled; see
--                                                db/add_embeddings_later.sql
--   No gin_trgm_ops indexes        deferred   ← also need pg_trgm extension
--
-- Idempotent: `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`
-- so re-running this is safe.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- ssa_listings  — one row per SSA listing (e.g. 13.18, 12.04, 1.15)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ssa_listings (
    id                  UUID            PRIMARY KEY,
    code                VARCHAR(16)     NOT NULL UNIQUE,
    title               TEXT            NOT NULL,
    body_system         VARCHAR(40)     NOT NULL,
    version             VARCHAR(16)     NOT NULL DEFAULT '1',
    summary             TEXT            NOT NULL,
    rule_json           JSON            NOT NULL,

    -- Placeholder until pgvector is enabled. The loader writes NULL here.
    -- When pgvector is available, db/add_embeddings_later.sql drops this
    -- column and recreates it as vector(384).
    summary_embedding   BYTEA,

    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ssa_listings_body_system
    ON ssa_listings(body_system);


-- ----------------------------------------------------------------------------
-- ssa_listing_criteria  — one row per node (internal + leaf) in rule_json
-- ----------------------------------------------------------------------------
-- The recursive AND/OR tree from each listing's rule_json gets flattened here
-- with parent_id pointers so we can answer "give me all leaves of listing X"
-- without re-parsing JSON. The original tree is also preserved on
-- ssa_listings.rule_json for fast tree-walks during consolidation.
CREATE TABLE IF NOT EXISTS ssa_listing_criteria (
    id                          UUID            PRIMARY KEY,
    listing_id                  UUID            NOT NULL
                                                REFERENCES ssa_listings(id)
                                                ON DELETE CASCADE,
    parent_id                   UUID                     REFERENCES ssa_listing_criteria(id)
                                                ON DELETE CASCADE,
    path                        VARCHAR(64)     NOT NULL,    -- "ROOT", "A.1", "B.3_OR_B.4"
    is_leaf                     BOOLEAN         NOT NULL,
    logic_operator              VARCHAR(8),                  -- "AND" | "OR" | NULL (leaves)
    criterion_text              TEXT,                        -- leaf statement OR internal group label
    keywords                    JSON,                        -- [str, ...] on leaves; NULL on internal
    duration_months_required    INT,                         -- e.g. 12 = condition must persist >= 12 mo
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- One node per (listing, path); makes the loader idempotent
    CONSTRAINT uq_listing_path UNIQUE (listing_id, path)
);

CREATE INDEX IF NOT EXISTS idx_criteria_listing
    ON ssa_listing_criteria(listing_id);
CREATE INDEX IF NOT EXISTS idx_criteria_parent
    ON ssa_listing_criteria(parent_id);
CREATE INDEX IF NOT EXISTS idx_criteria_listing_leaf
    ON ssa_listing_criteria(listing_id, is_leaf);


-- ----------------------------------------------------------------------------
-- ssa_listing_synonyms  — per-listing canonical/variant maps
-- ----------------------------------------------------------------------------
-- Scoped per-listing so "compression" in 1.15 (nerve root compression) doesn't
-- match against the same word in 4.11 (compression stockings).
CREATE TABLE IF NOT EXISTS ssa_listing_synonyms (
    id                  UUID            PRIMARY KEY,
    listing_id          UUID            NOT NULL
                                        REFERENCES ssa_listings(id)
                                        ON DELETE CASCADE,
    canonical           VARCHAR(128)    NOT NULL,
    variant             VARCHAR(256)    NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_listing_canonical_variant
        UNIQUE (listing_id, canonical, variant)
);

CREATE INDEX IF NOT EXISTS idx_synonyms_listing
    ON ssa_listing_synonyms(listing_id);

-- Plain B-tree on canonical/variant for now (gives us equality + prefix
-- lookup). pg_trgm-based fuzzy indexes will be added in add_embeddings_later.sql
-- once that extension is enabled.
CREATE INDEX IF NOT EXISTS idx_synonyms_canonical
    ON ssa_listing_synonyms(canonical);
CREATE INDEX IF NOT EXISTS idx_synonyms_variant
    ON ssa_listing_synonyms(variant);


-- ----------------------------------------------------------------------------
-- Confirm what just got created
-- ----------------------------------------------------------------------------
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'ssa_listing%'
ORDER BY table_name;
