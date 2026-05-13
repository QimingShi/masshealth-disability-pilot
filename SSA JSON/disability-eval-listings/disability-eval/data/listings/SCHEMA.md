# Listings — Database Schema

The listing JSON files in `data/listings/` are the source of truth. They get
loaded into four normalized tables that downstream code can query efficiently
without re-parsing JSON every request.

## Why split it across four tables?

Three different consumers want three different shapes of the same data:

1. **Triage / candidate identification** (`identify_candidate_listings` in
   `services/listings.py`) needs the listing-level summary embedding. It hits
   `ssa_listings` once per case.
2. **Per-leaf retrieval** (`retrieve_for_criterion` in `services/retrieval.py`)
   needs (a) the exact `criterion` text to embed for the original-query
   variant, (b) the leaf's `keywords[]` for the keyword variant, and (c) the
   listing's synonym map for the synonym variant. This needs `criteria` +
   `synonyms`.
3. **Consolidation** (`consolidate_listing` in `services/listings.py`) walks
   the AND/OR tree and combines per-leaf verdicts. This is faster against the
   raw `rule_json` jsonb than against reassembled SQL rows — so we store both.

Storing the tree both as denormalized rows AND as the original `rule_json`
costs a few extra bytes per listing and saves a recursive CTE on every case.
That trade is correct here.

## Schema

```sql
-- ============================================================
--  ssa_listings  — one row per listing (1.15, 12.04, etc.)
-- ============================================================
CREATE TABLE ssa_listings (
    id              SERIAL PRIMARY KEY,
    code            VARCHAR(16)     NOT NULL UNIQUE,
    title           TEXT            NOT NULL,
    body_system     VARCHAR(40)     NOT NULL,
    version         INT             NOT NULL DEFAULT 1,
    summary         TEXT            NOT NULL,
    summary_embedding   VECTOR(1024),  -- pgvector; populated by listings_embed worker
    rule_json       JSONB           NOT NULL,  -- full tree, used by consolidator
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ssa_listings_body_system ON ssa_listings(body_system);
CREATE INDEX idx_ssa_listings_summary_embed
    ON ssa_listings USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================
--  ssa_listing_criteria  — one row per node (internal + leaf)
-- ============================================================
-- The recursive tree gets flattened here with parent_id pointers so we can
-- query "all leaves of listing X" without parsing JSON. We DON'T use this to
-- reconstruct the tree at consolidation time — we walk rule_json directly
-- because Postgres recursive CTEs are slower than walking parsed JSON in
-- Python.
CREATE TABLE ssa_listing_criteria (
    id                  SERIAL PRIMARY KEY,
    listing_id          INT     NOT NULL REFERENCES ssa_listings(id) ON DELETE CASCADE,
    parent_id           INT              REFERENCES ssa_listing_criteria(id) ON DELETE CASCADE,
    path                VARCHAR(64)  NOT NULL,        -- "ROOT", "A.1", "B.3_OR_B.4", etc.
    is_leaf             BOOLEAN      NOT NULL,
    logic               VARCHAR(8),                    -- "AND" | "OR" | NULL (leaves)
    criterion           TEXT,                          -- leaf criterion text, OR group label on internal nodes
    duration_months     INT,                           -- only on leaves carrying durational criterion
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- One node per (listing, path); idempotent on re-load
    CONSTRAINT uq_listing_path UNIQUE (listing_id, path)
);

CREATE INDEX idx_criteria_listing       ON ssa_listing_criteria(listing_id);
CREATE INDEX idx_criteria_parent        ON ssa_listing_criteria(parent_id);
CREATE INDEX idx_criteria_listing_leaf  ON ssa_listing_criteria(listing_id, is_leaf);

-- ============================================================
--  ssa_listing_keywords  — one row per (leaf, keyword)
-- ============================================================
-- Denormalized for fast text search. The retrieval keyword variant builds its
-- query string from keywords[] for a specific criterion; we keep a separate
-- table rather than a Postgres array on criteria so we can:
--   1. add a per-keyword embedding cache later if useful
--   2. do `SELECT criterion_id FROM ssa_listing_keywords WHERE keyword ILIKE $1`
--      for diagnostic queries like "which listings use the term 'Spurling'?"
CREATE TABLE ssa_listing_keywords (
    id              SERIAL PRIMARY KEY,
    criterion_id    INT     NOT NULL REFERENCES ssa_listing_criteria(id) ON DELETE CASCADE,
    keyword         VARCHAR(128)    NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_criterion_keyword UNIQUE (criterion_id, keyword)
);

CREATE INDEX idx_keywords_criterion     ON ssa_listing_keywords(criterion_id);
CREATE INDEX idx_keywords_text_trgm     ON ssa_listing_keywords USING gin (keyword gin_trgm_ops);

-- ============================================================
--  ssa_listing_synonyms  — one row per (listing, canonical, variant)
-- ============================================================
-- Per-listing scoping is intentional — "compression" in 1.15 means nerve root
-- compression, in 4.11 it means compression stocking. Storing globally would
-- contaminate the synonym query. The retrieval synonym-variant query joins on
-- (listing_id, term-found-in-criterion) → variants[] and expands inline.
CREATE TABLE ssa_listing_synonyms (
    id              SERIAL PRIMARY KEY,
    listing_id      INT     NOT NULL REFERENCES ssa_listings(id) ON DELETE CASCADE,
    canonical       VARCHAR(128)    NOT NULL,    -- the term as it appears in chart language
    variant         VARCHAR(256)    NOT NULL,    -- alternative phrasing / abbreviation / lay term
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_listing_canonical_variant UNIQUE (listing_id, canonical, variant)
);

CREATE INDEX idx_synonyms_listing       ON ssa_listing_synonyms(listing_id);
CREATE INDEX idx_synonyms_canonical_trgm ON ssa_listing_synonyms USING gin (canonical gin_trgm_ops);
CREATE INDEX idx_synonyms_variant_trgm   ON ssa_listing_synonyms USING gin (variant gin_trgm_ops);
```

## Idempotency contract

The seed loader (`scripts/seed_listings.py`) is idempotent: running it twice on
the same file produces the same DB state. Specifically:

- `ssa_listings`: `INSERT ... ON CONFLICT (code) DO UPDATE` — bumps title,
  body_system, version, summary, rule_json. The summary embedding is
  invalidated and re-queued (set to NULL).
- `ssa_listing_criteria`: rows for the listing are deleted first, then
  re-inserted. The `(listing_id, path)` UNIQUE constraint guards against
  duplicates within a load. We delete-then-insert (rather than upsert) because
  paths can move between versions (e.g., adding a new node renumbers
  siblings); upserting would leave orphan rows.
- `ssa_listing_keywords` and `ssa_listing_synonyms`: same delete-and-insert
  pattern, scoped to the listing being loaded.

Concretely: `seed_listings.py` wraps each listing in a single transaction, so
a half-loaded listing never appears in the DB.

## What lives in `rule_json` vs SQL

The whole tree is duplicated — once flat in `ssa_listing_criteria`, once
nested in `ssa_listings.rule_json`. The flat table answers questions like
"give me every leaf of listing 1.15 with its keywords"; the nested column
answers "walk this tree applying these per-leaf verdicts." Both are useful.
Keep them in sync via the loader; never edit one without the other.

## Migrations

The DDL above belongs in `alembic/versions/0002_listings.py`. The original
`0001_initial.py` already creates the `cases`, `documents`, `chunks`, and
`audit_logs` tables; this migration adds the four listing tables on top.
