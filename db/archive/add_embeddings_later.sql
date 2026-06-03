-- ============================================================================
-- Deferred: enable pgvector + pg_trgm and swap embedding column
-- ============================================================================
-- Run this AFTER your DBA enables the `vector` and `pg_trgm` extensions on
-- the RDS parameter group (or installs them with rds_superuser).
--
-- This drops the placeholder BYTEA column, recreates it as a real
-- vector(384) column sized for sentence-transformers all-MiniLM-L6-v2,
-- and adds the ivfflat similarity index plus the gin_trgm indexes on
-- synonyms.
--
-- After this runs, you still need a separate embedding worker to populate
-- summary_embedding values. See db/compute_embeddings.py (to be written
-- after pgvector lands).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Verify they took
SELECT extname, extversion FROM pg_extension
WHERE extname IN ('vector', 'pg_trgm')
ORDER BY extname;

-- ----------------------------------------------------------------------------
-- Swap the placeholder column for a real vector column
-- ----------------------------------------------------------------------------
-- We DROP+ADD rather than ALTER TYPE because BYTEA → vector(384) is not a
-- defined conversion. The placeholder was always NULL, so no data is lost.
ALTER TABLE ssa_listings DROP COLUMN IF EXISTS summary_embedding;
ALTER TABLE ssa_listings ADD COLUMN summary_embedding vector(384);

-- ivfflat index for fast cosine similarity. lists=100 is reasonable for ~120
-- rows; tune up to ~sqrt(N) if the table grows substantially.
-- Note: this index only becomes useful after rows have summary_embedding
-- values populated.
CREATE INDEX IF NOT EXISTS idx_ssa_listings_summary_embed
    ON ssa_listings USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 100);

-- ----------------------------------------------------------------------------
-- pg_trgm fuzzy indexes on synonyms (used by retrieval queries)
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_synonyms_canonical_trgm
    ON ssa_listing_synonyms USING gin (canonical gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_synonyms_variant_trgm
    ON ssa_listing_synonyms USING gin (variant gin_trgm_ops);

-- Confirm everything is in place. Cast both detail columns to text so the
-- UNION can match types (array_agg returns name[]; format_type returns text).
SELECT 'extensions enabled'                                                  AS check_name,
       string_agg(extname, ', ' ORDER BY extname)                            AS detail
FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')
UNION ALL
SELECT 'summary_embedding column type'                                       AS check_name,
       format_type(atttypid, atttypmod)                                      AS detail
FROM pg_attribute
WHERE attrelid = 'ssa_listings'::regclass
  AND attname = 'summary_embedding';
