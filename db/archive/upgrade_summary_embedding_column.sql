-- ============================================================================
-- Swap ssa_listings.summary_embedding from BYTEA placeholder to FLOAT8 array
-- ============================================================================
-- Run this once before computing embeddings. If pgvector ever gets installed,
-- you can convert the FLOAT8[] column to vector(384) with:
--   ALTER TABLE ssa_listings
--       ALTER COLUMN summary_embedding TYPE vector(384) USING summary_embedding::vector;
-- The existing FLOAT8[] values convert cleanly to vector.
-- ============================================================================

ALTER TABLE ssa_listings DROP COLUMN IF EXISTS summary_embedding;
ALTER TABLE ssa_listings ADD  COLUMN summary_embedding FLOAT8[];

-- Verify the column type changed
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listings'::regclass
  AND attname = 'summary_embedding';
-- Expected: double precision[]
