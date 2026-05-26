-- ============================================================================
-- Swap summary_embedding from vector(384) (sentence-transformers) to
-- vector(1024) (Bedrock Titan Text Embeddings v2)
-- ============================================================================
-- Run this BEFORE db/compute_listing_embeddings.py if you've decided to use
-- Bedrock Titan instead of sentence-transformers. The two have different
-- dimensions so a column swap is required.
--
-- We DROP+ADD rather than ALTER TYPE — pgvector can't auto-convert between
-- different dimensions, and any existing vector(384) data would be wrong for
-- the new model anyway.
--
-- The ivfflat index gets recreated for the new dimension.
-- ============================================================================

DROP INDEX IF EXISTS idx_ssa_listings_summary_embed;

ALTER TABLE ssa_listings DROP COLUMN IF EXISTS summary_embedding;
ALTER TABLE ssa_listings ADD  COLUMN summary_embedding vector(1024);

CREATE INDEX idx_ssa_listings_summary_embed
    ON ssa_listings USING ivfflat (summary_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Confirm
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listings'::regclass
  AND attname = 'summary_embedding';
-- Expected: vector(1024)
