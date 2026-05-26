-- ============================================================================
-- Add criterion_embedding column to ssa_listing_criteria
-- ============================================================================
-- Run this in DBeaver before db/compute_criterion_embeddings.py.
--
-- Only LEAF criteria will be populated (is_leaf=TRUE with a non-NULL
-- criterion_text). Internal-node group labels stay NULL — they're navigation
-- structure, not match targets, and embedding short labels like "Two of the
-- following despite continuing treatment" produces noisy vectors.
--
-- Sized for Bedrock Titan Text Embeddings v2 (1024 dim) to match the
-- ssa_listings.summary_embedding column.
-- ============================================================================

ALTER TABLE ssa_listing_criteria
    ADD COLUMN IF NOT EXISTS criterion_embedding vector(1024);

-- ivfflat similarity index. lists=100 is appropriate for ~500-1000 rows.
CREATE INDEX IF NOT EXISTS idx_criteria_embedding
    ON ssa_listing_criteria USING ivfflat (criterion_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Confirm column added with correct type
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listing_criteria'::regclass
  AND attname = 'criterion_embedding';
-- Expected: vector(1024)

-- Confirm index exists
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ssa_listing_criteria'
  AND indexname = 'idx_criteria_embedding';
-- Expected: one row USING ivfflat (criterion_embedding vector_cosine_ops)
