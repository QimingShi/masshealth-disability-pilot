-- ============================================================================
-- Verify that summary_embedding values are populated and similarity works
-- ============================================================================

-- 1. Column type should now be vector(384)
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listings'::regclass
  AND attname = 'summary_embedding';
-- Expected: vector(1024)  (Bedrock Titan v2 — change to vector(384) if you
--                          switched back to sentence-transformers)

-- 2. Coverage: all 120 listings should have an embedding
SELECT COUNT(*)                      AS total_listings,
       COUNT(summary_embedding)      AS with_embedding,
       COUNT(*) - COUNT(summary_embedding) AS pending
FROM ssa_listings;
-- Expected: 120 / 120 / 0

-- 3. Spot-check one row: dimension and L2 norm
-- (norm should be ~1.0 because the worker normalized; tiny float-precision drift is fine)
SELECT code,
       array_length(summary_embedding::float8[], 1) AS dim,
       round(sqrt(sum(v * v))::numeric, 4)          AS l2_norm
FROM ssa_listings,
     LATERAL unnest(summary_embedding::float8[]) AS v
WHERE code = '13.18'
GROUP BY code, summary_embedding;
-- Expected: dim=1024, l2_norm ≈ 1.0000

-- 4. End-to-end smoke test: find the 5 listings most similar to 13.18
-- (use 13.18's own embedding as the query vector)
WITH q AS (
    SELECT summary_embedding AS qvec
    FROM ssa_listings
    WHERE code = '13.18'
)
SELECT l.code,
       l.body_system,
       LEFT(l.title, 60) AS title_preview,
       (1 - (l.summary_embedding <=> q.qvec))::numeric(5,3) AS cosine_similarity
FROM ssa_listings l, q
ORDER BY l.summary_embedding <=> q.qvec
LIMIT 5;
-- Expected: 13.18 itself at top with similarity 1.000, then related cancer
-- listings like 13.17 (small intestine), 13.21 (esophagus/stomach), etc.

-- 5. Confirm the ivfflat index exists
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ssa_listings'
  AND indexname = 'idx_ssa_listings_summary_embed';
-- Expected: one row showing USING ivfflat (summary_embedding vector_cosine_ops)
