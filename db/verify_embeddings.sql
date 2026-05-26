-- ============================================================================
-- Verify embeddings are populated and similarity search works end-to-end
-- ============================================================================
-- Run this in DBeaver after both embedding workers complete.
-- Section A covers listings (summary_embedding).
-- Section B covers criteria (criterion_embedding).
-- ============================================================================


-- ============================================================================
-- A. LISTINGS
-- ============================================================================

-- A.1 Column type should now be vector(1024)
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listings'::regclass
  AND attname = 'summary_embedding';
-- Expected: vector(1024)  (Bedrock Titan v2 — change to vector(384) if you
--                          switched back to sentence-transformers)

-- A.2 Coverage: all 120 listings should have an embedding
SELECT COUNT(*)                      AS total_listings,
       COUNT(summary_embedding)      AS with_embedding,
       COUNT(*) - COUNT(summary_embedding) AS pending
FROM ssa_listings;
-- Expected: 120 / 120 / 0

-- A.3 Spot-check one listing: dimension and L2 norm
-- (norm should be ~1.0 because the worker normalized; tiny float-precision drift is fine)
SELECT code,
       array_length(summary_embedding::float8[], 1) AS dim,
       round(sqrt(sum(v * v))::numeric, 4)          AS l2_norm
FROM ssa_listings,
     LATERAL unnest(summary_embedding::float8[]) AS v
WHERE code = '13.18'
GROUP BY code, summary_embedding;
-- Expected: dim=1024, l2_norm ≈ 1.0000

-- A.4 End-to-end smoke test: find the 5 listings most similar to 13.18
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

-- A.5 Confirm the listings ivfflat index exists
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ssa_listings'
  AND indexname = 'idx_ssa_listings_summary_embed';
-- Expected: one row showing USING ivfflat (summary_embedding vector_cosine_ops)


-- ============================================================================
-- B. CRITERIA (leaf criterion_embedding)
-- ============================================================================

-- B.1 Column type should be vector(1024)
SELECT format_type(atttypid, atttypmod) AS column_type
FROM pg_attribute
WHERE attrelid = 'ssa_listing_criteria'::regclass
  AND attname = 'criterion_embedding';
-- Expected: vector(1024)

-- B.2 Coverage: every leaf with non-null criterion_text should have an embedding.
-- Internal nodes stay NULL by design.
SELECT
    COUNT(*) FILTER (WHERE is_leaf = TRUE AND criterion_text IS NOT NULL) AS leaf_with_text,
    COUNT(criterion_embedding)                                            AS embedded,
    COUNT(*) FILTER (WHERE is_leaf = TRUE
                       AND criterion_text IS NOT NULL
                       AND criterion_embedding IS NULL)                   AS pending,
    COUNT(*) FILTER (WHERE is_leaf = FALSE)                               AS internal_nodes_skipped
FROM ssa_listing_criteria;
-- Expected: leaf_with_text == embedded; pending == 0; internal_nodes_skipped is the
-- count of internal nodes (these correctly stay NULL).

-- B.3 Spot-check 13.18 leaf "A" — the metastatic criterion
SELECT l.code, c.path,
       LEFT(c.criterion_text, 80) AS criterion_preview,
       array_length(c.criterion_embedding::float8[], 1) AS dim
FROM ssa_listing_criteria c
JOIN ssa_listings l ON l.id = c.listing_id
WHERE l.code = '13.18'
  AND c.path = 'A';
-- Expected: dim=1024 and the criterion text matches "Adenocarcinoma metastatic..."

-- B.4 End-to-end SQL similarity: find the 5 leaf criteria most similar to
-- 13.18 leaf "A" (metastatic adenocarcinoma). Use that leaf's own embedding
-- as the query vector — leaves about metastatic cancer should cluster.
WITH q AS (
    SELECT c.criterion_embedding AS qvec
    FROM ssa_listing_criteria c
    JOIN ssa_listings l ON l.id = c.listing_id
    WHERE l.code = '13.18' AND c.path = 'A'
)
SELECT l.code,
       c.path,
       LEFT(c.criterion_text, 70) AS criterion_preview,
       (1 - (c.criterion_embedding <=> q.qvec))::numeric(5,3) AS cosine_similarity
FROM ssa_listing_criteria c
JOIN ssa_listings l ON l.id = c.listing_id, q
WHERE c.is_leaf = TRUE
  AND c.criterion_embedding IS NOT NULL
ORDER BY c.criterion_embedding <=> q.qvec
LIMIT 5;
-- Expected: 13.18:A itself at top with similarity 1.000, then other metastatic /
-- distant-disease leaves from sibling cancer listings.

-- B.5 Confirm the criteria ivfflat index exists
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ssa_listing_criteria'
  AND indexname = 'idx_criteria_embedding';
-- Expected: one row showing USING ivfflat (criterion_embedding vector_cosine_ops)
