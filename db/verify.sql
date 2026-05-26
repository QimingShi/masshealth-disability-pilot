-- ============================================================================
-- Verify SSA listings load
-- ============================================================================
-- Run this in DBeaver after `python load_all_listings.py` completes.
-- Confirms row counts, body-system distribution, and one listing's full tree.
-- ============================================================================

-- Row counts. Expected (full bundle of 120 listings):
--    ssa_listings              120
--    ssa_listing_criteria     ~813  (252 internal nodes + 561 leaves)
--    ssa_listing_synonyms    ~2885
SELECT 'ssa_listings'         AS table_name, COUNT(*) AS row_count FROM ssa_listings
UNION ALL
SELECT 'ssa_listing_criteria' AS table_name, COUNT(*) AS row_count FROM ssa_listing_criteria
UNION ALL
SELECT 'ssa_listing_synonyms' AS table_name, COUNT(*) AS row_count FROM ssa_listing_synonyms
ORDER BY table_name;


-- Listings per body system (sanity check that nothing got dropped)
SELECT body_system, COUNT(*) AS listings
FROM ssa_listings
GROUP BY body_system
ORDER BY listings DESC, body_system;


-- Spot-check: listing 13.18 should have a small AND/OR tree with PRECONDITION + 4 alternatives
SELECT l.code,
       l.title,
       l.body_system,
       COUNT(c.id) AS criterion_node_count
FROM ssa_listings l
LEFT JOIN ssa_listing_criteria c ON c.listing_id = l.id
WHERE l.code = '13.18'
GROUP BY l.code, l.title, l.body_system;


-- 13.18's full rule tree, sorted by path
SELECT c.path,
       c.is_leaf,
       c.logic_operator,
       LEFT(c.criterion_text, 80) AS criterion_preview,
       c.keywords IS NOT NULL     AS has_keywords
FROM ssa_listing_criteria c
JOIN ssa_listings l ON l.id = c.listing_id
WHERE l.code = '13.18'
ORDER BY c.path;


-- 13.18's synonyms
SELECT canonical, variant
FROM ssa_listing_synonyms s
JOIN ssa_listings l ON l.id = s.listing_id
WHERE l.code = '13.18'
ORDER BY canonical, variant;


-- Confirm summary_embedding column exists but is NULL everywhere
-- (it'll get populated after pgvector is enabled and embeddings are computed)
SELECT COUNT(*) AS listings_total,
       COUNT(summary_embedding) AS listings_with_embedding,
       COUNT(*) - COUNT(summary_embedding) AS listings_pending_embedding
FROM ssa_listings;
