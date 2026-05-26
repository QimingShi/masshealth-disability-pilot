-- ============================================================================
-- Verify case schema + sample data + cross-table cosine similarity
-- ============================================================================
-- Run this in DBeaver AFTER load_case_to_db.py completes for at least one case.
-- ============================================================================


-- ============================================================================
-- 1. All 6 tables exist
-- ============================================================================
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('cases', 'source_pdfs', 'documents', 'chunks',
                     'chunk_icd_codes', 'allegations')
ORDER BY table_name;
-- Expected: 6 rows


-- ============================================================================
-- 2. Row counts across all case-side tables
-- ============================================================================
SELECT 'cases'            AS table_name, COUNT(*) AS rows FROM cases
UNION ALL
SELECT 'source_pdfs',     COUNT(*) FROM source_pdfs
UNION ALL
SELECT 'documents',       COUNT(*) FROM documents
UNION ALL
SELECT 'chunks',          COUNT(*) FROM chunks
UNION ALL
SELECT 'chunk_icd_codes', COUNT(*) FROM chunk_icd_codes
UNION ALL
SELECT 'allegations',     COUNT(*) FROM allegations
ORDER BY table_name;


-- ============================================================================
-- 3. Cases summary
-- ============================================================================
SELECT case_id,
       status,
       intake_date,
       (SELECT COUNT(*) FROM source_pdfs WHERE case_id = c.id) AS pdfs,
       (SELECT COUNT(*) FROM documents   WHERE case_id = c.id) AS docs,
       (SELECT COUNT(*) FROM chunks      WHERE case_id = c.id) AS chunks,
       (SELECT COUNT(*) FROM allegations WHERE case_id = c.id) AS allegations
FROM cases c
ORDER BY intake_date DESC;


-- ============================================================================
-- 4. Chunk embedding coverage (should be 100% if --skip-embeddings was NOT used)
-- ============================================================================
SELECT COUNT(*)                    AS total_chunks,
       COUNT(embedding)            AS chunks_with_embedding,
       COUNT(*) - COUNT(embedding) AS chunks_pending
FROM chunks;


-- ============================================================================
-- 5. ICD codes found across cases — sanity check for extraction
-- ============================================================================
SELECT code,
       COUNT(*)                                    AS occurrences,
       COUNT(DISTINCT chunks.case_id)              AS cases_with_code
FROM chunk_icd_codes
JOIN chunks ON chunks.id = chunk_icd_codes.chunk_id
GROUP BY code
ORDER BY occurrences DESC, code
LIMIT 20;


-- ============================================================================
-- 6. Allegations summary
-- ============================================================================
SELECT c.case_id, a.source, a.text
FROM allegations a
JOIN cases c ON c.id = a.case_id
ORDER BY c.case_id, a.source, a.text;


-- ============================================================================
-- 7. THE MONEY QUERY — cross-table cosine similarity
-- ============================================================================
-- For the colorectal case, find the SSA listings most similar to each of the
-- top-allegation embeddings. Proves that case-side embeddings and reference-
-- data embeddings live in the same Titan v2 vector space.
--
-- Expected for redacted-001 (allegation "Colon adenocarcinoma" → listing
-- 13.18 large intestine cancer near 0.85+).
WITH case_allegations AS (
    SELECT a.text AS allegation_text, a.embedding AS qvec
    FROM allegations a
    JOIN cases c ON c.id = a.case_id
    WHERE c.case_id = 'redacted-001'
)
SELECT ca.allegation_text,
       l.code,
       LEFT(l.title, 50) AS listing_title,
       (1 - (l.summary_embedding <=> ca.qvec))::numeric(5,3) AS cosine_sim
FROM case_allegations ca,
     LATERAL (
       SELECT code, title, summary_embedding
       FROM ssa_listings
       ORDER BY summary_embedding <=> ca.qvec
       LIMIT 3
     ) l
ORDER BY ca.allegation_text, cosine_sim DESC;


-- ============================================================================
-- 8. Per-leaf retrieval preview — chunks-vs-criterion cosine
-- ============================================================================
-- Pick a leaf criterion (13.18 leaf A — adenocarcinoma metastatic to or
-- beyond regional lymph nodes) and find the 5 chunks in the colorectal
-- case most semantically similar to it. This is what the matcher's per-leaf
-- retrieval will eventually look like in pure SQL.
WITH leaf AS (
    SELECT c.criterion_embedding AS qvec, c.criterion_text
    FROM ssa_listing_criteria c
    JOIN ssa_listings l ON l.id = c.listing_id
    WHERE l.code = '13.18' AND c.path = 'A'
)
SELECT ch.chunk_id,
       ch.section,
       ch.page_start,
       LEFT(ch.text, 80) AS chunk_preview,
       (1 - (ch.embedding <=> leaf.qvec))::numeric(5,3) AS similarity
FROM chunks ch
JOIN cases cs ON cs.id = ch.case_id
JOIN leaf ON TRUE
WHERE cs.case_id = 'redacted-001'
  AND ch.embedding IS NOT NULL
ORDER BY ch.embedding <=> leaf.qvec
LIMIT 5;
-- Expected: top hit should be the CT impression chunk that mentions "presumed
-- small metastasis at the liver dome" — that's the smoking-gun chunk the
-- Python matcher previously found in-memory. With SQL similarity, you get
-- the same answer via one query.
