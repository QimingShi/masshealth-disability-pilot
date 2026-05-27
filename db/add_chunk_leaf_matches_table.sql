-- ============================================================================
-- chunk_leaf_matches  — per-case retrieval cache linking chunks to criteria
-- ============================================================================
-- Each row records: "for case X, leaf criterion Y, chunk Z was retrieved with
-- cosine similarity S, at rank R among the top-K results."
--
-- Populated by run.py --from-db at retrieval time. Persisted for:
--   1. Downstream analytics — "which chunks tend to surface for criterion N
--      across many cases?" / "median similarity at rank 1 by body system"
--   2. Reviewer audit — "the AI said X about this leaf; here's exactly the
--      chunks it was looking at"
--   3. Replay / re-eval — feed the LLM the same chunks without re-running
--      SQL similarity (useful when iterating on prompt design)
--   4. Tuning K — measure how often the cited chunk landed below rank K
--
-- Idempotency: each match-run for a case does DELETE WHERE case_id; the
-- retrieval cache always reflects the most recent matcher run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS chunk_leaf_matches (
    id              BIGSERIAL    PRIMARY KEY,
    case_id         INT          NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    -- ssa_listing_criteria.id is UUID (see db/schema_v1.sql), so this must
    -- match. The case-side PKs (cases.id, chunks.id) are SERIAL/INT.
    leaf_id         UUID         NOT NULL REFERENCES ssa_listing_criteria(id) ON DELETE CASCADE,
    chunk_id        INT          NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    similarity      REAL         NOT NULL,           -- 1 - cosine_distance, range 0..1
    rank_in_leaf    INT          NOT NULL,           -- 1 = most similar for this leaf
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (case_id, leaf_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_clm_case
    ON chunk_leaf_matches(case_id);
CREATE INDEX IF NOT EXISTS idx_clm_case_leaf
    ON chunk_leaf_matches(case_id, leaf_id);
CREATE INDEX IF NOT EXISTS idx_clm_leaf_similarity
    ON chunk_leaf_matches(leaf_id, similarity DESC);
    -- supports "what chunks tend to surface for leaf X across cases" analytics
CREATE INDEX IF NOT EXISTS idx_clm_chunk
    ON chunk_leaf_matches(chunk_id);
    -- supports "what leaves did this chunk match against" reverse lookups

-- Confirm
SELECT 'chunk_leaf_matches' AS table_created,
       COUNT(*) AS existing_rows
FROM chunk_leaf_matches;
