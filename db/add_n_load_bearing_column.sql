-- ============================================================================
-- Add n_load_bearing column to listing_assessments
-- ============================================================================
-- Fix 2 in the groundedness scoring: avg_top_similarity now averages over
-- "load-bearing" leaves only — the leaves whose verdict actually drove the
-- consolidated root decision. See pipeline/consolidate.py::
-- load_bearing_leaf_paths for the algorithm.
--
-- This column tracks how many leaves were load-bearing, so reviewers can
-- distinguish "AI decided based on 1 of 6 criteria" (one OR-path satisfied)
-- from "AI decided based on 6 of 6 criteria" (an AND chain) — both can
-- legitimately produce a Meets verdict, but the triage implications differ.
--
-- Existing rows (populated under the old all-leaves formula) keep their
-- existing avg_top_similarity value but get NULL n_load_bearing. The
-- DELETE-then-INSERT idempotency in persist_listing_assessment means
-- re-running the matcher refreshes everything to the new semantics.
-- ============================================================================

ALTER TABLE listing_assessments
    ADD COLUMN IF NOT EXISTS n_load_bearing INT;

COMMENT ON COLUMN listing_assessments.n_load_bearing IS
    'Count of leaves whose verdict drove the root consolidation. avg_top_similarity is averaged over these leaves only. NULL on rows persisted before the load-bearing scoring change.';

-- Confirm
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'listing_assessments'
  AND column_name IN ('avg_top_similarity', 'n_load_bearing', 'n_leaves')
ORDER BY ordinal_position;
