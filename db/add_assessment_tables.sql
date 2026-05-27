-- ============================================================================
-- Layers 2 + 3 of the matcher audit trail:
--   • leaf_assessments / leaf_assessment_evidence  (per-criterion AI decision)
--   • listing_assessments                          (per-listing final + groundedness)
--   • case_summaries                               (one row per case, exec view)
--
-- Layer 1 (chunk_leaf_matches) was added by add_chunk_leaf_matches_table.sql.
--
-- The three layers form a complete audit trail for each matcher run:
--   "for case C, leaf L (criterion):
--      retrieval (Layer 1) surfaced chunks K1..K25 at similarities S1..S25,
--      assessment (Layer 2) decided 'met' citing K3 and K7,
--      summary    (Layer 3) rolled it up: listing Meets, groundedness 0.78."
--
-- All tables are idempotent per (case, ...) — re-running the matcher on a
-- case does DELETE WHERE case_id then INSERT. We don't keep per-run history;
-- if A/B run history is needed later, add a matcher_run_id column.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Layer 2.a  leaf_assessments  — one row per (case, leaf criterion)
-- ----------------------------------------------------------------------------
-- Records the LLM's per-criterion verdict from pipeline/evaluate.py.
-- "leaf" = a leaf node of the criterion tree = one atomic criterion the
-- reviewer must check.
--
-- evidence_strength is a categorical bucket we compute from the verdict +
-- evidence count + top-retrieval similarity. Lets reviewers sort their queue
-- by "which leaves does the AI sound confident about, vs. which need human
-- investigation?"
CREATE TABLE IF NOT EXISTS leaf_assessments (
    id                      BIGSERIAL    PRIMARY KEY,
    case_id                 INT          NOT NULL REFERENCES cases(id)                ON DELETE CASCADE,
    -- ssa_listings.id and ssa_listing_criteria.id are UUID (see schema_v1.sql),
    -- so these FK columns must be UUID too. Case-side PKs (cases.id, chunks.id)
    -- are SERIAL/INT.
    listing_id              UUID         NOT NULL REFERENCES ssa_listings(id)         ON DELETE CASCADE,
    leaf_id                 UUID         NOT NULL REFERENCES ssa_listing_criteria(id) ON DELETE CASCADE,

    leaf_path               VARCHAR(120),                       -- e.g. "13.18.A.1"
    criterion_text          TEXT,                                -- snapshot of leaf.criterion at eval time

    verdict                 VARCHAR(20)  NOT NULL,
                                        -- 'met' | 'not_met' | 'insufficient'
                                        -- (mirror of LeafResult.verdict)
    rationale               TEXT,                                -- AI's 1-2 sentence explanation
    evidence_strength       VARCHAR(20),
                                        -- 'strong' | 'moderate' | 'weak' | 'none'
                                        -- (computed; see pipeline/groundedness.py)

    n_evidence_cited        INT          NOT NULL DEFAULT 0,    -- = len(LeafResult.evidence)
    n_chunks_retrieved      INT          NOT NULL DEFAULT 0,    -- size of retrieval set fed to LLM
    top_chunk_similarity    REAL,                                -- similarity of rank-1 retrieved chunk

    model                   VARCHAR(64),                         -- e.g. "claude-sonnet-4-6"
    elapsed_seconds         REAL,                                -- wall-clock for the LLM call

    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (case_id, leaf_id)
);

CREATE INDEX IF NOT EXISTS idx_la_case
    ON leaf_assessments(case_id);
CREATE INDEX IF NOT EXISTS idx_la_case_listing
    ON leaf_assessments(case_id, listing_id);
CREATE INDEX IF NOT EXISTS idx_la_verdict
    ON leaf_assessments(verdict);
CREATE INDEX IF NOT EXISTS idx_la_strength
    ON leaf_assessments(evidence_strength);
CREATE INDEX IF NOT EXISTS idx_la_leaf
    ON leaf_assessments(leaf_id);
    -- supports "across cases, how does the AI decide leaf X" analytics


-- ----------------------------------------------------------------------------
-- Layer 2.b  leaf_assessment_evidence — citations attached to an assessment
-- ----------------------------------------------------------------------------
-- One row per cited chunk. quote is the verbatim text fragment the LLM
-- extracted (already validated server-side in evaluate.py: must be a literal
-- substring of the chunk's text).
CREATE TABLE IF NOT EXISTS leaf_assessment_evidence (
    id              BIGSERIAL    PRIMARY KEY,
    assessment_id   BIGINT       NOT NULL REFERENCES leaf_assessments(id) ON DELETE CASCADE,
    chunk_id        INT          NOT NULL REFERENCES chunks(id)           ON DELETE CASCADE,
    cite_order      INT          NOT NULL,        -- 1-based, order of LLM output

    quote           TEXT         NOT NULL,        -- verbatim substring of chunks.text
    relevance       TEXT,                          -- "why this proves the verdict"

    UNIQUE (assessment_id, cite_order)
);

CREATE INDEX IF NOT EXISTS idx_lae_assessment
    ON leaf_assessment_evidence(assessment_id);
CREATE INDEX IF NOT EXISTS idx_lae_chunk
    ON leaf_assessment_evidence(chunk_id);
    -- supports "which leaves cited this chunk" reverse lookups


-- ----------------------------------------------------------------------------
-- Layer 3.a  listing_assessments — per-listing final verdict + groundedness
-- ----------------------------------------------------------------------------
-- The roll-up of all leaf_assessments under one listing's criterion tree,
-- via the 3-valued AND/OR consolidation in pipeline/consolidate.py.
--
-- groundedness_score (0..1) is a composite:
--    0.5 * frac_leaves_decided
--  + 0.3 * avg_top_similarity
--  + 0.2 * min(avg_evidence_per_leaf, 2) / 2
-- See pipeline/groundedness.py for the canonical formula. The component
-- columns are persisted alongside so analytics can re-derive or compare.
CREATE TABLE IF NOT EXISTS listing_assessments (
    id                      BIGSERIAL    PRIMARY KEY,
    case_id                 INT          NOT NULL REFERENCES cases(id)        ON DELETE CASCADE,
    listing_id              UUID         NOT NULL REFERENCES ssa_listings(id) ON DELETE CASCADE,

    form_verdict            VARCHAR(40)  NOT NULL,
                            -- 'Meets'
                            -- | 'Does not meet/equal'
                            -- | 'Insufficient evidence (review chart)'
    root_verdict            VARCHAR(20)  NOT NULL,
                            -- raw 3-valued state: 'met' | 'not_met' | 'insufficient'

    groundedness_score      REAL,            -- composite 0..1
    frac_leaves_decided     REAL,            -- (n_met + n_not_met) / n_leaves
    avg_top_similarity      REAL,            -- mean rank-1 retrieval similarity
    avg_evidence_per_leaf   REAL,            -- mean cited evidence count

    n_leaves                INT          NOT NULL,
    n_met                   INT          NOT NULL DEFAULT 0,
    n_not_met               INT          NOT NULL DEFAULT 0,
    n_insufficient          INT          NOT NULL DEFAULT 0,

    -- Where on the candidate shortlist this listing landed
    candidate_score         REAL,
    candidate_rank          INT,             -- 1 = top-ranked candidate for this case

    -- Reviewer-facing 1-2 sentence summary (deterministic, see groundedness.py)
    decision_summary        TEXT,

    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (case_id, listing_id)
);

CREATE INDEX IF NOT EXISTS idx_la2_case
    ON listing_assessments(case_id);
CREATE INDEX IF NOT EXISTS idx_la2_verdict
    ON listing_assessments(form_verdict);
CREATE INDEX IF NOT EXISTS idx_la2_grounded
    ON listing_assessments(groundedness_score DESC);
CREATE INDEX IF NOT EXISTS idx_la2_case_rank
    ON listing_assessments(case_id, candidate_rank);


-- ----------------------------------------------------------------------------
-- Layer 3.b  case_summaries — one row per case, executive view
-- ----------------------------------------------------------------------------
-- The reviewer dashboard's first-stop row: "this case has N candidates,
-- M of them came back Meets, top result is listing X.YY, overall
-- groundedness G". Backs queries like "show me Cases where any_meets =
-- TRUE AND overall_groundedness >= 0.7" (high-confidence triage queue).
CREATE TABLE IF NOT EXISTS case_summaries (
    id                      BIGSERIAL    PRIMARY KEY,
    case_id                 INT          NOT NULL UNIQUE REFERENCES cases(id) ON DELETE CASCADE,

    top_listing_id          UUID         REFERENCES ssa_listings(id),
                                                 -- candidate_rank = 1 listing
    top_form_verdict        VARCHAR(40),
    any_meets               BOOLEAN      NOT NULL DEFAULT FALSE,

    n_candidates_evaluated  INT          NOT NULL,
    n_meets                 INT          NOT NULL DEFAULT 0,
    n_does_not_meet         INT          NOT NULL DEFAULT 0,
    n_insufficient          INT          NOT NULL DEFAULT 0,

    overall_groundedness    REAL,         -- mean of listing_assessments.groundedness_score

    summary_text            TEXT,         -- short reviewer-facing case summary

    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    elapsed_seconds         REAL
);

CREATE INDEX IF NOT EXISTS idx_cs_any_meets
    ON case_summaries(any_meets) WHERE any_meets = TRUE;
CREATE INDEX IF NOT EXISTS idx_cs_grounded
    ON case_summaries(overall_groundedness DESC);


-- ----------------------------------------------------------------------------
-- Confirm
-- ----------------------------------------------------------------------------
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
      'leaf_assessments',
      'leaf_assessment_evidence',
      'listing_assessments',
      'case_summaries'
  )
ORDER BY table_name;
-- Expected: 4 rows.
