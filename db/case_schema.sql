-- ============================================================================
-- Case-side schema — what the ingest pipeline produces per packet
-- ============================================================================
-- Five tables: cases, source_pdfs, documents, chunks, chunk_icd_codes,
-- allegations. Matches the shape of data/<case>/chunks.json so we can
-- migrate already-ingested cases without re-running Textract.
--
-- chunks.embedding uses vector(1024) — same dimension as listings.summary
-- and criteria.criterion, so all three live in the same Titan v2 space and
-- can be cosine-compared directly in SQL.
--
-- Run AFTER schema_v1.sql + the listings work — this assumes pgvector is
-- already installed (which it is on expedite as of session-end May 2026).
--
-- Idempotent: every CREATE uses IF NOT EXISTS.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- cases  — one row per packet
-- ----------------------------------------------------------------------------
-- The "intake" row. Tracks workflow status from pending through signed.
-- A case can have multiple source PDFs (typically an allegation supplement
-- plus medical evidence), multiple sub-documents within those, hundreds of
-- chunks, and one or more match_runs against the SSA listings.
CREATE TABLE IF NOT EXISTS cases (
    id              SERIAL          PRIMARY KEY,
    case_id         VARCHAR(64)     NOT NULL UNIQUE,
                                    -- the human-readable identifier
                                    -- (e.g. "redacted-001", "mepers-001")
    member_id       VARCHAR(64),    -- internal MMID; consider hashing for privacy
    intake_date     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status          VARCHAR(32)     NOT NULL DEFAULT 'pending',
                                    -- workflow:
                                    --   pending     no PDFs uploaded yet
                                    --   ingesting   Textract running
                                    --   ingested    chunks loaded; ready for matcher
                                    --   matching    pipeline running
                                    --   matched     forms generated, awaiting reviewer
                                    --   review      reviewer reading the forms
                                    --   signed      reviewer signed off
                                    --   archived    closed out
    reviewer_id     VARCHAR(64),    -- assigned reviewer (set when status='review')
    signed_at       TIMESTAMPTZ,
    signed_by       VARCHAR(64),
    notes           TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cases_status
    ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_reviewer
    ON cases(reviewer_id) WHERE reviewer_id IS NOT NULL;


-- ----------------------------------------------------------------------------
-- source_pdfs  — one row per uploaded PDF
-- ----------------------------------------------------------------------------
-- Multiple per case for multi-PDF ingest. Each row records where the file
-- came from (S3), where it lives locally, and where it lands in the
-- concatenated source.pdf (combined_page_start..combined_page_end).
CREATE TABLE IF NOT EXISTS source_pdfs (
    id                    SERIAL       PRIMARY KEY,
    case_id               INT          NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    role                  VARCHAR(40),  -- "allegation_source" | "medical_evidence" | ...
    s3_bucket             VARCHAR(255),
    s3_key                TEXT,
    local_path            TEXT,         -- relative path under _phi/<case_id>/
    combined_page_start   INT,           -- 1-based offset in the combined source.pdf
    combined_page_end     INT,
    page_count            INT,
    textract_job_id       VARCHAR(64),  -- which Textract job processed this
    sha256                VARCHAR(64),   -- file hash for change detection
    ingested_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_source_pdfs_case
    ON source_pdfs(case_id);


-- ----------------------------------------------------------------------------
-- documents  — sub-documents detected within a case packet
-- ----------------------------------------------------------------------------
-- Output of detect_sub_documents() in pipeline/ingest_real.py. One row per
-- LAYOUT_TITLE-bounded sub-document. Pages reference the combined source.pdf.
CREATE TABLE IF NOT EXISTS documents (
    id                 SERIAL       PRIMARY KEY,
    case_id            INT          NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    doc_id             VARCHAR(64)  NOT NULL,          -- e.g. "doc-01"
    doc_type           VARCHAR(64),                     -- oncology_progress_note, ct_imaging_report, ...
    doc_title          TEXT,
    encounter_date     DATE,
    facility           VARCHAR(255),
    page_start         INT,                             -- in combined source.pdf
    page_end           INT,
    source_pdf_id      INT          REFERENCES source_pdfs(id) ON DELETE SET NULL,
                                    -- which underlying PDF this sub-doc came from
                                    -- (matters for multi-PDF ingest). ON DELETE
                                    -- SET NULL because insert_source_pdfs is
                                    -- idempotent (delete + insert) and we don't
                                    -- want documents to block its re-run; the
                                    -- next insert_documents will repopulate
                                    -- via page-range inference.

    UNIQUE (case_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_case
    ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_type
    ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_encounter_date
    ON documents(encounter_date) WHERE encounter_date IS NOT NULL;


-- ----------------------------------------------------------------------------
-- chunks  — the building blocks of every case
-- ----------------------------------------------------------------------------
-- Output of chunk_by_layout() in pipeline/ingest_textract.py. Each row is
-- one section of a sub-document (HPI, Impression, a single lab panel, an
-- impression numbered finding, etc.) with the verbatim OCR text, bounding
-- boxes for citation highlighting, OCR confidence, and a Titan v2 embedding.
--
-- Critical: chunks.embedding is vector(1024), same dim as
-- ssa_listings.summary_embedding and ssa_listing_criteria.criterion_embedding.
-- This makes SQL similarity search across all three trivial.
CREATE TABLE IF NOT EXISTS chunks (
    id                  SERIAL       PRIMARY KEY,
    case_id             INT          NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    document_id         INT          NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id            VARCHAR(128) NOT NULL,         -- "doc-01:p1:HPI:0"

    section             VARCHAR(120),                   -- "HPI", "Impression", "Labs — CMP"
                                                        -- Capped at 120 in pipeline/ingest_textract.
                                                        -- _normalize_section_name so the printed
                                                        -- form-instruction text doesn't overflow.
    page_start          INT,
    page_end            INT,
    bbox                JSONB,                          -- {"<page>": [left,top,right,bottom], ...}
                                                        -- in Textract's 0-1 normalized coords

    text                TEXT         NOT NULL,
    ocr_confidence      REAL,                           -- average WORD confidence in this chunk

    layout_block_type   VARCHAR(40),                    -- "LAYOUT_TEXT", "LAYOUT_TABLE", ...
    is_table            BOOLEAN      DEFAULT FALSE,

    embedding           vector(1024),                   -- Bedrock Titan v2

    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (case_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_case
    ON chunks(case_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc
    ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section
    ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_chunks_low_confidence
    ON chunks(case_id) WHERE ocr_confidence < 90;
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
    ON chunks USING gin (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200);
    -- lists=200 is a starting value; tune up to ~sqrt(N) if chunk count grows
    -- substantially (e.g. >40K chunks suggests lists=200+).


-- ----------------------------------------------------------------------------
-- chunk_icd_codes  — structured ICD-10 extractions per chunk
-- ----------------------------------------------------------------------------
-- Output of extract_icd_codes() in pipeline/chunks.py. Lets the matcher's
-- candidate-identification signal #2 (ICD → body system filter) be a JOIN
-- instead of an in-process regex scan.
CREATE TABLE IF NOT EXISTS chunk_icd_codes (
    id                  SERIAL       PRIMARY KEY,
    chunk_id            INT          NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    code                VARCHAR(16)  NOT NULL,         -- "C18.9", "E87.6"
    description         TEXT,                           -- best-effort context from chart
    position_in_text    INT,                            -- character offset within chunks.text

    UNIQUE (chunk_id, code, position_in_text)
);

CREATE INDEX IF NOT EXISTS idx_chunk_icd_code
    ON chunk_icd_codes(code);
CREATE INDEX IF NOT EXISTS idx_chunk_icd_prefix
    ON chunk_icd_codes(LEFT(code, 1));
    -- supports "all C* codes for body system 'cancer'" type queries


-- ----------------------------------------------------------------------------
-- allegations  — what the member is claiming (and where it came from)
-- ----------------------------------------------------------------------------
-- Output of auto_extract_allegations() in pipeline/ingest_real.py, plus
-- manual reviewer additions/edits. Each allegation is linked back to the
-- chunk it was extracted from, so we can show the reviewer where the
-- allegation came from in the chart.
--
-- The embedding lets us do cosine similarity against listing summaries
-- (the candidate-identification signal #1) in pure SQL.
CREATE TABLE IF NOT EXISTS allegations (
    id                  SERIAL       PRIMARY KEY,
    case_id             INT          NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    text                TEXT         NOT NULL,
    source              VARCHAR(40),                    -- "chief_complaint" | "visit_diagnoses" |
                                                        -- "past_medical_history" |
                                                        -- "problem_list"       (Problem List / Active Problems) |
                                                        -- "supplement_part1"   (Part 1: health problems table) |
                                                        -- "supplement_part2"   (Part 2: provider/reason table) |
                                                        -- "supplement_form"    (regex match in supplement section) |
                                                        -- "narrative_phrase" | "manual"
    source_chunk_id     INT          REFERENCES chunks(id) ON DELETE SET NULL,
                                                        -- Allegation text is the member's claim and
                                                        -- stands on its own; if its source chunk gets
                                                        -- deleted (re-OCR), allegation survives with
                                                        -- a NULL source pointer. Without this clause,
                                                        -- re-running load_case_to_db.py raises a FK
                                                        -- violation because chunks can't be cascaded
                                                        -- past existing allegations.
    manually_curated    BOOLEAN      DEFAULT FALSE,
    embedding           vector(1024),                   -- Bedrock Titan v2
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_allegations_case
    ON allegations(case_id);
CREATE INDEX IF NOT EXISTS idx_allegations_embedding
    ON allegations USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);


-- ----------------------------------------------------------------------------
-- Confirm what we created
-- ----------------------------------------------------------------------------
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('cases', 'source_pdfs', 'documents', 'chunks',
                     'chunk_icd_codes', 'allegations')
ORDER BY table_name;
-- Expected: 6 rows (the 5 case-side tables I described, plus the migration log
-- if you have one).
