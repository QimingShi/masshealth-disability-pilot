-- ============================================================================
-- Backfill: documents.source_pdf_id was NULL for cases ingested before the
-- field was being set. Infer it from page overlap with source_pdfs.
--
-- For each document with NULL source_pdf_id, find the source_pdf in the
-- same case whose combined_page_start..combined_page_end range contains
-- the document's page_start. If a document spans pages from two source
-- PDFs (rare for sub-docs but possible), we attribute it to the source
-- PDF holding its FIRST page.
--
-- Idempotent and safe: only touches NULL rows. Skips cases where either
-- side is missing the page-range info needed to attribute.
-- ============================================================================

\timing on

-- Preview (no writes): what would get updated?
SELECT
    d.case_id,
    d.doc_id,
    d.page_start,
    d.page_end,
    sp.id           AS would_set_source_pdf_id,
    sp.role         AS source_pdf_role,
    sp.combined_page_start,
    sp.combined_page_end
FROM documents d
LEFT JOIN source_pdfs sp
    ON sp.case_id = d.case_id
   AND d.page_start IS NOT NULL
   AND sp.combined_page_start IS NOT NULL
   AND sp.combined_page_end   IS NOT NULL
   AND d.page_start BETWEEN sp.combined_page_start AND sp.combined_page_end
WHERE d.source_pdf_id IS NULL
ORDER BY d.case_id, d.page_start
LIMIT 100;

-- The actual backfill. Wrapped in a transaction so you can ROLLBACK if the
-- preview above looked wrong.
BEGIN;

UPDATE documents d
SET source_pdf_id = sp.id
FROM source_pdfs sp
WHERE d.source_pdf_id IS NULL
  AND d.case_id = sp.case_id
  AND d.page_start IS NOT NULL
  AND sp.combined_page_start IS NOT NULL
  AND sp.combined_page_end   IS NOT NULL
  AND d.page_start BETWEEN sp.combined_page_start AND sp.combined_page_end;

-- Report how many rows got linked + how many remain NULL (those would be
-- cases where the document's page_start or the source_pdf's combined_page
-- range is missing — usually older / hand-loaded cases).
SELECT
    COUNT(*) FILTER (WHERE source_pdf_id IS NOT NULL) AS linked,
    COUNT(*) FILTER (WHERE source_pdf_id IS NULL)     AS still_null,
    COUNT(*)                                          AS total_documents
FROM documents;

COMMIT;
-- If anything looked wrong above, run ROLLBACK; instead of COMMIT;
