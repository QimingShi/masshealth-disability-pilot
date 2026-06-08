-- ============================================================================
-- Change documents.source_pdf_id FK from default (NO ACTION) to ON DELETE
-- SET NULL.
--
-- Background: insert_source_pdfs is idempotent (delete + insert per case).
-- documents.source_pdf_id became actively populated in commit 2a6bd5e
-- (page-range inference), which exposed an FK-cascade order bug:
-- re-running ingest on a case fails with ForeignKeyViolation because
-- the source_pdfs DELETE happens before documents are torn down.
--
-- The semantics should be: source_pdfs is the "raw input" — if it's
-- recreated, downstream documents records can keep existing but lose
-- the link temporarily; insert_documents repopulates source_pdf_id via
-- page-range inference on every load.
--
-- This migration is safe to apply on a live DB. Re-running it is a no-op
-- because the constraint name is the same.
-- ============================================================================

\timing on

BEGIN;

-- Drop the old constraint (whatever delete-rule it had — usually NO ACTION).
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_pdf_id_fkey;

-- Re-add with ON DELETE SET NULL.
ALTER TABLE documents
    ADD CONSTRAINT documents_source_pdf_id_fkey
    FOREIGN KEY (source_pdf_id)
    REFERENCES source_pdfs(id)
    ON DELETE SET NULL;

COMMIT;

-- Verify
SELECT con.conname, pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class    rel ON rel.oid = con.conrelid
WHERE rel.relname = 'documents'
  AND con.conname = 'documents_source_pdf_id_fkey';
-- Expected: ...REFERENCES source_pdfs(id) ON DELETE SET NULL
