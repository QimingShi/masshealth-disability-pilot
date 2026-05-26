-- ============================================================================
-- Hot-fix: allegations.source_chunk_id FK must be ON DELETE SET NULL
-- ============================================================================
-- Symptom this fixes:
--   psycopg2.errors.ForeignKeyViolation: update or delete on table "chunks"
--   violates foreign key constraint "allegations_source_chunk_id_fkey" on
--   table "allegations"
--
-- Why this happens:
--   When you re-run load_case_to_db.py, the loader does DELETE-then-INSERT
--   on documents. Postgres cascades that DELETE to chunks (chunks.document_id
--   has ON DELETE CASCADE). BUT allegations.source_chunk_id originally had
--   no ON DELETE clause — defaulting to NO ACTION — so the cascade hits a
--   wall when an allegation from the previous run still points at a chunk
--   being deleted.
--
-- The right behavior for allegations:
--   ON DELETE SET NULL — keep the allegation text (it's the member's claim
--   and stands on its own) but clear the source pointer if the chunk goes
--   away. Re-running the load then re-inserts allegations with fresh chunk
--   pointers anyway.
-- ============================================================================

ALTER TABLE allegations
    DROP CONSTRAINT IF EXISTS allegations_source_chunk_id_fkey;

ALTER TABLE allegations
    ADD CONSTRAINT allegations_source_chunk_id_fkey
    FOREIGN KEY (source_chunk_id)
    REFERENCES chunks(id)
    ON DELETE SET NULL;

-- Confirm the constraint is now ON DELETE SET NULL
SELECT conname,
       pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'allegations'::regclass
  AND conname = 'allegations_source_chunk_id_fkey';
-- Expected: definition includes "ON DELETE SET NULL"
