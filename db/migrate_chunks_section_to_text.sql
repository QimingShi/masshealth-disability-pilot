-- ============================================================================
-- Widen chunks.section from VARCHAR(120) to TEXT.
--
-- Form headers from Textract sometimes include a long instruction line as
-- the section name (e.g. "PART 1 Your health problems List and describe
-- all your medical and mental health problems. If you are getting
-- treatment for the problem, please tell us what kind of treatment.").
-- That overflows 120 chars and bombs insert_chunks with
-- StringDataRightTruncation.
--
-- TEXT and VARCHAR have identical performance in Postgres, so the only
-- thing we lose by widening is the artificial cap.
--
-- Idempotent: ALTER COLUMN TYPE is a no-op if already TEXT.
-- ============================================================================

\timing on

BEGIN;

ALTER TABLE chunks
    ALTER COLUMN section TYPE TEXT;

COMMIT;

-- Verify
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'chunks' AND column_name = 'section';
-- Expected: data_type = 'text', character_maximum_length = NULL
