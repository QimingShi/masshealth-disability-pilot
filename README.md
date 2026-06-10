# MassHealth Disability Reviewer Assistant

AI decision-support tool for MassHealth disability reviewers. Given a member's
medical record packet, it matches the chart against SSA Blue Book disability
listings and produces an evidence-cited reviewer-facing form for each candidate
listing.

The reviewer remains the decision-maker. The AI surfaces candidate listings,
classifies each criterion as **met / not met / insufficient** with verbatim
chart citations, and ranks confidence — the form is a starting point for
reviewer evaluation, not a verdict.

Runs inside the UMass Chan Disability Evaluation Services (DES) cloud
environment: AWS Textract for OCR, Bedrock for embeddings (Titan v2) and LLM
(Claude Opus 4.6), and RDS Postgres + pgvector for retrieval.

## What it does, end-to-end

1. Lists every PDF in an S3 case folder
2. Classifies each PDF by filename (allegation supplement / medical evidence / skip)
3. Runs AWS Textract (LAYOUT + TABLES + FORMS) on the kept PDFs
4. Detects sub-documents and chunks the OCR output by section
5. Computes Bedrock Titan Text Embeddings v2 (1024-dim) for every chunk + allegation
6. Writes everything to Postgres (cases, source_pdfs, documents, chunks, allegations)
7. Identifies candidate SSA listings via pgvector cosine similarity (allegation → listing summary)
8. For each candidate, runs per-leaf retrieval (criterion → chunk cosine) and asks Claude Opus 4.6 via Bedrock to classify each criterion with verbatim chart citations
9. Consolidates the AND/OR criterion tree using 3-valued logic (met / not_met / insufficient)
10. Renders a populated reviewer form per candidate listing, plus a case-level summary HTML with a sticky left-sidebar nav grouping listings by verdict
11. Annotates the source PDF with yellow highlights over every cited chunk

Per-case cost on a typical 40-page packet: ~$5-7 (Textract is the dominant
cost; Bedrock embeddings + LLM eval together are ~$1-3).

## The one command

```cmd
:: 1. Auth + env (once per shell)
aws sso login --profile user
set DATABASE_URL=postgresql://USER@YOUR-RDS-HOST:5432/DB
set PGPASSWORD=<your password>

:: 2. Full pipeline for a case folder
py db\ingest_s3_to_db.py --folder "your-incoming-bucket/1234567 Psych- 100/"
```

The script auto-detects `case_id=1234567` from the 7-digit folder prefix and
runs all 10 stages above. Output lands in:

- `data/1234567/chunks.json` — lite chunks shape
- `_phi/1234567/` — source PDFs, Textract raw, bbox sidecar, annotated PDF (PHI)
- `output/1234567/1234567_EXPEDITESummary.html` ← **open this first**
- `output/1234567/{1_MEETS,2_INSUFFICIENT,3_DOES_NOT_MEET}_<code>.html` — one per candidate listing

### Useful flags

| Flag | When |
|---|---|
| `--dry-run` | Preview the file-classification plan without spending money |
| `--no-match` | Run only Stages 1+2 (load to DB). Use for bulk historical loads. |
| `--medical-pattern "(?i)\b(AI\|MR)\b"` | When a folder uses "MR" prefix instead of "AI" for medical records |

### Re-run matcher only (no re-OCR)

If the case is already in Postgres and you just want to re-evaluate (e.g.
after fixing a listing):

```cmd
py run.py --from-db 1234567
```

Costs ~$1-3 (just Bedrock LLM eval).

## Three-layer audit trail

Every matcher run persists a complete audit trail to Postgres:

| Layer | Tables | What it captures |
|---|---|---|
| **1. Retrieval** | `chunk_leaf_matches` | "For leaf X, these chunks scored Y by cosine" |
| **2. Criterion assessment** | `leaf_assessments`, `leaf_assessment_evidence` | "AI read those chunks, decided met/not_met/insufficient + evidence_strength, citing chunk Z with quote Q" |
| **3. Summary** | `listing_assessments`, `case_summaries` | "AND/OR rollup to Meets / Does not meet / Insufficient, plus a 0..1 groundedness score" |

The audit trail is idempotent per case — re-running the matcher does
`DELETE WHERE case_id` then `INSERT`, so the rows always reflect the
most recent matcher run.

### Groundedness formula

The composite 0..1 score in `listing_assessments`:

```
groundedness = 0.5 * frac_leaves_decided           // (n_met + n_not_met) / n_leaves
             + 0.3 * avg_top_similarity            // mean rank-1 retrieval similarity
                                                    // over LOAD-BEARING leaves only
             + 0.2 * (avg_evidence_per_leaf / 2)   // citation density, capped at 2
```

"Load-bearing" leaves are the ones whose verdict drove the consolidated root
decision — for OR-met cases, only the satisfying branch counts; for AND-not_met,
only the blocking leaves count. See `pipeline/consolidate.py::load_bearing_leaf_paths`.

### Useful queries

```sql
-- Reviewer triage: high-confidence MEETS cases
SELECT c.case_id, l.code, la.groundedness_score, la.decision_summary
FROM listing_assessments la
JOIN cases c          ON c.id = la.case_id
JOIN ssa_listings l   ON l.id = la.listing_id
WHERE la.form_verdict = 'Meets' AND la.groundedness_score >= 0.7
ORDER BY la.groundedness_score DESC;

-- Drill into a specific case
SELECT le.leaf_path, le.verdict, le.evidence_strength, le.rationale
FROM leaf_assessments le
JOIN cases c        ON c.id = le.case_id
JOIN ssa_listings l ON l.id = le.listing_id
WHERE c.case_id = '1234567' AND l.code = '12.02'
ORDER BY le.leaf_path;
```

## Project layout

```
.
├── run.py                              Main matcher entry (run on a case_id already in Postgres)
│
├── pipeline/                           Core matcher modules
│   ├── chunks.py                       Loaders + ICD extraction + body-system map
│   ├── evaluate.py                     Bedrock Claude per-leaf eval + citation guardrails
│   ├── consolidate.py                  AND/OR tree walk + load-bearing-leaf identification
│   ├── output.py                       Renders markdown + HTML forms + case-summary HTML
│   ├── annotate_pdf.py                 Yellow-highlight annotation on the source PDF
│   ├── ingest_textract.py              Textract response → layout-aware chunks
│   ├── ingest_real.py                  Multi-PDF S3 ingest orchestrator
│   ├── db.py                           Postgres DAL (case-side writes + read functions)
│   ├── db_matcher.py                   SQL pgvector candidates + retrieval + persistence
│   └── groundedness.py                 Score components + headline-outcome picker
│
├── db/                                 DB schemas + workers + entry points
│   ├── schema_v1.sql                   SSA listings tables (listings, criteria, synonyms)
│   ├── add_criterion_embeddings.sql    Criterion embedding column + ivfflat index
│   ├── case_schema.sql                 Case-side tables (cases, documents, chunks, allegations)
│   ├── add_chunk_leaf_matches_table.sql  Layer 1: retrieval cache
│   ├── add_assessment_tables.sql       Layers 2+3: per-leaf + per-listing + case audit trail
│   ├── add_n_load_bearing_column.sql   Add n_load_bearing column to listing_assessments
│   ├── verify_embeddings.sql           Diagnostic queries for listing/criterion embeddings
│   ├── verify_case_schema.sql          Diagnostic queries for case-side tables
│   ├── compute_listing_embeddings.py   Bedrock Titan v2 worker — embed listing summaries
│   ├── compute_criterion_embeddings.py Bedrock Titan v2 worker — embed leaf criteria
│   ├── load_case_to_db.py              Load chunks.json + chunks_with_bbox.json → Postgres
│   ├── ingest_s3_to_db.py              ⭐ End-to-end entry: S3 folder → Postgres + matcher
│   └── archive/                        Historical one-time migrations (already applied)
│
├── SSA JSON/                           SSA Blue Book listings + loader
│   ├── README.md                       SSA data overview
│   ├── load_all_listings.py            Loads listings_bundle.json into Postgres
│   ├── listings_bundle.json            All 120 listings (manually maintained from SSA 10-6-23 doc)
│   └── disability-eval-listings/       Per-listing JSON files (in sync with the bundle)
│
├── data/                               Case JSON artifacts (chunks.json per case)
│   └── chunks.json                     Synthetic baseline for offline testing
│
├── output/                             Per-case reviewer forms (gitignored — PHI quotes)
├── _phi/                               PHI source artifacts (gitignored)
├── requirements.txt
└── README.md
```

## Architecture decisions

**Bedrock, not direct Anthropic API.** All LLM calls go through Amazon Bedrock
(`pipeline/evaluate.py`) using an application inference profile bound to
Claude Opus 4.6. SSO via the `user` AWS profile. This keeps PHI inside the
account boundary and uses the institutional AWS billing.

**pgvector for retrieval.** Both candidate identification (allegation →
listing summary) and per-leaf retrieval (criterion → chunk) use Postgres
pgvector cosine similarity. 1024-dim Titan Text Embeddings v2 across all
three corpora (listings, criteria, chunks) so they live in the same space.

**Server-side citation guardrails.** Every chunk_id Claude cites must be in
the input set; every quote must be a whitespace-normalized verbatim substring
of that chunk. Hallucinated citations cause the leaf to be downgraded to
`insufficient` rather than silently included. See `pipeline/evaluate.py`.

**Three-valued verdicts.** Leaves and internal nodes carry `met`, `not_met`,
or `insufficient`. The matcher's consolidation never collapses `insufficient`
into `not_met` — the reviewer must see what the chart couldn't speak to.

**Idempotent end-to-end.** Re-running ingest on the same case_id does
`DELETE` + `INSERT` on every case-scoped table (chunks, allegations,
chunk_leaf_matches, leaf_assessments, listing_assessments, case_summaries).
No accumulating cruft from prior runs.

**Adaptive retry on Bedrock throttling.** `pipeline/evaluate.py` wraps
`invoke_model` with a two-layer retry: boto3 client adaptive mode
(max_attempts=10) + an outer 5-attempt loop with exponential backoff
(4s → 64s). Handles the back-to-back-call pattern of the matcher
(~35-50 calls per case) without bouncing on transient
`ServiceUnavailableException`s.

## Per-case output: what reviewers see

The case-summary HTML (`output/<case_id>/<case_id>_EXPEDITESummary.html`) is the
reviewer's single-page entry point:

- **Headline**: N candidates evaluated, breakdown of Meets / Does not meet /
  Insufficient counts, overall groundedness
- **Sticky left-sidebar nav**: every candidate listing, grouped by verdict
  priority (Meets → Does not meet → Insufficient), with verdict-colored left
  border and groundedness score per row. Click to jump.
- **3-column summary table** ("Possible Visualization of Output Report"):
  one row per allegation-mapped listing, showing the allegation, the best
  cited chart evidence (provider / date / page with clickable PDF link), and
  the matched SSI listing with verdict pill
- **Detailed listing sections below**: per-candidate forms with the full
  AND/OR criterion tree, per-leaf verdicts + rationale + verbatim citations.
  Each section has an anchor and a "↑ Back to summary table" link.

Citations are clickable links to `file:///.../_phi/<case>/source_annotated.pdf#page=N`
when the local source PDF exists; otherwise they render as plain page-number
text. The annotated PDF is generated at the end of the matcher run.

## Known limitations

- Some 8.x and 11.x listings haven't been deep-audited against the 10-6-23
  SSA doc (the doc only contains 2023 revisions, not the full Blue Book).
  The 6.xx and 12.xx ranges plus the 4 listings we caught in cancer/immune
  audits (13.13, 13.23, 13.29, 14.04) are confirmed correct.
- File-classification uses simple regex on filenames (`AI`, `MR`, `Supplement`).
  Override per-run via `--medical-pattern`. A future content-based classifier
  (read page-1 text + Bedrock prompt) is sketched but not built.
- Annotated PDF generation requires `chunks_with_bbox.json` from the Textract
  ingest path; legacy hand-transcribed cases get page-level (not bbox-level)
  citations.
- HTML citations use *relative* URLs to a copy of the source PDF placed
  alongside the HTML in `output/<case_id>/`. The case bundle is therefore
  self-contained: zip or copy the folder and every citation still resolves
  in a local browser (Edge/Chrome/Firefox). SharePoint's browser-side HTML
  preview may still strip the `#page=N` fragment when opened through the
  web viewer — for SharePoint hand-off, expect reviewers to download the
  folder and open `<case_id>_EXPEDITESummary.html` locally.

## Setup notes (first-time)

1. AWS account with Textract + Bedrock access (`AWSExpediteUsers` role)
2. RDS Postgres instance with pgvector extension enabled
3. Bedrock application inference profile for Claude Opus 4.6
4. SSO profile `user` configured: `aws configure sso --profile user`
5. `pip install -r requirements.txt`
6. Bring up the DB schema (one-time):

```sql
\i db/schema_v1.sql
\i db/add_criterion_embeddings.sql
\i db/case_schema.sql
\i db/add_chunk_leaf_matches_table.sql
\i db/add_assessment_tables.sql
\i db/add_n_load_bearing_column.sql
```

7. Load SSA listings + compute their embeddings (one-time):

```cmd
py "SSA JSON\load_all_listings.py"
py db\compute_listing_embeddings.py
py db\compute_criterion_embeddings.py
```

8. You're ready to run case ingests via `py db\ingest_s3_to_db.py --folder ...`

## License

Production code for UMass Chan Medical School Disability Evaluation Services. SSA
listings under `SSA JSON/` are derived from public regulation data and remain
in the public domain.
