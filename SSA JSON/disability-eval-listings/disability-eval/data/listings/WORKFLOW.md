# Listings — How to load them into the database

## What's here

The `data/listings/<body_system>/<code>.json` files are the source of truth
for SSA listing definitions. There are 107 listings covering all 14 adult
body systems.

Three documents define the data:

- `README.md` — JSON schema for an individual listing file (you're not
  reading that file right now; see [README.md](README.md))
- `SCHEMA.md` — DB schema rationale: why four tables, what lives where,
  idempotency contract
- `WORKFLOW.md` — this file

## End-to-end workflow

### 1. Validate the JSON files (no DB required)
```bash
python scripts/validate_listings.py
```
Catches malformed files, duplicate paths, missing fields, invalid body
systems. Good as a pre-commit check or CI gate. Exits non-zero on any
violation.

### 2. Run the migration
```bash
cd backend && alembic upgrade head
```
The 0002_listings migration adds the `ssa_listing_synonyms` table and the
`summary_embedding` pgvector column on `ssa_listings`. The two tables that
already existed from 0001 (`ssa_listings`, `ssa_listing_criteria`) keep their
schema unchanged.

### 3. Seed the listings
```bash
python scripts/seed_listings.py            # all 107 listings
python scripts/seed_listings.py 1.15 12.04 # specific listings
python scripts/seed_listings.py --dry-run  # parse + validate, no DB writes
```
The seed loader:
- Validates every file before opening any DB transactions (phase 1)
- Loads each listing in its own transaction (phase 2) — a partial load
  never appears in the DB
- Is idempotent: re-running on the same file produces the same DB state
- Sets `summary_embedding` to NULL on every reload so the embedding worker
  recomputes it next pass

### 4. Generate embeddings
```bash
python -m app.workers.listings_embed   # to be added in next iteration
```
Reads every `ssa_listings` row with NULL `summary_embedding`, runs the
summary text through the Titan v2 1024-dim embedder, and writes the vector
back. The triage call (`identify_candidate_listings`) uses these embeddings
to decide which listings are worth analyzing for a given case.

## What you get in the DB

After seed completes (numbers from current data):

| Table                   | Rows  |
|-------------------------|------:|
| ssa_listings            |   107 |
| ssa_listing_criteria    |   747 |
| ssa_listing_synonyms    | 2,679 |

The keywords for each leaf are stored on `ssa_listing_criteria.keywords`
as a JSON array (not in a separate table) — we only ever read keywords
per-leaf, never query "all listings with keyword X."

## Reloading after edits

Edit a JSON file, then:
```bash
python scripts/validate_listings.py data/listings/musculoskeletal/1.15.json
python scripts/seed_listings.py 1.15
```
The loader deletes existing criteria + synonyms for that listing and
re-inserts. Other listings are untouched. The summary embedding for the
edited listing is invalidated; the worker will pick it up on next pass.

## Adding a new listing

1. Create `data/listings/<body_system>/<code>.json` following the schema
   in `README.md`
2. Validate: `python scripts/validate_listings.py path/to/new.json`
3. Seed: `python scripts/seed_listings.py <code>`

No migration needed — schemas are stable.
