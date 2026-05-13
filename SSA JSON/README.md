# SSA Listings — Database Load Package

Everything you need to load all 120 adult SSA Blue Book listings into Postgres.

## TL;DR

```bash
# 1. Run the migrations (adds the synonyms table + embedding column)
cd backend && alembic upgrade head

# 2. Load all 120 listings in one shot
cd ..
python load_all_listings.py
```

After this completes:

| Table                  | Rows  |
|------------------------|------:|
| ssa_listings           |   120 |
| ssa_listing_criteria   |   813 |
| ssa_listing_synonyms   | 2,885 |

## What's in the package

```
disability-eval/
├── load_all_listings.py             ← Run this. One-shot loader.
├── data/
│   ├── listings_bundle.json         ← All 120 listings in a single JSON file
│   └── listings/
│       ├── README.md                ← Per-listing JSON schema
│       ├── SCHEMA.md                ← DB schema rationale
│       ├── WORKFLOW.md              ← End-to-end workflow
│       ├── musculoskeletal/         ← Individual JSON files (one per listing)
│       ├── mental/                  ← Edit these, re-bundle, re-load
│       └── ... (12 more body systems)
├── scripts/
│   ├── seed_listings.py             ← Alt loader: reads individual files
│   ├── validate_listings.py         ← CI/pre-commit check (no DB)
│   └── _listing_validator.py        ← Pure-Python validator
└── backend/
    ├── alembic/versions/
    │   └── 0002_listings.py         ← Adds synonyms table + embedding column
    └── app/models/
        └── listing.py               ← SQLAlchemy ORM models
```

## How `load_all_listings.py` works

The loader is a single self-contained Python file (no internal imports — it
inlines the validator). It reads `data/listings_bundle.json`, validates every
listing, then loads each one in its own transaction.

It's idempotent: running it twice produces the same DB state. For each listing:
- `ssa_listings` is upserted on `code` (UPDATE on conflict)
- `ssa_listing_criteria` for that listing is deleted, then re-inserted
- `ssa_listing_synonyms` for that listing is deleted, then re-inserted

The `summary_embedding` column is set NULL on every reload so the embedding
worker recomputes it next pass.

### Selective loads

```bash
# Just musculoskeletal 1.15 + mental 12.04
python load_all_listings.py 1.15 12.04

# Validate without writing to DB (CI/test)
python load_all_listings.py --dry-run

# Custom database URL
python load_all_listings.py --database-url postgresql://user:pw@host:5432/mydb
```

The loader also reads the `DATABASE_URL` env var if `--database-url` isn't
passed.

## Editing listings

The bundle JSON is built from the per-listing files under `data/listings/`.
If you edit a per-listing file:

```bash
# Re-validate the file you edited
python scripts/validate_listings.py data/listings/musculoskeletal/1.15.json

# Re-bundle (one Python one-liner — see below)
python -c "import json; from pathlib import Path; \
  d = {json.load(open(p))['code']: json.load(open(p)) \
       for p in sorted(Path('data/listings').rglob('*.json'))}; \
  json.dump(d, open('data/listings_bundle.json', 'w'), indent=2)"

# Re-load just that listing
python load_all_listings.py 1.15
```

Or skip the bundle entirely and use `scripts/seed_listings.py`, which reads
the individual JSON files directly.

## Coverage

120 listings across all 14 adult body systems:

| Body system            | Listings |
|------------------------|---------:|
| musculoskeletal        |        9 |
| special_senses         |        7 |
| respiratory            |        7 |
| cardiovascular         |        8 |
| digestive              |        6 |
| genitourinary          |        5 |
| hematological          |        5 |
| skin                   |        7 |
| endocrine              |        1 (narrative index) |
| congenital_multisystem |        1 |
| neurological           |       16 |
| mental                 |       11 |
| cancer                 |       28 |
| immune                 |        9 |

## Requirements

```bash
pip install sqlalchemy psycopg2-binary pgvector
```

The Postgres extensions `vector` and `pg_trgm` must be installed (handled by
the migrations).
