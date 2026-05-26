"""Load all SSA Blue Book listings into Postgres in one shot.

This is a self-contained loader you can run once after running migrations.
It reads listings_bundle.json (390 KB, 120 listings) and inserts:
  -   120 rows in ssa_listings
  -   813 rows in ssa_listing_criteria  (252 internal + 561 leaf nodes)
  - 2,885 rows in ssa_listing_synonyms

Idempotent: re-running upserts ssa_listings and replaces criteria/synonyms
for each listing (delete-then-insert). Each listing is wrapped in its own
transaction so a partial load never appears in the DB.

USAGE
-----
    # One-shot full load (most common):
    python load_all_listings.py

    # Specific listings only:
    python load_all_listings.py 1.15 12.04 13.29

    # Validate without writing to DB:
    python load_all_listings.py --dry-run

    # Use a different bundle path or database URL:
    python load_all_listings.py --bundle /path/to/listings_bundle.json
    python load_all_listings.py --database-url postgresql://user:pass@host/db

REQUIREMENTS
------------
    pip install sqlalchemy psycopg2-binary pgvector

    The migrations 0001_initial and 0002_listings must have run successfully
    (the loader checks that the tables exist before doing any work).

The loader does NOT compute summary embeddings — that's the job of the
listings_embed worker, which reads rows where summary_embedding IS NULL and
fills them in via the Titan v2 embedder. After this loader completes, every
listing's summary_embedding will be NULL by design.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
#  Validator (inlined so this script has no internal imports)
# ---------------------------------------------------------------------------

VALID_BODY_SYSTEMS = {
    "musculoskeletal", "special_senses", "respiratory", "cardiovascular",
    "digestive", "genitourinary", "hematological", "skin", "endocrine",
    "congenital_multisystem", "neurological", "mental", "cancer", "immune",
}


class ListingValidationError(ValueError):
    """Raised when a listing JSON document violates the schema contract."""


def _validate_node(node: Any, code: str, seen: set[str]) -> None:
    if not isinstance(node, dict):
        raise ListingValidationError(
            f"{code}: tree node is not an object (got {type(node).__name__})"
        )
    if "path" not in node:
        raise ListingValidationError(f"{code}: tree node missing 'path'")
    path = node["path"]
    if path in seen:
        raise ListingValidationError(f"{code}: duplicate path '{path}'")
    seen.add(path)

    if "children" in node:
        if "logic" not in node or node["logic"] not in {"AND", "OR"}:
            raise ListingValidationError(
                f"{code}: internal node {path} has invalid/missing 'logic'"
            )
        if not isinstance(node["children"], list) or not node["children"]:
            raise ListingValidationError(
                f"{code}: internal node {path} has empty/non-list 'children'"
            )
        for child in node["children"]:
            _validate_node(child, code, seen)
    else:
        if "criterion" not in node:
            raise ListingValidationError(
                f"{code}: leaf {path} missing 'criterion'"
            )
        if "keywords" not in node or not isinstance(node["keywords"], list) \
                or not node["keywords"]:
            raise ListingValidationError(
                f"{code}: leaf {path} has empty/non-list 'keywords'"
            )
    if "duration_months_required" in node:
        dur = node["duration_months_required"]
        if not isinstance(dur, int) or dur <= 0:
            raise ListingValidationError(
                f"{code}: node {path} has invalid duration_months_required={dur}"
            )


def validate_listing_doc(doc: Any) -> None:
    """Raise ListingValidationError if doc fails any schema rule."""
    if not isinstance(doc, dict):
        raise ListingValidationError("top-level value is not an object")
    for field in ("code", "title", "body_system", "summary", "rule_json"):
        if field not in doc:
            raise ListingValidationError(f"missing required field '{field}'")
    if doc["body_system"] not in VALID_BODY_SYSTEMS:
        raise ListingValidationError(
            f"{doc['code']}: invalid body_system '{doc['body_system']}'"
        )
    if "synonyms" in doc:
        if not isinstance(doc["synonyms"], dict):
            raise ListingValidationError(
                f"{doc['code']}: 'synonyms' must be an object"
            )
        for canonical, variants in doc["synonyms"].items():
            if not isinstance(variants, list):
                raise ListingValidationError(
                    f"{doc['code']}: synonyms['{canonical}'] is not a list"
                )
    _validate_node(doc["rule_json"], doc["code"], set())


# ---------------------------------------------------------------------------
#  Loader (uses raw psycopg2/SQLAlchemy core — no model classes required)
# ---------------------------------------------------------------------------
#
#  The script doesn't import the SQLAlchemy ORM models so it stays decoupled
#  from the rest of the backend. It uses SQLAlchemy core for the executemany
#  ergonomics and parameter binding but writes plain SQL.

DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    # Default to the EXPEDITE non-prod RDS instance. Username only; password
    # comes from the PGPASSWORD env var (or ~/.pgpass) so it's not in source.
    # Override with --database-url or by setting DATABASE_URL.
    "postgresql+psycopg2://shiq@expedite-nonprod-rds.cfqvorcy6lau.us-east-1.rds.amazonaws.com:5432/expedite?sslmode=require",
)
DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parent / "listings_bundle.json"


def _flatten_tree(
    listing_id: uuid.UUID,
    node: dict,
    parent_id: uuid.UUID | None,
    rows_out: list[dict],
) -> None:
    """Walk the rule_json tree, emitting one criterion row per node.

    Internal nodes get logic + criterion (group label) + is_leaf=False.
    Leaves get criterion + keywords (as JSON) + is_leaf=True.
    Children are emitted after their parent so parent_id references work.
    """
    is_leaf = "children" not in node
    row_id = uuid.uuid4()
    rows_out.append({
        "id": row_id,
        "listing_id": listing_id,
        "parent_id": parent_id,
        "path": node["path"],
        "is_leaf": is_leaf,
        "logic_operator": node.get("logic"),
        "criterion_text": node.get("criterion"),
        "keywords": json.dumps(node.get("keywords")) if is_leaf else None,
        "duration_months_required": node.get("duration_months_required"),
    })
    if not is_leaf:
        for child in node["children"]:
            _flatten_tree(listing_id, child, row_id, rows_out)


def _load_listing(conn, doc: dict) -> tuple[uuid.UUID, int, int]:
    """Load one listing in a single transaction. Returns (id, n_criteria, n_synonyms)."""
    from sqlalchemy import text

    code = doc["code"]

    # Upsert the listing header. summary_embedding is set NULL so the
    # embedding worker will pick it up next pass.
    listing_id_result = conn.execute(text("""
        INSERT INTO ssa_listings
            (id, code, title, body_system, summary, rule_json, version, summary_embedding)
        VALUES
            (:id, :code, :title, :body_system, :summary, CAST(:rule_json AS JSON),
             :version, NULL)
        ON CONFLICT (code) DO UPDATE SET
            title = EXCLUDED.title,
            body_system = EXCLUDED.body_system,
            summary = EXCLUDED.summary,
            rule_json = EXCLUDED.rule_json,
            version = EXCLUDED.version,
            summary_embedding = NULL
        RETURNING id
    """), {
        "id": uuid.uuid4(),
        "code": code,
        "title": doc["title"],
        "body_system": doc["body_system"],
        "summary": doc["summary"],
        "rule_json": json.dumps(doc["rule_json"]),
        "version": str(doc.get("version", "1")),
    })
    listing_id = listing_id_result.scalar_one()

    # Replace criteria for this listing (delete-then-insert)
    conn.execute(
        text("DELETE FROM ssa_listing_criteria WHERE listing_id = :lid"),
        {"lid": listing_id},
    )
    rows: list[dict] = []
    _flatten_tree(listing_id, doc["rule_json"], None, rows)
    if rows:
        conn.execute(text("""
            INSERT INTO ssa_listing_criteria
                (id, listing_id, parent_id, path, is_leaf, logic_operator,
                 criterion_text, keywords, duration_months_required)
            VALUES
                (:id, :listing_id, :parent_id, :path, :is_leaf, :logic_operator,
                 :criterion_text, CAST(:keywords AS JSON),
                 :duration_months_required)
        """), rows)

    # Replace synonyms for this listing (delete-then-insert)
    conn.execute(
        text("DELETE FROM ssa_listing_synonyms WHERE listing_id = :lid"),
        {"lid": listing_id},
    )
    syn_rows = [
        {"id": uuid.uuid4(), "listing_id": listing_id,
         "canonical": canonical, "variant": variant}
        for canonical, variants in doc.get("synonyms", {}).items()
        for variant in variants
    ]
    if syn_rows:
        conn.execute(text("""
            INSERT INTO ssa_listing_synonyms (id, listing_id, canonical, variant)
            VALUES (:id, :listing_id, :canonical, :variant)
        """), syn_rows)

    return listing_id, len(rows), len(syn_rows)


def _check_tables_exist(conn) -> None:
    """Fail early with a useful message if migrations haven't run."""
    from sqlalchemy import text

    required = ("ssa_listings", "ssa_listing_criteria", "ssa_listing_synonyms")
    for table in required:
        exists = conn.execute(
            text("SELECT to_regclass(:t) IS NOT NULL"),
            {"t": f"public.{table}"},
        ).scalar_one()
        if not exists:
            raise SystemExit(
                f"Table '{table}' does not exist. Run migrations first:\n"
                f"    cd backend && alembic upgrade head"
            )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load SSA listings bundle into the database.",
    )
    parser.add_argument(
        "codes", nargs="*",
        help="Specific listing codes to load (e.g. 1.15 12.04). Empty = all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate listings only; do not connect to DB.",
    )
    parser.add_argument(
        "--bundle", type=Path, default=DEFAULT_BUNDLE_PATH,
        help=f"Path to listings_bundle.json (default: {DEFAULT_BUNDLE_PATH})",
    )
    parser.add_argument(
        "--database-url", default=DEFAULT_DATABASE_URL,
        help="SQLAlchemy database URL (default: $DATABASE_URL or local).",
    )
    args = parser.parse_args()

    # ---- Phase 1: load + validate the bundle ----
    if not args.bundle.exists():
        print(f"Bundle not found: {args.bundle}", file=sys.stderr)
        return 1

    with args.bundle.open() as fh:
        bundle = json.load(fh)

    if not isinstance(bundle, dict):
        print(
            "Bundle must be a JSON object keyed by listing code.",
            file=sys.stderr,
        )
        return 1

    filter_codes = set(args.codes) if args.codes else None
    selected: list[dict] = []
    for code, doc in sorted(bundle.items()):
        if filter_codes is not None and code not in filter_codes:
            continue
        try:
            validate_listing_doc(doc)
        except ListingValidationError as e:
            print(f"  [INVALID] {code}: {e}", file=sys.stderr)
            return 2
        selected.append(doc)

    if filter_codes:
        missing = filter_codes - {d["code"] for d in selected}
        if missing:
            print(
                f"Codes not found in bundle: {sorted(missing)}",
                file=sys.stderr,
            )
            return 1

    print(f"Validated {len(selected)} listing(s) from {args.bundle.name}.")
    if args.dry_run:
        print("DRY RUN: skipping DB writes.")
        return 0

    # ---- Phase 2: connect and load ----
    try:
        from sqlalchemy import create_engine
    except ImportError:
        print(
            "SQLAlchemy is required. Install with:\n"
            "    pip install sqlalchemy psycopg2-binary pgvector",
            file=sys.stderr,
        )
        return 1

    engine = create_engine(args.database_url, future=True)

    with engine.connect() as conn:
        _check_tables_exist(conn)

    loaded = 0
    errors = 0
    total_criteria = 0
    total_synonyms = 0

    for doc in selected:
        # Each listing gets its own transaction so a failure on one doesn't
        # roll back the others.
        try:
            with engine.begin() as conn:
                _, n_crit, n_syn = _load_listing(conn, doc)
            loaded += 1
            total_criteria += n_crit
            total_synonyms += n_syn
            print(
                f"  [OK] {doc['code']:<8} "
                f"({n_crit:3d} criteria, {n_syn:3d} synonyms)  "
                f"{doc['title'][:50]}"
            )
        except Exception as e:  # noqa: BLE001 — surface any DB error to the user
            errors += 1
            print(
                f"  [ERROR] {doc['code']}: {e!r}",
                file=sys.stderr,
            )

    print(
        f"\nLoaded {loaded} listing(s); {errors} error(s).\n"
        f"  ssa_listing_criteria rows inserted: {total_criteria}\n"
        f"  ssa_listing_synonyms rows inserted: {total_synonyms}\n"
        f"\nNext step: run the listings_embed worker to populate "
        f"summary_embedding for each listing."
    )
    return 0 if errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
