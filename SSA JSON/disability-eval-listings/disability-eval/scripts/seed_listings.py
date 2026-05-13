"""Seed SSA listings from data/listings/**/*.json into the database.

Usage:
    python scripts/seed_listings.py                  # load all
    python scripts/seed_listings.py 1.15 12.04       # load specific listings
    python scripts/seed_listings.py --dry-run        # parse + validate, no DB writes

Idempotent: re-running on the same file produces the same DB state. Each
listing is wrapped in a transaction; a half-loaded listing never appears in
the DB.

Reload strategy:
  1. UPSERT ssa_listings on (code) — bumps title, body_system, summary,
     rule_json, version. Sets summary_embedding to NULL so the embedding
     worker recomputes it on next pass.
  2. DELETE all ssa_listing_criteria for the listing, then re-insert.
     Delete-then-insert (rather than upsert) because paths can shift between
     versions; upserting would leave orphan rows under stale paths.
  3. DELETE all ssa_listing_synonyms for the listing, then re-insert.

The recursive tree walk is depth-first: parent's row is inserted before
its children so we have a parent_id to pass down.

Run from the repo root:
    cd /path/to/disability-eval && python -m scripts.seed_listings
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Resolve imports whether run as `python -m scripts.seed_listings` or directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from app.db import SessionLocal, engine                       # noqa: E402
from app.models.listing import (                              # noqa: E402
    SsaListing, SsaListingCriterion, SsaListingSynonym,
)
from _listing_validator import (                              # noqa: E402
    ListingValidationError, validate_listing_doc,
)


LISTINGS_DIR = REPO_ROOT / "data" / "listings"


# ---------------------------------------------------------------------------
#  Loader — writes one listing's JSON into the DB tables
# ---------------------------------------------------------------------------

def _walk_and_insert_criteria(
    session: Session,
    listing_id: uuid.UUID,
    node: dict,
    parent_id: uuid.UUID | None,
) -> None:
    """Insert the node's row, then recurse into children.

    Internal nodes have `criterion_text` set to the optional group label (or
    NULL); leaves have it set to the criterion statement. Keywords are stored
    as a JSON array on the criterion row.
    """
    has_children = "children" in node
    is_leaf = not has_children

    row = SsaListingCriterion(
        id=uuid.uuid4(),
        listing_id=listing_id,
        parent_id=parent_id,
        path=node["path"],
        is_leaf=is_leaf,
        logic_operator=node.get("logic"),
        criterion_text=node.get("criterion"),
        keywords=node.get("keywords") if is_leaf else None,
        duration_months_required=node.get("duration_months_required"),
    )
    session.add(row)
    session.flush()  # populates row.id so we can pass it to children

    if has_children:
        for child in node["children"]:
            _walk_and_insert_criteria(session, listing_id, child, row.id)


def load_listing(session: Session, doc: dict, source_path: Path) -> uuid.UUID:
    """Load one listing into the DB. Returns the listing UUID.

    Wrapped by the caller in a transaction. On error, rollback is the
    caller's responsibility.
    """
    code = doc["code"]

    # ---- ssa_listings: upsert by code ----
    existing = session.execute(
        select(SsaListing).where(SsaListing.code == code)
    ).scalar_one_or_none()

    if existing is None:
        listing = SsaListing(
            id=uuid.uuid4(),
            code=code,
            title=doc["title"],
            body_system=doc["body_system"],
            summary=doc["summary"],
            rule_json=doc["rule_json"],
            version=str(doc.get("version", "1")),
            summary_embedding=None,  # to be populated by embedding worker
        )
        session.add(listing)
        session.flush()
    else:
        existing.title = doc["title"]
        existing.body_system = doc["body_system"]
        existing.summary = doc["summary"]
        existing.rule_json = doc["rule_json"]
        existing.version = str(doc.get("version", "1"))
        existing.summary_embedding = None  # invalidate; worker will recompute
        listing = existing

    # ---- ssa_listing_criteria: delete-then-reinsert ----
    session.execute(
        delete(SsaListingCriterion).where(
            SsaListingCriterion.listing_id == listing.id
        )
    )
    session.flush()
    _walk_and_insert_criteria(session, listing.id, doc["rule_json"], parent_id=None)

    # ---- ssa_listing_synonyms: delete-then-reinsert ----
    session.execute(
        delete(SsaListingSynonym).where(
            SsaListingSynonym.listing_id == listing.id
        )
    )
    session.flush()
    for canonical, variants in doc.get("synonyms", {}).items():
        for variant in variants:
            session.add(
                SsaListingSynonym(
                    id=uuid.uuid4(),
                    listing_id=listing.id,
                    canonical=canonical,
                    variant=variant,
                )
            )

    return listing.id


# ---------------------------------------------------------------------------
#  Discovery + CLI
# ---------------------------------------------------------------------------

def discover_listing_files(filter_codes: set[str] | None = None) -> list[Path]:
    """Return all JSON files under data/listings/, optionally filtered by code."""
    files: list[Path] = []
    for path in sorted(LISTINGS_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        if filter_codes is not None:
            try:
                with path.open() as fh:
                    code = json.load(fh).get("code")
            except json.JSONDecodeError:
                continue
            if code not in filter_codes:
                continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed SSA listing JSON files into the database."
    )
    parser.add_argument(
        "codes", nargs="*",
        help="Specific listing codes to load (e.g. 1.15 12.04). Empty = all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse + validate listings only; do not write to DB.",
    )
    args = parser.parse_args()

    filter_codes = set(args.codes) if args.codes else None
    files = discover_listing_files(filter_codes)

    if not files:
        print("No listing files found.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} listing file(s).")

    # Phase 1: parse + validate everything before opening any DB transactions
    parsed: list[tuple[Path, dict]] = []
    failed = 0
    for path in files:
        try:
            with path.open() as fh:
                doc = json.load(fh)
            validate_listing_doc(doc, path)
            parsed.append((path, doc))
        except json.JSONDecodeError as e:
            print(f"  [PARSE FAIL] {path.relative_to(REPO_ROOT)}: {e}",
                  file=sys.stderr)
            failed += 1
        except ListingValidationError as e:
            print(f"  [VALIDATION FAIL] {path.relative_to(REPO_ROOT)}: {e}",
                  file=sys.stderr)
            failed += 1

    if failed:
        print(
            f"\n{failed} file(s) failed validation. Aborting load.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"\nDRY RUN: {len(parsed)} listing(s) parsed and validated. "
              "No DB writes performed.")
        return 0

    # Phase 2: load each listing in its own transaction
    loaded = 0
    errors = 0
    for path, doc in parsed:
        session: Session = SessionLocal()
        try:
            listing_id = load_listing(session, doc, path)
            session.commit()
            loaded += 1
            print(f"  [OK] {doc['code']:<8} {doc['title'][:60]}")
        except IntegrityError as e:
            session.rollback()
            errors += 1
            print(
                f"  [INTEGRITY ERROR] {path.relative_to(REPO_ROOT)}: {e.orig}",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001 — top-level loader, log all
            session.rollback()
            errors += 1
            print(
                f"  [LOAD ERROR] {path.relative_to(REPO_ROOT)}: {e!r}",
                file=sys.stderr,
            )
        finally:
            session.close()

    print(
        f"\nLoaded {loaded} listing(s); {errors} error(s).\n"
        f"Run the listings_embed worker to populate summary_embedding fields."
    )
    return 0 if errors == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
