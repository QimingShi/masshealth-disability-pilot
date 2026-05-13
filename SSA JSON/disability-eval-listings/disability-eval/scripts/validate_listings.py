"""Validate SSA listing JSON files against the schema contract.

Usage:
    python scripts/validate_listings.py
    python scripts/validate_listings.py data/listings/musculoskeletal/1.15.json

Exit codes:
    0  all files valid
    1  no files found
    2  one or more validation failures

This script does not touch the database — it only checks structure. Run it as
a pre-commit hook or in CI before letting seed_listings.py near production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _listing_validator import (  # noqa: E402
    ListingValidationError,
    validate_listing_doc,
)


LISTINGS_DIR = REPO_ROOT / "data" / "listings"


def main() -> int:
    if len(sys.argv) > 1:
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        files = sorted(LISTINGS_DIR.rglob("*.json"))

    if not files:
        print("No listing files found.", file=sys.stderr)
        return 1

    failed = 0
    for path in files:
        try:
            with path.open() as fh:
                doc = json.load(fh)
            validate_listing_doc(doc, path)
            print(f"  [OK] {path.relative_to(REPO_ROOT)}")
        except json.JSONDecodeError as e:
            print(
                f"  [PARSE FAIL] {path.relative_to(REPO_ROOT)}: {e}",
                file=sys.stderr,
            )
            failed += 1
        except ListingValidationError as e:
            print(
                f"  [INVALID] {path.relative_to(REPO_ROOT)}: {e}",
                file=sys.stderr,
            )
            failed += 1

    print(f"\n{len(files) - failed} valid, {failed} invalid.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
