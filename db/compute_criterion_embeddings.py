"""Compute and store Bedrock Titan v2 embeddings for every LEAF criterion.

Run once after `db/add_criterion_embeddings.sql` adds the criterion_embedding
column. Idempotent — re-run anytime to fill in newly-added leaves or to refresh
after a model swap.

Why only leaves?
    Internal nodes either have NULL criterion_text or just a group label like
    "Two of the following despite continuing treatment" — embedding those is
    mostly noise. Only is_leaf=TRUE rows with non-NULL criterion_text get
    embedded. Internal nodes keep their criterion_embedding NULL by design.

Dependencies + connection: same as db/compute_listing_embeddings.py.

Cost: ~561 leaves × $0.00002/1K input tokens ≈ ~$0.0005 total Bedrock spend.
Runtime: ~3-4 minutes (Titan v2 has no batch API, one invoke_model per text).
"""
from __future__ import annotations

import json
import os
import sys
import time
from urllib.parse import urlparse

import boto3
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector


INFERENCE_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:251862868170:"
    "application-inference-profile/ejscpg2fvj5j"
)
EMBEDDING_DIM = 1024  # Titan v2 — must match criterion_embedding column


# ---------------------------------------------------------------------------
#  Bedrock embedding
# ---------------------------------------------------------------------------

def embed_text(client, text: str) -> np.ndarray:
    body = {
        "inputText": text,
        "dimensions": EMBEDDING_DIM,
        "normalize": True,
    }
    response = client.invoke_model(
        modelId=INFERENCE_PROFILE_ARN,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
#  Postgres connection — delegate to pipeline.db.connect_pg so the
#  db/local_override.py credentials file (gitignored) is honored everywhere.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.db import connect_pg   # noqa: E402


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Setting up Bedrock client (profile=user, region=us-east-1)...")
    session = boto3.Session(profile_name="user", region_name="us-east-1")
    bedrock = session.client("bedrock-runtime")

    # connect_pg() (from pipeline.db) already calls register_vector internally
    print("Connecting to Postgres...")
    conn = connect_pg()
    print(f"  connected: {conn.get_dsn_parameters().get('host')} / "
          f"{conn.get_dsn_parameters().get('dbname')}")

    cur = conn.cursor()

    # Verify column type
    cur.execute("""
        SELECT format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'ssa_listing_criteria'::regclass
          AND attname = 'criterion_embedding'
    """)
    row = cur.fetchone()
    if not row:
        print(
            "ERROR: ssa_listing_criteria.criterion_embedding column not found.\n"
            "Run db/add_criterion_embeddings.sql first.",
            file=sys.stderr,
        )
        return 2
    col_type = row[0]
    print(f"  criterion_embedding column type: {col_type}")
    if col_type != f"vector({EMBEDDING_DIM})":
        print(
            f"\nERROR: column type is {col_type!r} but Titan v2 produces "
            f"vector({EMBEDDING_DIM}).",
            file=sys.stderr,
        )
        return 3

    # Smoke test
    print("Smoke-testing Bedrock connectivity...")
    try:
        test_vec = embed_text(bedrock, "test")
    except Exception as e:
        print(f"\nERROR: Bedrock call failed — {type(e).__name__}: {e}",
              file=sys.stderr)
        return 4
    print(f"  smoke test OK: got {len(test_vec)}-dim vector, L2 norm "
          f"{float(np.linalg.norm(test_vec)):.4f}")

    # Fetch leaves that need embeddings
    cur.execute("""
        SELECT c.id, l.code, c.path, c.criterion_text
        FROM ssa_listing_criteria c
        JOIN ssa_listings l ON l.id = c.listing_id
        WHERE c.is_leaf = TRUE
          AND c.criterion_text IS NOT NULL
          AND c.criterion_embedding IS NULL
        ORDER BY l.code, c.path
    """)
    rows = cur.fetchall()
    print(f"Leaf criteria needing embeddings: {len(rows)}")
    if not rows:
        print("All leaves already have embeddings. Done.")
        return 0

    # Embed each
    print("Embedding via Bedrock Titan v2...")
    start = time.time()
    success = 0
    failed: list[tuple[str, str, str]] = []
    for i, (crit_id, code, path, criterion_text) in enumerate(rows, 1):
        try:
            vec = embed_text(bedrock, criterion_text)
        except Exception as e:
            failed.append((code, path, f"{type(e).__name__}: {e}"))
            print(
                f"  [{i:>4}/{len(rows)}] {code}:{path:<20} FAILED — {e}",
                file=sys.stderr,
            )
            continue

        cur.execute(
            "UPDATE ssa_listing_criteria "
            "SET criterion_embedding = %s "
            "WHERE id = %s",
            (vec, crit_id),
        )
        success += 1
        if i % 20 == 0 or i == len(rows):
            conn.commit()  # checkpoint every 20
            elapsed = time.time() - start
            print(
                f"  [{i:>4}/{len(rows)}] {code}:{path:<20} ✓  "
                f"({elapsed:.0f}s elapsed, {i/elapsed:.1f} req/s)"
            )
    conn.commit()

    # Final summary
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE is_leaf = TRUE AND criterion_text IS NOT NULL) AS embedable_leaves,
            COUNT(criterion_embedding)                                            AS embedded
        FROM ssa_listing_criteria
    """)
    total, embedded = cur.fetchone()
    print(f"\nFinal: {embedded}/{total} leaf criteria have embeddings.")
    if failed:
        print(f"\n{len(failed)} leaves failed:")
        for code, path, err in failed[:5]:
            print(f"  {code}:{path}  — {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        print("Re-run the script to retry failed leaves.")

    cur.close()
    conn.close()
    return 0 if not failed else 5


if __name__ == "__main__":
    raise SystemExit(main())
