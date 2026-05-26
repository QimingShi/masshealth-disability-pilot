"""Compute and store Bedrock Titan v2 embeddings for every SSA listing.

Run once after `db/switch_to_titan_embeddings.sql` upgrades the column to
vector(1024). Idempotent: re-run anytime to fill in newly-added listings or
to refresh after a model swap.

Dependencies:
    pip install pgvector psycopg2-binary boto3 numpy

Connections:
    DATABASE_URL  — Postgres URL (same one load_all_listings.py uses)
    AWS profile   — Bedrock client uses the SSO-logged-in `user` profile

Embedding model:
    Bedrock Titan Text Embeddings v2 via application inference profile
    arn:aws:bedrock:us-east-1:251862868170:application-inference-profile/ejscpg2fvj5j

    Output dim: 1024 (must match summary_embedding column).
    Cost: ~$0.00002 per 1K input tokens → ~$0.0001 for the 120 listings.
    Throughput: ~1 invocation/second sequentially (no batching API in Titan v2).
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
EMBEDDING_DIM = 1024  # Titan v2 — must match summary_embedding column dimension


# ---------------------------------------------------------------------------
#  Bedrock embedding
# ---------------------------------------------------------------------------

def embed_text(client, text: str) -> np.ndarray:
    """One Titan v2 invocation. Returns a unit-normalized float32 vector."""
    body = {
        "inputText": text,
        "dimensions": EMBEDDING_DIM,
        "normalize": True,    # L2-normalize so dot product == cosine similarity
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
#  Postgres connection
# ---------------------------------------------------------------------------

def connect_pg():
    """Connect to Postgres using DATABASE_URL or PG* env vars."""
    url = os.environ.get("DATABASE_URL")
    if url:
        url = url.replace("postgresql+psycopg2://", "postgresql://")
        parsed = urlparse(url)
        kwargs = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "user": parsed.username,
            "password": parsed.password or os.environ.get("PGPASSWORD"),
            "dbname": parsed.path.lstrip("/"),
        }
        if parsed.query:
            for kv in parsed.query.split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k in {"sslmode", "sslrootcert"}:
                        kwargs[k] = v
    else:
        kwargs = {}   # rely on PG* env vars
    return psycopg2.connect(**kwargs)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    # ---- Set up Bedrock client ----
    print(f"Setting up Bedrock client (profile=user, region=us-east-1)...")
    session = boto3.Session(profile_name="user", region_name="us-east-1")
    bedrock = session.client("bedrock-runtime")

    # ---- Connect to Postgres ----
    print("Connecting to Postgres...")
    conn = connect_pg()
    register_vector(conn)
    print(f"  connected: {conn.get_dsn_parameters().get('host')} / "
          f"{conn.get_dsn_parameters().get('dbname')}")

    cur = conn.cursor()

    # ---- Verify column type matches expected dimension ----
    cur.execute("""
        SELECT format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'ssa_listings'::regclass
          AND attname = 'summary_embedding'
    """)
    row = cur.fetchone()
    if not row:
        print("ERROR: ssa_listings.summary_embedding column not found.",
              file=sys.stderr)
        return 2
    col_type = row[0]
    print(f"  summary_embedding column type: {col_type}")
    if col_type != f"vector({EMBEDDING_DIM})":
        print(
            f"\nERROR: column type is {col_type!r} but Titan v2 produces "
            f"vector({EMBEDDING_DIM}).\n"
            f"Run db/switch_to_titan_embeddings.sql first.",
            file=sys.stderr,
        )
        return 3

    # ---- Smoke-test Bedrock with one call before bulk work ----
    print("Smoke-testing Bedrock connectivity...")
    try:
        test_vec = embed_text(bedrock, "test")
    except Exception as e:
        print(f"\nERROR: Bedrock call failed — {type(e).__name__}: {e}",
              file=sys.stderr)
        return 4
    print(f"  smoke test OK: got {len(test_vec)}-dim vector, L2 norm "
          f"{float(np.linalg.norm(test_vec)):.4f}")

    # ---- Fetch listings needing embeddings ----
    cur.execute("""
        SELECT id, code, summary
        FROM ssa_listings
        WHERE summary_embedding IS NULL
        ORDER BY code
    """)
    rows = cur.fetchall()
    print(f"Listings needing embeddings: {len(rows)}")
    if not rows:
        print("All listings already have embeddings. Done.")
        return 0

    # ---- Embed each listing (Titan has no batch API; one call per text) ----
    print("Embedding via Bedrock Titan v2...")
    start = time.time()
    success = 0
    failed: list[tuple[str, str]] = []
    for i, (listing_id, code, summary) in enumerate(rows, 1):
        try:
            vec = embed_text(bedrock, summary)
        except Exception as e:
            failed.append((code, f"{type(e).__name__}: {e}"))
            print(f"  [{i:>3}/{len(rows)}] {code:<8} FAILED — {e}",
                  file=sys.stderr)
            continue

        cur.execute(
            "UPDATE ssa_listings SET summary_embedding = %s, updated_at = NOW() "
            "WHERE id = %s",
            (vec, listing_id),
        )
        success += 1
        if i % 10 == 0 or i == len(rows):
            conn.commit()  # checkpoint every 10
            elapsed = time.time() - start
            print(f"  [{i:>3}/{len(rows)}] {code:<8} ✓  "
                  f"({elapsed:.0f}s elapsed, "
                  f"{i/elapsed:.1f} req/s)")
    conn.commit()

    # ---- Final summary ----
    cur.execute(
        "SELECT COUNT(*) AS total, COUNT(summary_embedding) AS embedded "
        "FROM ssa_listings"
    )
    total, embedded = cur.fetchone()
    print(f"\nFinal: {embedded}/{total} listings have embeddings.")
    if failed:
        print(f"\n{len(failed)} listing(s) failed:")
        for code, err in failed[:5]:
            print(f"  {code}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        print("Re-run the script to retry failed listings.")

    cur.close()
    conn.close()
    return 0 if not failed else 5


if __name__ == "__main__":
    raise SystemExit(main())
