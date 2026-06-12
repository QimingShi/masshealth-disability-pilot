"""Compute and store Bedrock Titan v2 embeddings for every LEAF criterion.

Run once after `db/add_criterion_embeddings.sql` adds the criterion_embedding
column. Idempotent — re-run anytime to fill in newly-added leaves or to refresh
after a model swap.

Why only leaves?
    Internal nodes either have NULL criterion_text or just a group label like
    "Two of the following despite continuing treatment" — embedding those is
    mostly noise. Only is_leaf=TRUE rows with non-NULL criterion_text get
    embedded. Internal nodes keep their criterion_embedding NULL by design.

Embedding source (enriched):
    parent code + title + body_system
    leaf criterion_text
    leaf-level keywords (the JSON array on each rule node)

    Embedding the bare criterion_text alone produced retrieval misses on
    short clinical fragments — e.g. an allegation containing "Hodgkin
    lymphoma" matched 13.05 Lymphoma's paragraph C ("Hodgkin lymphoma
    with failure to achieve...") only weakly because the embedding had
    no anchor that this fragment lives under "13.05 Lymphoma (cancer)".
    Enriching with the parent listing's code+title+body_system gives
    the leaf a stable medical-context prefix; including the leaf
    keywords brings everyday synonyms (e.g. "ABVD", "BEACOPP") into
    the vector mass even when the criterion prose is terse.

    Mirrors the listing-side enrichment in db/compute_listing_embeddings.py
    so both sides of the matcher land in the same neighborhood.

CLI:
    py db\\compute_criterion_embeddings.py            # embed only NULL rows
    py db\\compute_criterion_embeddings.py --force    # re-embed every leaf
                                                       # (use after enrichment changes)

Dependencies + connection: same as db/compute_listing_embeddings.py.

Cost: ~561 leaves × $0.00002/1K input tokens ≈ ~$0.0005 total Bedrock spend.
Runtime: ~3-4 minutes (Titan v2 has no batch API, one invoke_model per text).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.parse import urlparse

import boto3
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector


INFERENCE_PROFILE_ARN = os.environ.get(
    "TITAN_INFERENCE_PROFILE_ARN",
    # Set TITAN_INFERENCE_PROFILE_ARN to your Bedrock application-inference-profile ARN.
    # Example shape: arn:aws:bedrock:<region>:<account>:application-inference-profile/<id>
    "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/REPLACE_ME",
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
#  Postgres connection
# ---------------------------------------------------------------------------

def connect_pg():
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
        kwargs = {}
    return psycopg2.connect(**kwargs)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def _build_embed_text(
    code: str,
    title: str,
    body_system: str | None,
    criterion_text: str,
    keywords: list[str] | None,
) -> str:
    """Build the enriched embedding input for a single leaf criterion.

    Format (mirrors the listing-side enrichment so both halves of the
    matcher's cosine math live in the same vocabulary):

        <code> <title>
        Body system: <body_system>

        <criterion_text>

        Common terms: <kw1>, <kw2>, ...

    The code+title prefix anchors retrieval against allegations that
    name the diagnosis explicitly ("Hodgkin lymphoma", "depression")
    even when the criterion prose is terse or jargon-heavy. The
    keyword tail brings everyday synonyms into vector mass without
    diluting the criterion's specificity.
    """
    parts = [f"{code} {title}".strip()]
    if body_system:
        parts.append(f"Body system: {body_system}")
    parts.append(criterion_text)
    if keywords:
        # Dedupe (case-insensitive) preserving first-seen order, and
        # cap at 50 to keep Titan's 8K-token input ceiling comfortable
        # on leaves that carry many keyword variants.
        uniq = []
        seen = set()
        for k in keywords:
            if not isinstance(k, str):
                continue
            kl = k.strip().lower()
            if kl and kl not in seen:
                seen.add(kl)
                uniq.append(k.strip())
        if uniq:
            parts.append("Common terms: " + ", ".join(uniq[:50]))
    return "\n\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute Titan v2 embeddings for SSA leaf criteria."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all leaves (default: only rows with NULL embedding). "
             "Use after changing the embedding source text.",
    )
    args = p.parse_args()

    print("Setting up Bedrock client (profile=user, region=us-east-1)...")
    session = boto3.Session(profile_name="user", region_name="us-east-1")
    bedrock = session.client("bedrock-runtime")

    print("Connecting to Postgres...")
    conn = connect_pg()
    register_vector(conn)
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

    # Fetch leaves that need embeddings.
    # JOIN on ssa_listings to pull the parent's code+title+body_system, which
    # we splice into the embedding input alongside the leaf's keywords array.
    embed_filter = "" if args.force else "AND c.criterion_embedding IS NULL"
    cur.execute(f"""
        SELECT c.id,
               l.code,
               l.title,
               l.body_system,
               c.path,
               c.criterion_text,
               c.keywords
        FROM ssa_listing_criteria c
        JOIN ssa_listings l ON l.id = c.listing_id
        WHERE c.is_leaf = TRUE
          AND c.criterion_text IS NOT NULL
          {embed_filter}
        ORDER BY l.code, c.path
    """)
    rows = cur.fetchall()
    mode = "ALL (--force)" if args.force else "rows with NULL embedding"
    print(f"Leaf criteria to embed: {len(rows)}  ({mode})")
    if not rows:
        print("Nothing to do. Pass --force to re-embed everything.")
        return 0

    # Embed each leaf with enriched input (code + title + body_system +
    # criterion_text + keywords).
    print("Embedding via Bedrock Titan v2 (enriched source: parent code "
          "+ title + body_system + criterion_text + keywords)...")
    start = time.time()
    success = 0
    failed: list[tuple[str, str, str]] = []
    for i, (crit_id, code, title, body_system, path,
            criterion_text, keywords) in enumerate(rows, 1):
        # ssa_listing_criteria.keywords is JSON; psycopg2 may hand it back
        # as a Python list already (json adaptor) or as a raw str depending
        # on column type and connection settings. Normalize defensively.
        kw_list: list[str] | None
        if keywords is None:
            kw_list = None
        elif isinstance(keywords, list):
            kw_list = keywords
        elif isinstance(keywords, str):
            try:
                kw_list = json.loads(keywords)
                if not isinstance(kw_list, list):
                    kw_list = None
            except json.JSONDecodeError:
                kw_list = None
        else:
            kw_list = None

        embed_input = _build_embed_text(
            code, title, body_system, criterion_text, kw_list,
        )
        try:
            vec = embed_text(bedrock, embed_input)
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
