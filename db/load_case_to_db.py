"""Load an existing case (chunks.json + bbox sidecar) into Postgres.

Used to migrate cases that were ingested into JSON files before the DB schema
existed. Computes Bedrock Titan v2 embeddings for every chunk and every
allegation, then writes everything to Postgres in one transaction per case.

Does NOT call Textract — assumes the JSON artifacts are already on disk.

Usage:
    python db/load_case_to_db.py <case_id>
    python db/load_case_to_db.py <case_id> --skip-embeddings   # quick test
    python db/load_case_to_db.py <case_id> --recompute-embeddings  # re-do all

Reads from:
    data/<case_id>/chunks.json                 (documents, lite chunks, allegations)
    _phi/<case_id>/chunks_with_bbox.json       (chunks with bbox + confidence + ...)
    _phi/<case_id>/ingest_manifest.json        (optional; multi-PDF source info)

Writes to:
    cases, source_pdfs, documents, chunks (+ embedding),
    chunk_icd_codes, allegations (+ embedding)

Idempotent. Re-run anytime to refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np

# Ensure pipeline package is importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db import (    # noqa: E402
    transaction, upsert_case, insert_source_pdfs, insert_documents,
    insert_chunks, insert_chunk_icd_codes, insert_allegations,
    set_case_status,
)
from pipeline.chunks import extract_icd_codes, Chunk   # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent

INFERENCE_PROFILE_ARN = os.environ.get(
    "TITAN_INFERENCE_PROFILE_ARN",
    # Set TITAN_INFERENCE_PROFILE_ARN to your Bedrock application-inference-profile ARN.
    # Example shape: arn:aws:bedrock:<region>:<account>:application-inference-profile/<id>
    "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/REPLACE_ME",
)
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
#  Bedrock embedding (same pattern as the listing/criterion workers)
# ---------------------------------------------------------------------------

def make_bedrock_client():
    session = boto3.Session(profile_name="user", region_name="us-east-1")
    return session.client("bedrock-runtime")


def embed_text(client, text: str) -> np.ndarray:
    body = {"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True}
    response = client.invoke_model(
        modelId=INFERENCE_PROFILE_ARN,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    return np.array(
        json.loads(response["body"].read())["embedding"],
        dtype=np.float32,
    )


def embed_batch(client, texts: list[str], *, label: str) -> list[np.ndarray]:
    """Sequentially embed a list of texts. Titan v2 has no batch API."""
    print(f"  Embedding {len(texts)} {label} via Bedrock Titan v2...")
    start = time.time()
    out: list[np.ndarray] = []
    for i, text in enumerate(texts, 1):
        out.append(embed_text(client, text))
        if i % 25 == 0 or i == len(texts):
            elapsed = time.time() - start
            print(f"    [{i:>4}/{len(texts)}]  "
                  f"({elapsed:.0f}s elapsed, {i/elapsed:.1f} req/s)")
    return out


# ---------------------------------------------------------------------------
#  Read JSON artifacts
# ---------------------------------------------------------------------------

def load_case_json(case_id: str) -> tuple[dict, dict, dict | None]:
    """Returns (chunks_json, bbox_sidecar, manifest_or_None)."""
    chunks_path = REPO_ROOT / "data" / case_id / "chunks.json"
    bbox_path = REPO_ROOT / "_phi" / case_id / "chunks_with_bbox.json"
    manifest_path = REPO_ROOT / "_phi" / case_id / "ingest_manifest.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found: {chunks_path}")
    if not bbox_path.exists():
        raise FileNotFoundError(
            f"chunks_with_bbox.json not found: {bbox_path}. "
            f"Re-run ingest to regenerate the bbox sidecar."
        )

    chunks_json = json.loads(chunks_path.read_text(encoding="utf-8"))
    bbox = json.loads(bbox_path.read_text(encoding="utf-8"))
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return chunks_json, bbox, manifest


def merge_chunk_metadata(chunks_lite: list[dict],
                        chunks_with_bbox: list[dict]) -> list[dict]:
    """Merge the lite chunks.json shape with the bbox sidecar's richer fields.

    The bbox sidecar is authoritative for everything except (some) doc_meta
    fields. We index by chunk_id."""
    bbox_by_id = {c["chunk_id"]: c for c in chunks_with_bbox}
    merged: list[dict] = []
    for c in chunks_lite:
        rich = bbox_by_id.get(c["chunk_id"]) or {}
        merged.append({
            **c,
            "bbox":             rich.get("bbox_by_page"),
            "ocr_confidence":   rich.get("ocr_confidence"),
            "layout_block_type": rich.get("layout_block_type"),
            "is_table":         rich.get("is_table", False),
        })
    return merged


# ---------------------------------------------------------------------------
#  ICD extraction shim (uses pipeline.chunks.extract_icd_codes)
# ---------------------------------------------------------------------------

def extract_icd_from_chunks(chunks_lite: list[dict]) -> list[dict]:
    """Adapt the dict-shaped chunks to the dataclass extract_icd_codes wants.
    Returns hits in the format insert_chunk_icd_codes expects."""
    # Build minimal Chunk dataclasses
    objs = [
        Chunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            section=c.get("section", ""),
            text=c.get("text", ""),
            page_start=c.get("page_start", 0),
            page_end=c.get("page_end", 0),
        )
        for c in chunks_lite
    ]
    return extract_icd_codes(objs)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Load an existing case's chunks.json into Postgres "
                    "with Bedrock Titan v2 embeddings.",
    )
    p.add_argument("case_id", help="case_id matching data/<case_id>/chunks.json")
    p.add_argument(
        "--skip-embeddings", action="store_true",
        help="Insert chunks with embedding=NULL (for fast testing of the load path)",
    )
    args = p.parse_args()

    print(f"=== Loading case {args.case_id} into Postgres ===\n")

    # 1. Read artifacts
    print("[1/5] Reading JSON artifacts...")
    chunks_json, bbox_sidecar, manifest = load_case_json(args.case_id)
    docs = chunks_json.get("documents", [])
    chunks_lite = chunks_json.get("chunks", [])
    allegations = chunks_json.get("allegations", [])
    chunks_with_bbox = bbox_sidecar.get("chunks", [])
    print(f"      {len(docs)} sub-docs, {len(chunks_lite)} chunks, "
          f"{len(allegations)} allegations")
    if manifest:
        print(f"      manifest: {len(manifest.get('pdfs', []))} source PDF(s)")

    # 2. Merge chunk metadata
    chunks = merge_chunk_metadata(chunks_lite, chunks_with_bbox)

    # 3. Compute embeddings (chunks + allegations)
    if args.skip_embeddings:
        print("[2/5] Skipping embeddings (--skip-embeddings)")
        chunk_embeddings = None
        allegation_embeddings = None
    else:
        print("[2/5] Computing embeddings via Bedrock Titan v2...")
        bedrock = make_bedrock_client()
        chunk_embeddings = embed_batch(
            bedrock, [c["text"] for c in chunks], label="chunks",
        )
        if allegations:
            # Enrich each allegation with a minimal medical-context prefix
            # before embedding. "Diagnosis: <text>" anchors the embedding
            # in medical vocabulary without diluting the vector mass with
            # source-tag boilerplate. Mirrors the listing-side enrichment
            # (code + title + body_system + ...) — both sides land in the
            # same neighborhood instead of bare-word vs clinical-paragraph.
            def _enrich_allegation(a: dict) -> str:
                text = (a.get("text") or "").strip()
                return f"Diagnosis: {text}"

            allegation_embeddings = embed_batch(
                bedrock,
                [_enrich_allegation(a) for a in allegations],
                label="allegations",
            )
        else:
            allegation_embeddings = None

    # 4. Extract ICD hits before opening the DB transaction
    print("[3/5] Extracting ICD codes from chunks...")
    icd_hits = extract_icd_from_chunks(chunks_lite)
    print(f"      found {len(icd_hits)} ICD hit(s) across "
          f"{len({h['chunk_id'] for h in icd_hits})} chunk(s)")

    # 5. Write everything to DB in one transaction
    print("[4/5] Writing to Postgres...")
    with transaction() as conn:
        case_pk = upsert_case(
            conn, args.case_id, status="ingesting",
            notes=chunks_json.get("_note"),
        )
        print(f"      case pk = {case_pk}")

        # Source PDFs from manifest (multi-PDF case)
        source_pdf_map: dict[str, int] = {}
        if manifest and manifest.get("pdfs"):
            source_pdf_map = insert_source_pdfs(
                conn, case_pk, manifest["pdfs"],
            )
            print(f"      {len(source_pdf_map)} source_pdf rows")

        doc_id_map = insert_documents(conn, case_pk, docs, source_pdf_map)
        print(f"      {len(doc_id_map)} document rows")

        chunk_id_map = insert_chunks(
            conn, case_pk, doc_id_map, chunks, embeddings=chunk_embeddings,
        )
        print(f"      {len(chunk_id_map)} chunk rows")

        n_icd = insert_chunk_icd_codes(conn, chunk_id_map, icd_hits)
        print(f"      {n_icd} chunk_icd_codes rows")

        n_alleg = insert_allegations(
            conn, case_pk, chunk_id_map, allegations,
            embeddings=allegation_embeddings,
        )
        print(f"      {n_alleg} allegation rows")

        set_case_status(conn, case_pk, "ingested")
        print(f"      case status -> 'ingested'")

    print("\n[5/5] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
