"""Data access layer (DAL) for case-side writes.

Wraps psycopg2 + pgvector with idempotent upsert/insert functions that mirror
the case JSON shape. Designed so existing pipeline code can persist to the DB
with minimal changes — the caller just passes its existing dataclasses.

Connection: reads DATABASE_URL (SQLAlchemy-style or libpq-style accepted)
and PGPASSWORD env var. Same convention as the embedding workers.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


# ---------------------------------------------------------------------------
#  Connection
# ---------------------------------------------------------------------------

def connect_pg():
    """Connect to Postgres using DATABASE_URL or PG* env vars.

    DATABASE_URL accepts both 'postgresql://...' and 'postgresql+psycopg2://...'
    forms. Password may be embedded in the URL OR provided via PGPASSWORD.
    """
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
        kwargs = {}  # rely on PGHOST/PGUSER/PGPASSWORD/PGDATABASE
    conn = psycopg2.connect(**kwargs)
    register_vector(conn)
    return conn


@contextmanager
def transaction() -> Iterator[psycopg2.extensions.connection]:
    """Open a connection + transaction. Commits on clean exit, rolls back on
    exception. Use for write-heavy operations where you want all-or-nothing
    semantics across many statements."""
    conn = connect_pg()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Cases
# ---------------------------------------------------------------------------

def upsert_case(conn, case_id: str, *, status: str = "ingested",
                member_id: str | None = None, notes: str | None = None) -> int:
    """Insert or update a case row. Returns the case's integer id (PK)."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cases (case_id, member_id, status, notes)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (case_id) DO UPDATE SET
            status     = EXCLUDED.status,
            member_id  = COALESCE(EXCLUDED.member_id, cases.member_id),
            notes      = COALESCE(EXCLUDED.notes,     cases.notes),
            updated_at = NOW()
        RETURNING id
    """, (case_id, member_id, status, notes))
    return cur.fetchone()[0]


def set_case_status(conn, case_id_pk: int, status: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE cases SET status = %s, updated_at = NOW() WHERE id = %s",
        (status, case_id_pk),
    )


# ---------------------------------------------------------------------------
#  Source PDFs
# ---------------------------------------------------------------------------

def insert_source_pdfs(conn, case_id_pk: int, pdfs: list[dict]) -> dict[str, int]:
    """Insert source_pdfs rows. Returns {local_path: source_pdf_pk} for
    later joining when we insert documents.

    Idempotent via delete-then-insert scoped to this case.
    """
    cur = conn.cursor()
    # documents.source_pdf_id has an FK to source_pdfs(id). On databases
    # where the FK is still ON DELETE NO ACTION (older schema), naively
    # deleting source_pdfs throws ForeignKeyViolation if any documents
    # rows still reference us. Null out those refs first — insert_documents
    # repopulates them via page-range inference on the next step anyway.
    # On databases that have applied the ON DELETE SET NULL migration this
    # UPDATE is redundant but cheap (0 rows match for fresh cases).
    cur.execute(
        "UPDATE documents SET source_pdf_id = NULL WHERE case_id = %s",
        (case_id_pk,),
    )
    cur.execute("DELETE FROM source_pdfs WHERE case_id = %s", (case_id_pk,))
    out: dict[str, int] = {}
    for p in pdfs:
        cur.execute("""
            INSERT INTO source_pdfs
                (case_id, role, s3_bucket, s3_key, local_path,
                 combined_page_start, combined_page_end, page_count,
                 textract_job_id, sha256)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            case_id_pk,
            p.get("role"),
            p.get("s3_bucket") or p.get("bucket"),
            p.get("s3_key") or p.get("key"),
            p.get("local_path"),
            p.get("combined_page_start") or (p.get("combined_pages") or [None])[0],
            p.get("combined_page_end")   or (p.get("combined_pages") or [None, None])[1],
            p.get("page_count"),
            p.get("textract_job_id"),
            p.get("sha256"),
        ))
        local_path = p.get("local_path") or ""
        out[local_path] = cur.fetchone()[0]
    return out


# ---------------------------------------------------------------------------
#  Documents
# ---------------------------------------------------------------------------

def insert_documents(conn, case_id_pk: int, documents: list[dict],
                     source_pdf_map: dict[str, int] | None = None
                     ) -> dict[str, int]:
    """Insert documents rows. Returns {doc_id (string): documents.id (int)}
    so chunks can reference them by FK.

    Idempotent via delete-then-insert scoped to this case.

    source_pdf_id resolution (in order):
      1. Explicit: each doc dict may set "source_pdf_local_path" matching a
         key in source_pdf_map (path -> source_pdfs.id from insert_source_pdfs).
      2. Inferred: if step 1 doesn't resolve, after the INSERT we run a
         single UPDATE that joins page_start to source_pdfs.combined_page_*.
         Covers callers (older ingest paths, chunks.json-only loaders) that
         don't set source_pdf_local_path explicitly.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE case_id = %s", (case_id_pk,))
    out: dict[str, int] = {}
    for d in documents:
        page_range = d.get("page_range_in_packet") or [None, None]
        # Resolve source_pdf_id if we have a mapping (multi-PDF case)
        source_pdf_id = None
        if source_pdf_map and "source_pdf_local_path" in d:
            source_pdf_id = source_pdf_map.get(d["source_pdf_local_path"])
        cur.execute("""
            INSERT INTO documents
                (case_id, doc_id, doc_type, doc_title, encounter_date,
                 facility, page_start, page_end, source_pdf_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            case_id_pk,
            d["doc_id"],
            d.get("doc_type"),
            d.get("doc_title"),
            _parse_date(d.get("encounter_date")),
            d.get("facility"),
            page_range[0],
            page_range[1],
            source_pdf_id,
        ))
        out[d["doc_id"]] = cur.fetchone()[0]

    # Step 2: page-range inference for rows that still have NULL source_pdf_id.
    # Joins document.page_start to the source_pdf whose combined_page range
    # covers it. This is the same logic as db/backfill_document_source_pdf_id.sql
    # — running it inline keeps fresh ingests from also leaving NULL FKs.
    cur.execute("""
        UPDATE documents d
        SET source_pdf_id = sp.id
        FROM source_pdfs sp
        WHERE d.case_id = %s
          AND d.source_pdf_id IS NULL
          AND sp.case_id = d.case_id
          AND d.page_start IS NOT NULL
          AND sp.combined_page_start IS NOT NULL
          AND sp.combined_page_end   IS NOT NULL
          AND d.page_start BETWEEN sp.combined_page_start AND sp.combined_page_end
    """, (case_id_pk,))

    return out


# ---------------------------------------------------------------------------
#  Chunks
# ---------------------------------------------------------------------------

def insert_chunks(conn, case_id_pk: int, doc_id_map: dict[str, int],
                  chunks: list[dict],
                  embeddings: list[np.ndarray] | None = None
                  ) -> dict[str, int]:
    """Insert chunks rows. Returns {chunk_id (string): chunks.id (int)} for
    later joining when we insert chunk_icd_codes and allegations.

    Idempotent: delete-then-insert scoped to this case.

    Embeddings is an optional parallel list. If None, embedding column stays
    NULL — populate later via db/compute_chunk_embeddings.py.
    """
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(
            f"len(chunks)={len(chunks)} but len(embeddings)={len(embeddings)}"
        )

    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE case_id = %s", (case_id_pk,))

    out: dict[str, int] = {}
    for i, c in enumerate(chunks):
        document_id = doc_id_map.get(c["doc_id"])
        if document_id is None:
            raise KeyError(
                f"chunk {c['chunk_id']!r} references doc_id {c['doc_id']!r} "
                f"which wasn't in the documents map"
            )
        emb = embeddings[i] if embeddings is not None else None
        bbox = c.get("bbox")
        if bbox is None and "bbox_by_page" in c:
            bbox = c["bbox_by_page"]

        # Defensive truncation for column-width safety. The chunker caps
        # the section slug in chunk_id to 60 chars via _safe(), and the
        # migrate_chunks_section_to_text.sql migration widens
        # chunks.section to TEXT. The truncations below are belt-and-
        # suspenders for databases that haven't been migrated yet, so we
        # never crash on StringDataRightTruncation. Section is preserved
        # in full when the column allows; chunk_id stays unique within
        # the case via its rowN:idx suffix.
        section = c.get("section")
        if section and len(section) > 200:
            section = section[:200].rstrip() + "…"
        chunk_id = c["chunk_id"]
        if len(chunk_id) > 128:
            chunk_id = chunk_id[:128]

        cur.execute("""
            INSERT INTO chunks
                (case_id, document_id, chunk_id, section, page_start, page_end,
                 bbox, text, ocr_confidence, layout_block_type, is_table,
                 embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            case_id_pk,
            document_id,
            chunk_id,
            section,
            c.get("page_start"),
            c.get("page_end"),
            json.dumps(bbox) if bbox else None,
            c["text"],
            c.get("ocr_confidence"),
            c.get("layout_block_type"),
            c.get("is_table", False),
            emb,
        ))
        out[c["chunk_id"]] = cur.fetchone()[0]
    return out


# ---------------------------------------------------------------------------
#  Chunk extractions (ICD codes)
# ---------------------------------------------------------------------------

def insert_chunk_icd_codes(conn, chunk_id_map: dict[str, int],
                           icd_hits: list[dict]) -> int:
    """Insert chunk_icd_codes rows. Returns count inserted.

    icd_hits format (from pipeline.chunks.extract_icd_codes):
        [{"code": "C18.9", "chunk_id": "doc-04:p7:VisitDx:0", "context": "..."}]
    """
    cur = conn.cursor()
    inserted = 0
    for hit in icd_hits:
        chunk_pk = chunk_id_map.get(hit["chunk_id"])
        if chunk_pk is None:
            continue  # chunk wasn't inserted, skip
        cur.execute("""
            INSERT INTO chunk_icd_codes (chunk_id, code, description, position_in_text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id, code, position_in_text) DO NOTHING
        """, (
            chunk_pk,
            hit["code"][:16],
            (hit.get("context") or "")[:300],
            hit.get("position", 0),
        ))
        inserted += cur.rowcount
    return inserted


# ---------------------------------------------------------------------------
#  Allegations
# ---------------------------------------------------------------------------

def insert_allegations(conn, case_id_pk: int, chunk_id_map: dict[str, int],
                       allegations: list[dict],
                       embeddings: list[np.ndarray] | None = None) -> int:
    """Insert allegations rows. Idempotent: delete-then-insert for this case.

    allegations format (from pipeline.ingest_real.auto_extract_allegations):
        [{"text": "...", "source": "visit_diagnoses",
          "source_chunk_id": "doc-04:p7:VisitDx:0"}]
    """
    if embeddings is not None and len(embeddings) != len(allegations):
        raise ValueError(
            f"len(allegations)={len(allegations)} but len(embeddings)={len(embeddings)}"
        )

    cur = conn.cursor()
    cur.execute("DELETE FROM allegations WHERE case_id = %s", (case_id_pk,))
    for i, a in enumerate(allegations):
        chunk_pk = chunk_id_map.get(a.get("source_chunk_id", ""))
        emb = embeddings[i] if embeddings is not None else None
        cur.execute("""
            INSERT INTO allegations
                (case_id, text, source, source_chunk_id,
                 manually_curated, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            case_id_pk,
            a["text"],
            a.get("source"),
            chunk_pk,
            a.get("manually_curated", False),
            emb,
        ))
    return len(allegations)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

# ============================================================================
#  Read-side: load a case + its chunks/allegations + the listing reference data
#  out of Postgres. Returns dataclasses (or plain dicts when a dataclass would
#  be more friction than help) so run.py can swap from JSON to DB-loaded data
#  without rewriting downstream stages.
# ============================================================================

def get_case_pk_by_case_id(conn, case_id_str: str) -> int | None:
    """Resolve the human-readable case_id (e.g. 'redacted-001-fresh') to the
    integer primary key used by foreign keys. Returns None if not found."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM cases WHERE case_id = %s", (case_id_str,))
    row = cur.fetchone()
    return row[0] if row else None


def get_source_pdfs(conn, case_pk: int) -> list[dict]:
    """Return all source_pdfs rows for a case, ordered by their page offset
    in the combined source.pdf so callers can rebuild the same combined PDF.

    Each dict has: role, s3_bucket, s3_key, local_path, combined_page_start,
    combined_page_end, page_count. Empty list if the case wasn't ingested via
    db/ingest_s3_to_db.py (i.e. no S3-tracked PDFs).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT role, s3_bucket, s3_key, local_path,
               combined_page_start, combined_page_end, page_count
        FROM source_pdfs
        WHERE case_id = %s
        ORDER BY combined_page_start NULLS LAST, id
    """, (case_pk,))
    cols = ("role", "s3_bucket", "s3_key", "local_path",
            "combined_page_start", "combined_page_end", "page_count")
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_chunks_for_case(conn, case_pk: int, *, with_embedding: bool = True
                        ) -> list[dict]:
    """Return all chunks for a case, joined with document metadata.

    Each dict has the keys the rest of the pipeline expects:
      id (int pk), chunk_id (str), doc_id (str), doc_type, doc_title,
      encounter_date, section, page_start, page_end, text, bbox, embedding.
    """
    cur = conn.cursor()
    cols = (
        "c.id, c.chunk_id, d.doc_id, d.doc_type, d.doc_title, "
        "d.encounter_date, c.section, c.page_start, c.page_end, c.text, "
        "c.bbox, c.ocr_confidence, c.layout_block_type, c.is_table"
    )
    if with_embedding:
        cols += ", c.embedding"
    cur.execute(f"""
        SELECT {cols}
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.case_id = %s
        ORDER BY c.page_start, c.id
    """, (case_pk,))
    out: list[dict] = []
    for row in cur.fetchall():
        rec = {
            "id":               row[0],
            "chunk_id":         row[1],
            "doc_id":           row[2],
            "doc_type":         row[3],
            "doc_title":        row[4],
            "encounter_date":   row[5].isoformat() if row[5] else None,
            "section":          row[6],
            "page_start":       row[7],
            "page_end":         row[8],
            "text":             row[9],
            "bbox":             row[10],
            "ocr_confidence":   row[11],
            "layout_block_type": row[12],
            "is_table":         row[13],
        }
        if with_embedding:
            rec["embedding"] = row[14]
        out.append(rec)
    return out


def get_allegations_for_case(conn, case_pk: int) -> list[dict]:
    """Return all allegations for a case (with embeddings)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, source, source_chunk_id, manually_curated, embedding
        FROM allegations
        WHERE case_id = %s
        ORDER BY id
    """, (case_pk,))
    return [
        {
            "id":                row[0],
            "text":              row[1],
            "source":            row[2],
            "source_chunk_pk":   row[3],
            "manually_curated":  row[4],
            "embedding":         row[5],
        }
        for row in cur.fetchall()
    ]


def get_listings(conn, *, with_embedding: bool = False) -> list[dict]:
    """Return all SSA listings. Optionally include summary_embedding for
    candidate-identification work in Python."""
    cur = conn.cursor()
    cols = "id, code, title, body_system, summary, rule_json"
    if with_embedding:
        cols += ", summary_embedding"
    cur.execute(f"SELECT {cols} FROM ssa_listings ORDER BY code")
    out = []
    for row in cur.fetchall():
        rec = {
            "id":           row[0],
            "code":         row[1],
            "title":        row[2],
            "body_system":  row[3],
            "summary":      row[4],
            "rule_json":    row[5],
        }
        if with_embedding:
            rec["summary_embedding"] = row[6]
        out.append(rec)
    return out


def get_listing_synonyms(conn, listing_pk) -> dict[str, list[str]]:
    # listing_pk is UUID — psycopg2 binds str/UUID transparently.
    """Return the synonym map for one listing in {canonical: [variants]} shape."""
    cur = conn.cursor()
    cur.execute("""
        SELECT canonical, variant
        FROM ssa_listing_synonyms
        WHERE listing_id = %s
    """, (listing_pk,))
    out: dict[str, list[str]] = {}
    for canonical, variant in cur.fetchall():
        out.setdefault(canonical, []).append(variant)
    return out


def get_listing_criteria_tree(conn, listing_pk) -> dict:
    # listing_pk is UUID — see ssa_listings.id in db/schema_v1.sql.
    """Return the listing's full rule_json with each node's DB id attached
    inline at `_db_id`. The tree shape matches what evaluate.py and
    consolidate.py expect."""
    cur = conn.cursor()
    cur.execute(
        "SELECT rule_json FROM ssa_listings WHERE id = %s", (listing_pk,)
    )
    row = cur.fetchone()
    if not row:
        raise KeyError(f"No ssa_listings row with id={listing_pk}")
    rule_json = row[0]

    # Index every (path → criterion id) pair so we can decorate the tree
    cur.execute("""
        SELECT path, id FROM ssa_listing_criteria
        WHERE listing_id = %s
    """, (listing_pk,))
    path_to_id = {p: cid for p, cid in cur.fetchall()}

    def annotate(node: dict) -> None:
        if node.get("path") in path_to_id:
            node["_db_id"] = path_to_id[node["path"]]
        for child in node.get("children", []):
            annotate(child)

    annotate(rule_json)
    return rule_json


def _parse_date(s: str | None) -> str | None:
    """Best-effort parse of a date string into ISO format (YYYY-MM-DD).
    Returns None if input is None or can't be parsed; psycopg2 will INSERT
    None as SQL NULL."""
    if not s:
        return None
    # Accept ISO already, MM/DD/YY, MM/DD/YYYY, MM-DD-YYYY
    import re
    s = s.strip()
    # ISO already
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    # US format with 2- or 4-digit year
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m:
        mo, da, yr = m.group(1), m.group(2), m.group(3)
        if len(yr) == 2:
            yr = "20" + yr if int(yr) < 70 else "19" + yr
        return f"{yr.zfill(4)}-{mo.zfill(2)}-{da.zfill(2)}"
    return None
