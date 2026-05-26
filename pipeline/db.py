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
    """Connect to Postgres.

    Resolution order:
      1. db/local_override.py defines `connect_pg()` — use that (local dev
         convenience; the file is gitignored so credentials never get
         committed). Pattern: copy db/local_override.example.py to
         db/local_override.py and fill in your DATABASE_URL.
      2. DATABASE_URL env var (with optional PGPASSWORD fallback).
      3. Standard libpq env vars (PGHOST / PGUSER / PGPASSWORD / PGDATABASE).

    Prints which path it used to stderr so silent fall-through bugs are
    obvious. Set PGDB_QUIET=1 in env to suppress.
    """
    quiet = os.environ.get("PGDB_QUIET") == "1"

    def _log(msg: str) -> None:
        if not quiet:
            import sys
            print(f"[connect_pg] {msg}", file=sys.stderr)

    # 1) Local override (gitignored)
    try:
        from db.local_override import connect_pg as _override
        _log("using db/local_override.py")
        conn = _override()
        register_vector(conn)
        return conn
    except ImportError as e:
        _log(f"no db/local_override.py (ImportError: {e}); trying env vars")

    # 2) DATABASE_URL env var
    url = os.environ.get("DATABASE_URL")
    if url:
        _log(f"using DATABASE_URL env var (host={urlparse(url).hostname})")
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
        # 3) Bare libpq env vars (PGHOST / PGUSER / PGPASSWORD / PGDATABASE).
        # WARNING: without any of those set, psycopg2 defaults to host='localhost',
        # which is why you'd see "connection refused localhost" errors when
        # nothing is configured.
        _log("no DATABASE_URL set; relying on PG* env vars or libpq defaults "
             "(host=localhost if PGHOST not set)")
        kwargs = {}
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
            c["chunk_id"],
            c.get("section"),
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
