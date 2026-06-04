"""Make output/<case_id>/ self-contained so it can be moved to SharePoint
or zipped + shipped without losing citation hyperlinks.

The HTML forms cite chart evidence via relative URLs like
"source.pdf#page=12" — those links only resolve if source.pdf actually
sits alongside the HTML. This module ensures that file is present by
fetching it from S3 (using source_pdfs records in the DB) and combining
the originals into a single PDF, cached on disk so re-runs are fast.

Why we don't reuse _phi/<case_id>/source.pdf:
  - _phi/ is treated as an ephemeral PHI workspace; reviewers may clean
    it up between matcher runs.
  - SharePoint hand-off only covers output/<case_id>/ — we can't ship
    _phi/ as part of the case bundle.

Why we don't ship the original S3-side PDFs:
  - There are typically several per case (one Supplement + multiple AI
    files) with names that aren't reviewer-friendly. Combining matches
    the page numbering the matcher recorded against (chunks.page_start
    references the COMBINED PDF, not the per-file pagination).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def ensure_source_pdf(
    conn,
    case_pk: int,
    out_dir: Path,
    *,
    phi_dir: Path | None = None,
    profile_name: str = None,
    region_name: str = None,
    force: bool = False,
) -> Path | None:
    """Make sure out_dir/source.pdf exists; return its path (or None on failure).

    Resolution order:
      1. out_dir/source.pdf already cached    -> use it as-is
      2. phi_dir/source.pdf is on local disk  -> copy into out_dir
      3. Fetch each source_pdfs row from S3, combine in combined_page_start
         order, save to out_dir/source.pdf

    Args:
        conn: open psycopg2 connection (for the get_source_pdfs lookup)
        case_pk: cases.id (int PK)
        out_dir: the rendered-output directory (output/<case_id>)
        phi_dir: optional local PHI workspace (HERE/_phi/<case_id>) that may
                 contain a pre-built source.pdf to short-circuit S3 fetch
        profile_name: AWS profile (default: $AWS_PROFILE or 'user')
        region_name: AWS region (default: $AWS_REGION or 'us-east-1')
        force: re-fetch even if the file is already cached

    Returns:
        Path to out_dir/source.pdf on success, or None if:
          - the source_pdfs table is empty for this case (case wasn't
            ingested via db/ingest_s3_to_db.py), and no phi_dir fallback
            file exists either
          - all S3 downloads fail
    """
    out_pdf = out_dir / "source.pdf"

    # 1. Cached?
    if not force and out_pdf.exists() and out_pdf.stat().st_size > 0:
        return out_pdf

    # 2. Local phi_dir copy?
    if phi_dir is not None:
        candidate = phi_dir / "source.pdf"
        if candidate.exists() and candidate.stat().st_size > 0:
            import shutil
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, out_pdf)
            print(f"      bundle: copied {candidate} -> {out_pdf}")
            return out_pdf

    # 3. Fetch from S3
    from .db import get_source_pdfs
    rows = get_source_pdfs(conn, case_pk)
    if not rows:
        return None

    # Filter to rows that actually have an S3 location. local_path-only rows
    # (synthetic / hand-loaded cases) can't be re-fetched.
    s3_rows = [r for r in rows if r.get("s3_bucket") and r.get("s3_key")]
    if not s3_rows:
        return None

    try:
        import boto3
    except ImportError:
        print("      bundle: boto3 not installed; cannot fetch source PDFs")
        return None

    profile = profile_name or os.environ.get("AWS_PROFILE", "user")
    region  = region_name  or os.environ.get("AWS_REGION", "us-east-1")
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3", region_name=region)

    import tempfile
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="source-pdf-fetch-") as tmpdir:
        tmp = Path(tmpdir)
        local_paths: list[Path] = []
        for r in s3_rows:
            bucket, key = r["s3_bucket"], r["s3_key"]
            # Preserve the key's basename so we can tell what each came from
            # if we ever need to inspect the temp files (we don't normally).
            local = tmp / Path(key).name
            try:
                print(f"      bundle: fetching s3://{bucket}/{key}")
                s3.download_file(bucket, key, str(local))
                local_paths.append(local)
            except Exception as e:
                print(f"      bundle: WARNING — failed to fetch "
                      f"s3://{bucket}/{key}: {e}")
                continue

        if not local_paths:
            return None

        # Combine using pymupdf (same library as the annotation step uses,
        # so we don't pull in a new dependency).
        try:
            import fitz   # PyMuPDF
        except ImportError:
            print("      bundle: pymupdf not installed; cannot combine PDFs")
            return None

        combined = fitz.open()
        try:
            for p in local_paths:
                try:
                    with fitz.open(p) as src:
                        combined.insert_pdf(src)
                except Exception as e:
                    print(f"      bundle: WARNING — failed to read {p.name}: {e}")
                    continue
            if combined.page_count == 0:
                return None
            combined.save(str(out_pdf))
        finally:
            combined.close()

    print(f"      bundle: wrote combined source.pdf "
          f"({out_pdf.stat().st_size // 1024} KB) to {out_pdf}")
    return out_pdf


# ---------------------------------------------------------------------------
#  Bbox sidecar reconstruction (for annotated-PDF highlights)
# ---------------------------------------------------------------------------

def reconstruct_bbox_sidecar(
    conn,
    case_pk: int,
    output_path: Path,
) -> Path | None:
    """Rebuild the chunks_with_bbox.json sidecar from chunks.bbox in the DB.

    Used when a case is run on a machine that doesn't have
    _phi/<case>/chunks_with_bbox.json on disk. The sidecar feeds
    pipeline/annotate_pdf, which draws yellow highlights at each cited
    chunk's bbox to produce source_annotated.pdf.

    The on-disk format (from pipeline/ingest_real) is:
        {"chunks": [
            {"chunk_id": "doc-01-p3-clinical",
             "bbox_by_page": {"3": [left, top, right, bottom]}},
            ...
        ]}

    The DB column chunks.bbox is JSONB shaped as {"<page>": [l,t,r,b], ...}
    — i.e. the inner bbox_by_page directly. We just need to wrap each row.

    Returns the written path, or None if there are no chunks-with-bbox to
    reconstruct (e.g. an older case ingested before bbox was captured).
    """
    from .db import get_chunks_for_case

    rows = get_chunks_for_case(conn, case_pk, with_embedding=False)
    if not rows:
        return None

    chunks_with_bbox: list[dict] = []
    for r in rows:
        bbox = r.get("bbox")
        if not bbox:
            # Chunk has no bbox info — common for hand-loaded cases or
            # the older ingest path that didn't persist bbox to DB. Skip.
            continue
        chunks_with_bbox.append({
            "chunk_id":     r["chunk_id"],
            "bbox_by_page": bbox,
        })
    if not chunks_with_bbox:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"chunks": chunks_with_bbox}),
        encoding="utf-8",
    )
    return output_path
