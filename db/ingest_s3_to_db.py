"""End-to-end: S3 PDF(s) → Textract → chunks → Postgres → matcher → forms.

One command that runs the full pipeline, no manual chaining.

Usage:
    # Full pipeline (default) — ingest + match + render reviewer forms
    python db/ingest_s3_to_db.py <case_id> --pdfs <bucket>/<key>:<role> ...

    # Ingest only (skip matcher, e.g. for bulk historical loads)
    python db/ingest_s3_to_db.py <case_id> --pdfs ... --no-match

    # Single-PDF shorthand
    python db/ingest_s3_to_db.py <case_id> <bucket> <key>

Pipeline (always runs Stages 1+2; Stage 3 runs by default, suppress with --no-match):
  1. Textract async start_document_analysis on each PDF (LAYOUT, TABLES, FORMS)
  2. Concatenate into combined source.pdf, remap page numbers
  3. Sub-document detection + chunking via existing pipeline.ingest_real
  4. Bedrock Titan v2 embeddings for every chunk + allegation
  5. Write everything to Postgres in one transaction per case
  6. (--match, default-on) Run run_from_db: SQL candidate identification +
     per-leaf retrieval + Bedrock Claude evaluation + persist to layers
     1-3 (chunk_leaf_matches, leaf_assessments + evidence,
     listing_assessments + case_summaries) + render reviewer forms

Cost (typical 40-page packet):
  - Textract LAYOUT+TABLES+FORMS:  ~$4    (Stage 1, one-time per packet)
  - Bedrock Titan v2 embeddings:   ~$0.005 (Stage 4)
  - Bedrock Claude evaluation:     ~$1-3   (Stage 6, only with --match)
  - Total:                          ~$5-7  per case end-to-end

Side-effect files (gitignored, contain PHI):
  data/<case_id>/chunks.json                 lite chunks shape
  _phi/<case_id>/source.pdf                  combined PDF (for citation links)
  _phi/<case_id>/source_NN.pdf               per-PDF originals
  _phi/<case_id>/textract_raw.json           combined Textract response
  _phi/<case_id>/chunks_with_bbox.json       chunks + bbox + confidence
  _phi/<case_id>/ingest_manifest.json        which PDF → which page range
  _phi/<case_id>/source_annotated.pdf        (with --match) bbox-highlighted PDF
  output/<case_id>/*.md, *.html              (with --match) reviewer forms

The local artifacts get written even though we also persist to DB — useful
for debugging, re-chunking without re-Textract, and as a fallback if DB
state diverges.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make pipeline + sibling scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest_real import ingest_multi_pdf_case   # noqa: E402


def parse_pdf_spec(s: str) -> dict:
    """Parse '<bucket>/<key>:role' into {'bucket', 'key', 'role'}.

    role defaults to 'medical_evidence' if not specified.
    The first '/' separates bucket from key.
    The last ':' separates role from the rest (so colons inside the key are OK).
    """
    role = "medical_evidence"
    if ":" in s:
        # Split off the role from the right
        s, _, role = s.rpartition(":")
    if "/" not in s:
        raise argparse.ArgumentTypeError(
            f"PDF spec {s!r} must include bucket and key separated by '/'"
        )
    bucket, _, key = s.partition("/")
    return {"bucket": bucket, "key": key, "role": role}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ingest case PDFs from S3 directly into Postgres "
                    "(Textract + chunking + embeddings + DB write).",
    )
    p.add_argument("case_id",
                   help="case identifier — outputs land in data/<case_id>/ and _phi/<case_id>/")

    # Two ways to specify PDFs:
    #   single PDF:  positional bucket + key
    #   multi PDF:   --pdfs flag with one or more bucket/key[:role] specs
    p.add_argument("bucket", nargs="?", help="single-PDF mode: S3 bucket")
    p.add_argument("key",    nargs="?", help="single-PDF mode: S3 object key")
    p.add_argument(
        "--pdfs", nargs="+", type=parse_pdf_spec,
        help="multi-PDF mode: one or more '<bucket>/<key>:<role>' specs "
             "(role optional, defaults to medical_evidence)",
    )

    p.add_argument("--profile",   default="user",       help="AWS profile (default: user)")
    p.add_argument("--region",    default="us-east-1",  help="AWS region (default: us-east-1)")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Insert chunks with embedding=NULL (faster, no Bedrock cost)")
    p.add_argument("--no-match", action="store_true",
                   help="Skip the matcher (Stage 3). Use when bulk-loading cases "
                        "without evaluating them — e.g. historical backfill. "
                        "By default the matcher runs and produces reviewer forms.")
    args = p.parse_args()

    # Resolve PDFs into the list-of-dicts shape ingest_multi_pdf_case wants
    if args.pdfs:
        pdfs = args.pdfs
    elif args.bucket and args.key:
        pdfs = [{"bucket": args.bucket, "key": args.key, "role": "medical_evidence"}]
    else:
        p.error("Either provide positional <bucket> <key> for a single PDF, "
                "or --pdfs <bucket>/<key>[:<role>] ... for multi-PDF cases.")
        return 2

    print(f"=== End-to-end ingest: {args.case_id} ===")
    print(f"PDFs to ingest:")
    for pdf in pdfs:
        print(f"  s3://{pdf['bucket']}/{pdf['key']}   role={pdf['role']}")
    print()

    # ---- Stage 1: Textract + chunking + write JSON artifacts ----
    print("Stage 1: Textract OCR + chunking (writes local artifacts)...")
    print("-" * 70)
    chunks_path = ingest_multi_pdf_case(
        case_id=args.case_id,
        pdfs=pdfs,
        profile_name=args.profile,
        region_name=args.region,
    )
    print(f"\nStage 1 done. chunks.json: {chunks_path}")
    print()

    # ---- Stage 2: load to DB (with embeddings) ----
    # We import here so the script can run Stage 1 even if pipeline.db's deps
    # (psycopg2 + pgvector) aren't installed yet.
    print("Stage 2: Compute embeddings + write to Postgres...")
    print("-" * 70)
    import db.load_case_to_db as loader  # noqa: E402
    # Reuse loader's main by simulating its CLI args
    argv_backup = sys.argv
    try:
        sys.argv = ["load_case_to_db.py", args.case_id]
        if args.skip_embeddings:
            sys.argv.append("--skip-embeddings")
        loader.main()
    finally:
        sys.argv = argv_backup

    # ---- Stage 3: run the matcher (unless --no-match) ----
    # SQL candidate identification + per-leaf retrieval + Bedrock Claude eval +
    # persist to chunk_leaf_matches / leaf_assessments / listing_assessments /
    # case_summaries + render markdown + HTML reviewer forms.
    if not args.no_match:
        print()
        print("Stage 3: Running matcher (Bedrock Claude eval + form rendering)...")
        print("-" * 70)
        # run.py is at project root; we already added that to sys.path above
        import run as run_module  # noqa: E402
        rc = run_module.run_from_db(args.case_id)
        if rc != 0:
            print(f"Matcher returned non-zero ({rc}); ingest already persisted.",
                  file=sys.stderr)
            return rc
    else:
        print()
        print("Stage 3 skipped (--no-match). Run later with:")
        print(f"  py run.py --from-db {args.case_id}")

    print()
    print("=== Done. Full pipeline complete. ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
