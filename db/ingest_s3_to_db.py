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
  output/<case_id>/source_annotated.pdf      (with --match) bbox-highlighted PDF
                                              (lives alongside HTML so citations
                                               are relative + bundle is portable)
  output/<case_id>/*.md, *.html              (with --match) reviewer forms

The local artifacts get written even though we also persist to DB — useful
for debugging, re-chunking without re-Textract, and as a fallback if DB
state diverges.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make pipeline + sibling scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ingest_real import ingest_multi_pdf_case   # noqa: E402


# ---- Filename classification (used by --folder mode) ------------------------
#
# These patterns reflect the MassHealth case-folder naming convention:
#   - The disability supplement (member's allegation) has "Supplement" in the name.
#   - Medical evidence files are prefixed "AI" (Additional/Acquired Information),
#     word-bounded so "rail", "main" etc. don't false-positive.
#   - Everything else (release forms, MA review forms, completed prior
#     evaluations, non-PDF files) is skipped.
#
# Override via --allegation-pattern / --medical-pattern if a folder uses
# a different convention (e.g. some folders prefix medical records with
# "MR" instead of "AI" -- pass --medical-pattern "(?i)\b(AI|MR)\b").

DEFAULT_ALLEGATION_PATTERN = r"(?i)supplement"
DEFAULT_MEDICAL_PATTERN    = r"(?i)\bAI\b"


def classify_filename(filename: str,
                      allegation_pattern: str = DEFAULT_ALLEGATION_PATTERN,
                      medical_pattern: str = DEFAULT_MEDICAL_PATTERN
                      ) -> str | None:
    """Classify a filename as 'allegation_source' | 'medical_evidence' | None.

    Order matters: allegation pattern is checked first so a file matching
    both (rare but possible) gets routed to allegation.
    """
    if not filename.lower().endswith(".pdf"):
        return None
    if re.search(allegation_pattern, filename):
        return "allegation_source"
    if re.search(medical_pattern, filename):
        return "medical_evidence"
    return None


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


def parse_s3_folder(s: str) -> tuple[str, str]:
    """Parse 's3://bucket/path/' or 'bucket/path/' into (bucket, prefix)."""
    if s.startswith("s3://"):
        s = s[5:]
    s = s.strip("/")
    if "/" not in s:
        # Treat as bucket root
        return s, ""
    bucket, _, prefix = s.partition("/")
    # Ensure prefix ends with /, otherwise S3 will match neighboring folders
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def extract_case_id_from_folder(prefix: str) -> str | None:
    """Pull the 7-digit MassHealth case id from the last path component.

    Convention: case folders are named like '1234567 Psych- 100/' with the
    case id as a 7-digit prefix. The lookahead requires a non-digit or end-
    of-string after the 7 digits — so '12345678 Folder' (8 digits) fails
    rather than silently truncating to its first 7 digits. Returns None if no match.
    """
    parts = [p for p in prefix.strip("/").split("/") if p]
    if not parts:
        return None
    m = re.match(r"^(\d{7})(?=\D|$)", parts[-1])
    return m.group(1) if m else None


def list_and_classify_folder(s3_folder: str, *,
                              profile_name: str = "user",
                              region_name: str = "us-east-1",
                              allegation_pattern: str = DEFAULT_ALLEGATION_PATTERN,
                              medical_pattern: str = DEFAULT_MEDICAL_PATTERN
                              ) -> tuple[list[dict], list[str]]:
    """List PDFs under an S3 prefix and classify by filename.

    Returns (kept_pdfs, skipped_basenames) — kept_pdfs in the
    list-of-dicts shape ingest_multi_pdf_case wants, skipped_basenames
    listed for the operator to audit.
    """
    import boto3
    bucket, prefix = parse_s3_folder(s3_folder)

    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3", region_name=region_name)

    # Auto-paginate in case the folder has > 1000 objects
    kept: list[dict] = []
    skipped: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip the prefix itself if S3 returned a "directory" marker
            if key.endswith("/"):
                continue
            basename = Path(key).name
            role = classify_filename(basename,
                                     allegation_pattern, medical_pattern)
            if role is None:
                skipped.append(basename)
                continue
            kept.append({"bucket": bucket, "key": key, "role": role,
                         "basename": basename})
    return kept, skipped


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ingest case PDFs from S3 directly into Postgres "
                    "(Textract + chunking + embeddings + DB write).",
    )
    # case_id is positional but optional when --folder is given and the
    # folder name has a 7+ digit prefix (the MA convention). Auto-extract.
    p.add_argument("case_id", nargs="?",
                   help="case identifier — outputs land in data/<case_id>/ and "
                        "_phi/<case_id>/. Optional with --folder if the prefix "
                        "name starts with a 7+ digit case number.")

    # Three ways to specify PDFs:
    #   single PDF:  positional bucket + key
    #   multi PDF:   --pdfs flag with one or more bucket/key[:role] specs
    #   folder mode: --folder <s3-prefix> auto-discovers + classifies
    p.add_argument("bucket", nargs="?", help="single-PDF mode: S3 bucket")
    p.add_argument("key",    nargs="?", help="single-PDF mode: S3 object key")
    p.add_argument(
        "--pdfs", nargs="+", type=parse_pdf_spec,
        help="multi-PDF mode: one or more '<bucket>/<key>:<role>' specs "
             "(role optional, defaults to medical_evidence)",
    )
    p.add_argument(
        "--folder",
        help="folder mode: S3 prefix (e.g. 'bucket/1234567 Psych- 100/'). "
             "Lists every PDF in the prefix, classifies by filename, ingests "
             "the kept files. Default rules: 'Supplement'->allegation, "
             "'AI' (word-bounded)->medical_evidence, others skipped.",
    )
    p.add_argument(
        "--allegation-pattern", default=DEFAULT_ALLEGATION_PATTERN,
        help=f"regex for allegation files (default: {DEFAULT_ALLEGATION_PATTERN!r})",
    )
    p.add_argument(
        "--medical-pattern", default=DEFAULT_MEDICAL_PATTERN,
        help=f"regex for medical-evidence files (default: {DEFAULT_MEDICAL_PATTERN!r})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the folder classification plan and exit without ingesting. "
             "Only meaningful with --folder.",
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
    if args.folder:
        # ---- Folder mode ---------------------------------------------------
        bucket, prefix = parse_s3_folder(args.folder)
        # Auto-extract case_id from folder name if not provided. Tracking the
        # source explicitly so we can mark "(auto-detected)" in the banner.
        case_id_was_auto = False
        if not args.case_id:
            args.case_id = extract_case_id_from_folder(prefix)
            case_id_was_auto = True
            if not args.case_id:
                p.error(f"could not auto-detect 7-digit case_id from folder "
                        f"{args.folder!r}; pass case_id explicitly")
                return 2

        print(f"=== Folder ingest: s3://{bucket}/{prefix} ===")
        suffix = " (auto-detected from folder name)" if case_id_was_auto else ""
        print(f"case_id: {args.case_id}{suffix}")
        print()

        kept, skipped = list_and_classify_folder(
            args.folder,
            profile_name=args.profile,
            region_name=args.region,
            allegation_pattern=args.allegation_pattern,
            medical_pattern=args.medical_pattern,
        )

        print(f"Found {len(kept) + len(skipped)} file(s) in folder. "
              f"Classification:")
        for spec in kept:
            print(f"  + {spec['role']:<20}  {spec['basename']}")
        for name in skipped:
            print(f"    {'(skip)':<20}  {name}")
        print()

        n_alleg = sum(1 for s in kept if s["role"] == "allegation_source")
        n_med   = sum(1 for s in kept if s["role"] == "medical_evidence")
        print(f"Will ingest: {len(kept)} files "
              f"({n_alleg} allegation_source + {n_med} medical_evidence)")
        print(f"Will skip:   {len(skipped)} files")
        print()

        if n_med == 0 and n_alleg == 0:
            print("ERROR: no files matched the classification patterns. "
                  "Check --allegation-pattern / --medical-pattern, or use "
                  "--pdfs to specify files explicitly.", file=sys.stderr)
            return 4
        if n_med == 0:
            print("WARNING: no medical_evidence files — matcher will have "
                  "no chart content to evaluate against.", file=sys.stderr)

        if args.dry_run:
            print("(--dry-run) stopping before ingest.")
            return 0

        pdfs = kept

    elif args.pdfs:
        pdfs = args.pdfs
    elif args.bucket and args.key:
        if not args.case_id:
            p.error("case_id is required in single-PDF mode")
            return 2
        pdfs = [{"bucket": args.bucket, "key": args.key, "role": "medical_evidence"}]
    else:
        p.error("Provide one of: positional <case_id> <bucket> <key> (single PDF), "
                "--pdfs <bucket>/<key>[:<role>] ... (multi-PDF), "
                "or --folder <s3-prefix> (auto-discover + classify).")
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
