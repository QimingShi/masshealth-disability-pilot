"""CLI driver for Textract ingestion.

Usage:
    python ingest.py <case_id> <s3_bucket> <s3_key> [--profile NAME] [--region REGION] [--no-pdf]

    python ingest.py --rechunk <case_id>
        Re-runs chunking from a previously saved Textract response, without
        re-running (or paying for) Textract.

Examples:
    # Ingest a packet from S3
    python ingest.py member-002 \
        umasschan-forhealth-expedite-incoming-data-nonprod \
        "REDACTED MEPERS DOCUMENT_Redacted 4.23.25 (1).pdf"

    # Then run the matcher on the produced chunks
    python run.py data/member-002/chunks.json
"""
import argparse
import sys
from pathlib import Path

from pipeline.ingest_real import ingest_packet, rechunk_from_raw


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Textract ingestion for the disability matcher pipeline.")
    p.add_argument("case_id", help="Case identifier (e.g. member-002). Outputs go to data/<case_id>/ and _phi/<case_id>/")
    p.add_argument("bucket", nargs="?", help="S3 bucket containing the PDF")
    p.add_argument("key", nargs="?", help="S3 object key (PDF filename)")
    p.add_argument("--profile", default="user", help="AWS profile name (default: user)")
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument("--no-pdf", action="store_true", help="Skip downloading PDF (citation links won't work)")
    p.add_argument("--rechunk", action="store_true", help="Re-chunk from saved Textract response (no API call)")
    args = p.parse_args(argv[1:])

    if args.rechunk:
        path = rechunk_from_raw(args.case_id)
        print(f"\nDone. Run the matcher on: {path}")
        return 0

    if not args.bucket or not args.key:
        p.error("bucket and key are required (unless --rechunk)")
        return 2

    path = ingest_packet(
        bucket=args.bucket,
        key=args.key,
        case_id=args.case_id,
        profile_name=args.profile,
        region_name=args.region,
        download_pdf=not args.no_pdf,
    )
    print(f"\nDone. Run the matcher on: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
