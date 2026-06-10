"""Upload a rendered case bundle to the outgoing S3 bucket.

Standalone companion to the auto-publish step in run.py — use when you've
manually fixed up output/<case>/, want to re-publish without re-running
the matcher, or are pushing a bundle that was rendered before the
OUTPUT_PUBLISH_BUCKET env var was configured.

Usage:
    set OUTPUT_PUBLISH_BUCKET=umasschan-forhealth-expedite-outgoing-data-nonprod
    py publish_output.py <case_id>                 # publish output/<case_id>/
    py publish_output.py <case_id> --dry-run       # show what would upload
    py publish_output.py <case_id> --bucket <name> # override OUTPUT_PUBLISH_BUCKET
    py publish_output.py --all                     # publish every case in output/

SharePoint URL baking — opt in via SHAREPOINT_PUBLISH_BASE_URL env var
(or --sharepoint-base-url) so citation links work in the SharePoint
web viewer. With the base URL set, every uploaded .html has its
relative <a href="*.pdf"> rewritten to absolute SharePoint URLs:
    set SHAREPOINT_PUBLISH_BASE_URL=https://tenant.sharepoint.com/sites/X/Y/Inbound%20from%20AWS/
The case's actual SharePoint folder is computed as
    <base><case_id>_EXPEDITESummary/
Override the suffix with --sharepoint-folder-suffix if your tenant lays
folders out differently. Without these flags the upload is byte-identical
to the local files (works for the local browser / download workflow).

Idempotent: re-running on the same case_id overwrites existing S3 objects
(same key = same content); cost is just the PUT requests.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make pipeline importable when invoked from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.output_publish import publish_to_s3   # noqa: E402


HERE = Path(__file__).parent
OUTPUT_ROOT = HERE / "output"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case_id", nargs="?",
                   help="case_id under output/ (omit if using --all)")
    p.add_argument("--all", action="store_true",
                   help="publish every case folder under output/")
    p.add_argument("--bucket",
                   default=os.environ.get("OUTPUT_PUBLISH_BUCKET"),
                   help="S3 bucket name (defaults to $OUTPUT_PUBLISH_BUCKET)")
    p.add_argument("--profile",
                   default=os.environ.get("AWS_PROFILE", "user"),
                   help="AWS profile (default: $AWS_PROFILE or 'user')")
    p.add_argument("--region",
                   default=os.environ.get("AWS_REGION", "us-east-1"),
                   help="AWS region (default: $AWS_REGION or 'us-east-1')")
    p.add_argument("--dry-run", action="store_true",
                   help="log what would be uploaded, don't actually upload")
    p.add_argument("--sharepoint-base-url",
                   default=os.environ.get("SHAREPOINT_PUBLISH_BASE_URL"),
                   help="If set, rewrite relative .pdf hrefs in uploaded HTML "
                        "to absolute SharePoint URLs (needed for citation "
                        "clicks in the SharePoint web viewer). Should be the "
                        "parent URL ending with '/', e.g. "
                        "https://tenant.sharepoint.com/.../Inbound%20from%20AWS/. "
                        "Defaults to $SHAREPOINT_PUBLISH_BASE_URL.")
    p.add_argument("--sharepoint-folder-suffix",
                   default="_EXPEDITESummary",
                   help="Suffix appended to case_id to form the SharePoint "
                        "sub-folder name. Default '_EXPEDITESummary'.")
    args = p.parse_args(argv[1:])

    if not args.bucket:
        print("ERROR: no target bucket. Set OUTPUT_PUBLISH_BUCKET in your "
              "environment or pass --bucket.", file=sys.stderr)
        return 2

    if args.all and args.case_id:
        print("ERROR: --all and <case_id> are mutually exclusive.", file=sys.stderr)
        return 2
    if not args.all and not args.case_id:
        p.print_help()
        return 2

    # Resolve the set of case folders to publish. Local folders are named
    # "<case_id>_EXPEDITESummary"; strip the suffix to recover the bare
    # case_id (used as the S3 prefix, matching what SharePoint already
    # has for inbound bundles).
    SUFFIX = "_EXPEDITESummary"
    if args.all:
        if not OUTPUT_ROOT.exists():
            print(f"No output/ directory at {OUTPUT_ROOT}", file=sys.stderr)
            return 1
        cases = sorted(
            p.name.removesuffix(SUFFIX) if p.name.endswith(SUFFIX) else p.name
            for p in OUTPUT_ROOT.iterdir() if p.is_dir()
        )
        if not cases:
            print(f"No case folders under {OUTPUT_ROOT}", file=sys.stderr)
            return 1
        print(f"Publishing {len(cases)} case folder(s) to s3://{args.bucket}/")
    else:
        cases = [args.case_id]

    total_keys = 0
    failed: list[str] = []
    for cid in cases:
        # Try the new "<case_id>_EXPEDITESummary" folder first; fall back to
        # the legacy "<case_id>" folder name for backwards compatibility
        # with cases rendered before the rename.
        out_dir = OUTPUT_ROOT / f"{cid}{SUFFIX}"
        if not out_dir.exists():
            out_dir = OUTPUT_ROOT / cid
        if not out_dir.exists():
            print(f"  skip {cid}: no {cid}{SUFFIX}/ or {cid}/ under {OUTPUT_ROOT}")
            failed.append(cid)
            continue
        print(f"\nCase {cid}:")
        try:
            keys = publish_to_s3(
                case_id=cid,
                out_dir=out_dir,
                bucket=args.bucket,
                profile_name=args.profile,
                region_name=args.region,
                dry_run=args.dry_run,
                sharepoint_base_url=args.sharepoint_base_url,
                sharepoint_folder_suffix=args.sharepoint_folder_suffix,
            )
            total_keys += len(keys)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed.append(cid)

    print(f"\nDone. {total_keys} object(s) "
          f"{'staged' if args.dry_run else 'uploaded'}. "
          f"{len(failed)} case(s) failed: {failed}" if failed
          else f"\nDone. {total_keys} object(s) "
          f"{'staged' if args.dry_run else 'uploaded'}.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
