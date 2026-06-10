"""Full ingest of every case from the incoming S3 bucket.

Pure-Python orchestrator. Same flow as batch_ingest.ps1 but runs from
any shell (cmd, PowerShell, bash on WSL). Each case goes through the
full pipeline: download from S3 -> Textract OCR -> chunk -> embed ->
load to DB -> matcher -> auto-publish to outgoing S3 / SharePoint.

Usage:
    python batch_ingest.py                 # all cases in the folder list
    python batch_ingest.py --force         # re-run cases even if already done
    python batch_ingest.py 124264 7777777  # just those (must match leading
                                           #   digits of a folder name below)

Cost: ~$5-7 per case (Textract is the dominant cost). For 16 cases ~ $80-110
      + ~$2-6/case for the matcher = total ~$112-192.
Time: ~5-10 min per case sequentially -> ~1.5-2.5 hours total.

Resumability: a case is "done" when
    output/<case>_EXPEDITESummary/0_<case>_EXPEDITESummary.html
exists. Re-running after an SSO expiry / network blip just skips the
finished cases — only the remaining ones re-ingest. --force overrides.

Edit FOLDERS below to add/remove cases. Leading digits become the
case_id; the rest is the S3 prefix under INCOMING_BUCKET.

Prereqs:
    aws sso login --profile user
    set DATABASE_URL=postgresql://<user>@<host>:5432/<db>
    set PGPASSWORD=<your-password>
    set OUTPUT_PUBLISH_BUCKET=umasschan-forhealth-expedite-outgoing-data-nonprod
    set SHAREPOINT_PUBLISH_BASE_URL=https://...   (optional, enables PDF companion)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

INCOMING_BUCKET = "umasschan-forhealth-expedite-incoming-data-nonprod"

# Source folders under s3://INCOMING_BUCKET/. Case ID = leading digit run.
FOLDERS = [
    "124264 Med - 120/",
    "2010125 Med - 100/",
    "2011162 Psych-120/",
    "2173204 Med - 100/",
    "2181878 Psych- 100/",
    "2182555 Psych- 100/",
    "2198279- 231/",
    "2199880 Psych- 120/",
    "2203790 Med - 120/",
    "2204762 - Med 120/",
    "2208661 Med - 120/",
    "2209253 Psych- 120/",
    "2218548 Psych- 120/",
    "2220672 -230/",
    "2231212- 210/",
    "7777777 Redacted/",
]


def _check_sso(profile: str) -> bool:
    """Return True if AWS SSO token is valid for the given profile."""
    try:
        r = subprocess.run(
            ["aws", "sts", "get-caller-identity",
             "--profile", profile, "--output", "text"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _case_id_for(folder: str) -> str | None:
    """Extract leading digit run from a folder name. None if no match."""
    m = re.match(r"^(\d+)", folder)
    return m.group(1) if m else None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("case_ids", nargs="*",
                   help="Only process these case IDs (default: all in FOLDERS)")
    p.add_argument("--force", action="store_true",
                   help="Re-run cases even if marker file exists")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "user"),
                   help="AWS profile (default: $AWS_PROFILE or 'user')")
    args = p.parse_args(argv[1:])

    # Required env vars
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    if not os.environ.get("PGPASSWORD"):
        print("ERROR: PGPASSWORD not set", file=sys.stderr)
        return 1
    if not os.environ.get("OUTPUT_PUBLISH_BUCKET"):
        print("WARN: OUTPUT_PUBLISH_BUCKET not set; outputs will NOT auto-publish")

    # SSO pre-flight — Textract + Bedrock both need a live token
    print("Verifying AWS SSO token...")
    if not _check_sso(args.profile):
        print(f"ERROR: AWS SSO token invalid. Run: "
              f"aws sso login --profile {args.profile}", file=sys.stderr)
        return 2
    print("  SSO token OK")

    # Filter folders by --case_ids if given
    selected = FOLDERS
    if args.case_ids:
        selected = [f for f in FOLDERS
                    if _case_id_for(f) in set(args.case_ids)]
        unknown = set(args.case_ids) - {_case_id_for(f) for f in selected}
        if unknown:
            print(f"ERROR: unknown case ID(s): {sorted(unknown)}",
                  file=sys.stderr)
            print(f"  Available: {[_case_id_for(f) for f in FOLDERS]}",
                  file=sys.stderr)
            return 3

    log_path = REPO_ROOT / f"batch_ingest_{datetime.now():%Y%m%d-%H%M%S}.log"
    print(f"\nIngesting {len(selected)} case(s) from s3://{INCOMING_BUCKET}/")
    print(f"  redirect stdout to log if desired: "
          f"python batch_ingest.py > {log_path.name} 2>&1")

    start = time.time()
    ok: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    sep = "=" * 75
    py = sys.executable

    for i, folder in enumerate(selected, 1):
        case_id = _case_id_for(folder)
        if not case_id:
            print(f"SKIP: couldn't parse case_id from '{folder}'")
            failed.append(folder)
            continue

        # Resumability check
        marker = (REPO_ROOT / "output" / f"{case_id}_EXPEDITESummary"
                  / f"0_{case_id}_EXPEDITESummary.html")
        if marker.exists() and not args.force:
            print(f"\n[{i}/{len(selected)}] SKIP {case_id}  "
                  f"(already has {marker.relative_to(REPO_ROOT)}; "
                  f"pass --force to re-run)")
            skipped.append(case_id)
            continue

        print(f"\n{sep}")
        print(f"[{i}/{len(selected)}] INGESTING {case_id}  (from '{folder}')")
        print(sep)

        cmd = [py, "db/ingest_s3_to_db.py", case_id,
               "--folder", f"{INCOMING_BUCKET}/{folder}"]
        print(f"  $ {' '.join(cmd)}")
        try:
            rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        except KeyboardInterrupt:
            print("\n  interrupted by user")
            raise

        if rc != 0:
            print(f"  FAILED {case_id} (exit {rc})")
            failed.append(f"{case_id} ({folder})")
            # SSO can expire mid-batch — short-circuit if so
            if not _check_sso(args.profile):
                print("ERROR: AWS SSO expired mid-batch. Stopping. "
                      f"Run: aws sso login --profile {args.profile}",
                      file=sys.stderr)
                break
        else:
            print(f"  OK {case_id}")
            ok.append(case_id)

    elapsed_min = (time.time() - start) / 60.0
    print(f"\n{sep}")
    print(f"BATCH COMPLETE in {elapsed_min:.1f} minutes")
    print(sep)
    print(f"Total cases in list:   {len(selected)}")
    print(f"Skipped ({len(skipped)}) — already done: "
          f"{', '.join(skipped) or '(none)'}")
    print(f"OK      ({len(ok)}):                     "
          f"{', '.join(ok) or '(none)'}")
    print(f"Failed  ({len(failed)}):                     "
          f"{', '.join(failed) or '(none)'}")
    if failed:
        print(f"\nTo retry just the failures: re-run python batch_ingest.py — "
              f"failed cases will be retried automatically (no marker file).")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
