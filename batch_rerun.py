"""Re-process every case in _phi/ WITHOUT re-running Textract.

Pure-Python orchestrator. Runs from cmd, PowerShell, or any shell —
avoids the shell-syntax friction of the .ps1 version (which required
PowerShell for env vars, $env: syntax, etc.). Same behaviour: rechunk
from cached Textract, re-load to DB, re-run matcher with auto-publish.

Usage (from repo root):
    python batch_rerun.py                       # all cases under _phi/
    python batch_rerun.py 2010125 2218548       # just those
    python batch_rerun.py --skip-rechunk        # re-load + matcher only
    python batch_rerun.py --skip-matcher        # re-chunk + re-load only
    python batch_rerun.py --force               # re-run even if marker exists

3 stages per case (each optional via flags):
    1. rechunk_from_raw     re-chunk saved Textract JSON with latest
                            chunker + allegation extractor (no $)
    2. load_case_to_db      re-embed chunks + allegations into Postgres
                            (~$0.05/case in Bedrock embeddings)
    3. run.py --from-db     re-run matcher (~$2-6/case in Bedrock LLM)
                            Auto-publishes to S3 at end (HTML + PDF
                            companion) when OUTPUT_PUBLISH_BUCKET set.

Resumability: a case is skipped when
    output/<case>_EXPEDITESummary/0_<case>_EXPEDITESummary.html
already exists. Pass --force to override and re-process anyway.
Failures don't stop the batch — they're reported in the final summary.

Prereqs (any shell):
    aws sso login --profile user
    set DATABASE_URL=postgresql://<user>@<host>:5432/<db>      (cmd)
    set PGPASSWORD=<your-password>                              (cmd)
    set OUTPUT_PUBLISH_BUCKET=<outgoing-bucket>                 (cmd)
    set SHAREPOINT_PUBLISH_BASE_URL=https://...                 (cmd, optional)
or in PowerShell: $env:NAME = "value"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


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


def _discover_cases() -> list[str]:
    """Find case IDs from _phi/ subfolders that contain textract_raw.json."""
    phi = REPO_ROOT / "_phi"
    if not phi.exists():
        return []
    return sorted(
        d.name for d in phi.iterdir()
        if d.is_dir() and (d / "textract_raw.json").exists()
    )


def _run(cmd: list[str], stage_label: str) -> int:
    """Run a subprocess, streaming output to stdout. Returns exit code.

    No capture: live output lets the operator see Bedrock progress, see
    Postgres errors, abort with Ctrl-C, etc. Tee to a log via shell:
        python batch_rerun.py > batch.log 2>&1
    """
    print(f"  [{stage_label}] $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    except KeyboardInterrupt:
        print(f"\n  [{stage_label}] interrupted by user")
        raise


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("case_ids", nargs="*",
                   help="Case IDs to process (default: all in _phi/)")
    p.add_argument("--skip-rechunk", action="store_true",
                   help="Skip stage 1 (use existing chunks.json)")
    p.add_argument("--skip-load", action="store_true",
                   help="Skip stage 2 (DB already current)")
    p.add_argument("--skip-matcher", action="store_true",
                   help="Skip stage 3 (no matcher re-run)")
    p.add_argument("--force", action="store_true",
                   help="Re-process cases even if marker file exists")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "user"),
                   help="AWS profile (default: $AWS_PROFILE or 'user')")
    args = p.parse_args(argv[1:])

    # Required env vars (DB needed for stages 2+3; bucket optional)
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    if not os.environ.get("PGPASSWORD"):
        print("ERROR: PGPASSWORD not set", file=sys.stderr)
        return 1
    if not os.environ.get("OUTPUT_PUBLISH_BUCKET"):
        print("WARN: OUTPUT_PUBLISH_BUCKET not set; outputs will NOT auto-publish")

    # SSO pre-flight — stages 2+3 both call Bedrock
    if not (args.skip_load and args.skip_matcher):
        print("Verifying AWS SSO token...")
        if not _check_sso(args.profile):
            print(f"ERROR: AWS SSO token invalid. Run: "
                  f"aws sso login --profile {args.profile}", file=sys.stderr)
            return 2
        print("  SSO token OK")

    # Resolve which cases to process
    cases = args.case_ids or _discover_cases()
    if not cases:
        print("ERROR: no cases to process. Pass case IDs explicitly, or "
              "populate _phi/<case>/textract_raw.json first.", file=sys.stderr)
        return 3

    log_path = REPO_ROOT / f"batch_rerun_{datetime.now():%Y%m%d-%H%M%S}.log"
    print(f"\nRe-processing {len(cases)} case(s): {', '.join(cases)}")
    print(f"  redirect stdout to log if desired: "
          f"python batch_rerun.py > {log_path.name} 2>&1")

    start = time.time()
    ok: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    sep = "=" * 75
    py = sys.executable  # whichever python is running this script

    for i, case_id in enumerate(cases, 1):
        # Resumability check — skip if the case is already rendered
        marker = (REPO_ROOT / "output" / f"{case_id}_EXPEDITESummary"
                  / f"0_{case_id}_EXPEDITESummary.html")
        if marker.exists() and not args.force:
            print(f"\n[{i}/{len(cases)}] SKIP {case_id}  "
                  f"(already has {marker.relative_to(REPO_ROOT)}; "
                  f"pass --force to re-run)")
            skipped.append(case_id)
            continue

        print(f"\n{sep}")
        print(f"[{i}/{len(cases)}] RE-PROCESSING {case_id}")
        print(sep)

        # Stage 1: rechunk from cached Textract
        if not args.skip_rechunk:
            rc = _run(
                [py, "-c",
                 f"from pipeline.ingest_real import rechunk_from_raw; "
                 f"rechunk_from_raw('{case_id}')"],
                "1/3 rechunk")
            if rc != 0:
                print(f"  FAILED rechunk for {case_id}")
                failed.append(f"{case_id} (rechunk)")
                continue

        # Stage 2: re-load to DB with fresh embeddings
        if not args.skip_load:
            rc = _run([py, "db/load_case_to_db.py", case_id], "2/3 load")
            if rc != 0:
                print(f"  FAILED load for {case_id}")
                failed.append(f"{case_id} (load)")
                continue

        # Stage 3: re-run matcher (and auto-publish if env vars set)
        if not args.skip_matcher:
            rc = _run([py, "run.py", "--from-db", case_id], "3/3 matcher")
            if rc != 0:
                print(f"  FAILED matcher for {case_id}")
                failed.append(f"{case_id} (matcher)")
                # SSO can expire mid-batch — short-circuit if so
                if not _check_sso(args.profile):
                    print("ERROR: AWS SSO expired mid-batch. Stopping.",
                          file=sys.stderr)
                    break
                continue

        print(f"  OK {case_id}")
        ok.append(case_id)

    elapsed_min = (time.time() - start) / 60.0
    print(f"\n{sep}")
    print(f"BATCH COMPLETE in {elapsed_min:.1f} minutes")
    print(sep)
    print(f"Cases attempted:  {len(cases)}")
    print(f"Skipped ({len(skipped)}):     {', '.join(skipped) or '(none)'}")
    print(f"OK      ({len(ok)}):          {', '.join(ok) or '(none)'}")
    print(f"Failed  ({len(failed)}):     {', '.join(failed) or '(none)'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
