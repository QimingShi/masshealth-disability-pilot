"""Upload a self-contained case bundle (output/<case_id>/) to S3.

Reviewers consume the rendered forms via SharePoint or other downstream tools
that read from the outgoing S3 bucket. This module mirrors the local layout
to S3 with one key per file under <bucket>/<case_id>/.

The bundle is already self-contained — relative-URL HTML citations resolve
against source.pdf / source_annotated.pdf sitting next to them — so a
plain "aws s3 sync s3://<bucket>/<case_id> ./local-folder" on the receiving
side produces a working folder reviewers can open in any local browser.

Auto-publish is opt-in by env var (OUTPUT_PUBLISH_BUCKET); the standalone
publish_output.py script invokes the same function for one-off re-publishes.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path


# Files we skip — build-time artifacts, hidden dotfiles, anything not for
# the reviewer. The reconstructed bbox sidecar (_chunks_with_bbox.json)
# should already be deleted by run.py after annotation, but exclude it
# defensively in case the cleanup didn't happen (e.g. annotation step
# crashed before the deletion).
def _should_skip(name: str) -> bool:
    return (
        name.startswith(".")
        or name.startswith("_")
        or name.endswith(".tmp")
        or name.endswith(".lock")
    )


# Content types Office / browser viewers need to render properly. mimetypes
# covers most cases but is incomplete for docx out of the box on some
# platforms — guarantee the right type for our known artifact families.
_KNOWN_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".md":   "text/markdown; charset=utf-8",
    ".pdf":  "application/pdf",
    ".docx": ("application/vnd.openxmlformats-officedocument."
              "wordprocessingml.document"),
    ".json": "application/json",
    ".txt":  "text/plain; charset=utf-8",
}


def _content_type_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _KNOWN_CONTENT_TYPES:
        return _KNOWN_CONTENT_TYPES[ext]
    guess, _ = mimetypes.guess_type(str(path))
    return guess


def publish_to_s3(
    case_id: str,
    out_dir: Path,
    bucket: str,
    *,
    profile_name: str | None = None,
    region_name:  str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Upload every file in out_dir to s3://<bucket>/<case_id>/<filename>.

    Args:
        case_id: human-readable case id; becomes the top-level S3 prefix
        out_dir: local source folder (typically output/<case_id>/)
        bucket: target S3 bucket name (no s3:// prefix)
        profile_name: AWS profile (default: $AWS_PROFILE or 'user')
        region_name:  AWS region  (default: $AWS_REGION or 'us-east-1')
        dry_run: log what would happen, but don't actually upload

    Returns:
        List of S3 keys that were uploaded (or would be, in dry-run).
    """
    if not out_dir.exists():
        print(f"      publish: out_dir {out_dir} does not exist; nothing to upload")
        return []

    try:
        import boto3
    except ImportError:
        print("      publish: boto3 not installed; cannot upload")
        return []

    profile = profile_name or os.environ.get("AWS_PROFILE", "user")
    region  = region_name  or os.environ.get("AWS_REGION", "us-east-1")
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3", region_name=region)

    uploaded: list[str] = []
    total_bytes = 0
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip(path.name):
            continue

        # Preserve any sub-directory structure within out_dir, although in
        # practice run.py writes everything flat under out_dir.
        rel = path.relative_to(out_dir)
        key = f"{case_id}/{rel.as_posix()}"

        size = path.stat().st_size
        ct   = _content_type_for(path)
        extra = {"ContentType": ct} if ct else {}

        if dry_run:
            ct_label = ct or "(no content-type)"
            print(f"      publish: [dry-run] s3://{bucket}/{key}  "
                  f"({size // 1024} KB, {ct_label})")
        else:
            s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
            print(f"      publish: {rel}  ->  s3://{bucket}/{key}  "
                  f"({size // 1024} KB)")

        uploaded.append(key)
        total_bytes += size

    if uploaded and not dry_run:
        print(f"      publish: uploaded {len(uploaded)} file(s) "
              f"({total_bytes // 1024} KB total) to s3://{bucket}/{case_id}/")
    elif uploaded and dry_run:
        print(f"      publish: would upload {len(uploaded)} file(s) "
              f"({total_bytes // 1024} KB total)")
    else:
        print(f"      publish: no eligible files found under {out_dir}")
    return uploaded
