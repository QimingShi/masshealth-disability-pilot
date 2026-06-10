"""Upload a self-contained case bundle (output/<case_id>/) to S3.

Reviewers consume the rendered forms via SharePoint or other downstream tools
that read from the outgoing S3 bucket. This module mirrors the local layout
to S3 with one key per file under <bucket>/<case_id>/.

The bundle is already self-contained — relative-URL HTML citations resolve
against source.pdf / source_annotated.pdf sitting next to them — so a
plain "aws s3 sync s3://<bucket>/<case_id> ./local-folder" on the receiving
side produces a working folder reviewers can open in any local browser.

SharePoint complication: modern SharePoint Online tenants disable inline
JavaScript execution (NoScript policy) when serving HTML from document
libraries. The Forms/AllItems.aspx wrapper also breaks relative-URL
resolution. To make citation links work directly in the SharePoint
viewer, this module can optionally rewrite every relative .pdf href
in uploaded HTMLs to an absolute SharePoint URL (see `sharepoint_base_url`).
With that flag set, the served HTML has fully-qualified URLs baked in —
no script execution required.

Auto-publish is opt-in by env var (OUTPUT_PUBLISH_BUCKET); the standalone
publish_output.py script invokes the same function for one-off re-publishes.
"""
from __future__ import annotations

import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import quote


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


# Matches the opening of an <a> tag whose href is a relative .pdf path
# (optionally with #fragment or ?query). Captures:
#   group(1) — attrs before href=
#   group(2) — the href value
#   group(3) — attrs after the closing href=" quote, up to and excluding >
# Case-insensitive; DOTALL so multi-line tag splits don't trip us up.
_A_PDF_HREF_RE = re.compile(
    r'<a\b([^>]*?)\bhref="([^"]+\.pdf(?:[?#][^"]*)?)"([^>]*)>',
    re.IGNORECASE | re.DOTALL,
)

# Already-absolute href prefixes — never rewrite these.
_ABSOLUTE_PREFIXES = ("http://", "https://", "//", "mailto:", "tel:",
                      "javascript:", "data:", "#")


def _rewrite_html_for_sharepoint(html: str, folder_url: str) -> tuple[str, int]:
    """Rewrite every relative .pdf href in <a> tags to absolute folder_url + path.

    folder_url must end with '/'. Also injects target="_blank" rel="noopener"
    on any rewritten link that doesn't already have a target attribute, so
    citations open in a new tab (SharePoint otherwise navigates the wrapper
    page itself).

    Returns (rewritten_html, n_links_rewritten). On zero matches the input
    is returned unchanged. Idempotent: hrefs already starting with http(s)://
    or other absolute schemes are skipped, so re-running on the same HTML
    is a no-op.
    """
    if not folder_url.endswith("/"):
        folder_url = folder_url + "/"
    n = 0

    def _replace(match: re.Match) -> str:
        nonlocal n
        before, href, after = match.group(1), match.group(2), match.group(3)
        if href.lower().startswith(_ABSOLUTE_PREFIXES):
            return match.group(0)

        # Split off fragment / query before URL-encoding the path
        if "#" in href:
            path, sep, rest = href.partition("#")
        elif "?" in href:
            path, sep, rest = href.partition("?")
        else:
            path, sep, rest = href, "", ""
        new_href = folder_url + quote(path, safe="/") + sep + rest

        # Preserve other attributes, inject target/rel if absent
        attrs = (before + " " + after).strip()
        if "target=" not in attrs.lower():
            attrs = (attrs + ' target="_blank" rel="noopener"').strip()
        n += 1
        return f'<a href="{new_href}" {attrs}>' if attrs else f'<a href="{new_href}">'

    rewritten = _A_PDF_HREF_RE.sub(_replace, html)
    return rewritten, n


def publish_to_s3(
    case_id: str,
    out_dir: Path,
    bucket: str,
    *,
    profile_name: str | None = None,
    region_name:  str | None = None,
    dry_run: bool = False,
    sharepoint_base_url: str | None = None,
    sharepoint_folder_suffix: str = "_EXPEDITESummary",
) -> list[str]:
    """Upload every file in out_dir to s3://<bucket>/<case_id>/<filename>.

    Args:
        case_id: human-readable case id; becomes the top-level S3 prefix
        out_dir: local source folder (typically output/<case_id>/)
        bucket: target S3 bucket name (no s3:// prefix)
        profile_name: AWS profile (default: $AWS_PROFILE or 'user')
        region_name:  AWS region  (default: $AWS_REGION or 'us-east-1')
        dry_run: log what would happen, but don't actually upload
        sharepoint_base_url: if set, every .html file is rewritten in-flight
            so its relative .pdf hrefs become absolute SharePoint URLs:
            <sharepoint_base_url><case_id><sharepoint_folder_suffix>/<pdf>.
            Required for citation clicks to work in SharePoint's web viewer
            (SharePoint disables inline JS and breaks relative resolution).
            Pass the parent URL ending with '/'. Example:
              https://tenant.sharepoint.com/sites/X/Y/Inbound%20from%20AWS/
            Local copies on disk are not modified.
        sharepoint_folder_suffix: appended to case_id to form the SharePoint
            sub-folder name. Default '_EXPEDITESummary' matches the layout
            run.py writes locally. Ignored when sharepoint_base_url is None.

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

    # Pre-compute the SharePoint folder for this case (where PDFs live).
    sp_folder_url: str | None = None
    if sharepoint_base_url:
        base = sharepoint_base_url
        if not base.endswith("/"):
            base = base + "/"
        sp_folder_url = f"{base}{case_id}{sharepoint_folder_suffix}/"
        print(f"      publish: SharePoint URL bake enabled  "
              f"base={sp_folder_url}")

    uploaded: list[str] = []
    total_bytes = 0
    total_links_rewritten = 0
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

        # HTML rewriting path: read, transform, upload as bytes via put_object.
        # Avoids touching the local file. Falls back to plain upload if
        # rewriting yields zero matches (don't waste a put_object call).
        is_html = path.suffix.lower() in (".html", ".htm")
        rewritten_body: bytes | None = None
        if is_html and sp_folder_url:
            try:
                html = path.read_text(encoding="utf-8")
                new_html, n_links = _rewrite_html_for_sharepoint(html, sp_folder_url)
                if n_links > 0:
                    rewritten_body = new_html.encode("utf-8")
                    total_links_rewritten += n_links
                    print(f"      publish: {rel}  rewrote {n_links} citation "
                          f"href(s) to absolute SharePoint URLs")
            except (OSError, UnicodeDecodeError) as e:
                print(f"      publish: WARN couldn't rewrite {rel} ({e}); "
                      f"uploading verbatim")

        if dry_run:
            ct_label = ct or "(no content-type)"
            marker = " [SP-rewritten]" if rewritten_body is not None else ""
            print(f"      publish: [dry-run] s3://{bucket}/{key}  "
                  f"({size // 1024} KB, {ct_label}){marker}")
        elif rewritten_body is not None:
            put_kwargs = {"Bucket": bucket, "Key": key,
                          "Body": rewritten_body}
            if ct:
                put_kwargs["ContentType"] = ct
            s3.put_object(**put_kwargs)
            size = len(rewritten_body)
            print(f"      publish: {rel}  ->  s3://{bucket}/{key}  "
                  f"({size // 1024} KB, SP-rewritten)")
        else:
            s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
            print(f"      publish: {rel}  ->  s3://{bucket}/{key}  "
                  f"({size // 1024} KB)")

        uploaded.append(key)
        total_bytes += size

    if sp_folder_url and total_links_rewritten > 0:
        print(f"      publish: rewrote {total_links_rewritten} total "
              f"citation href(s) across HTML files")

    if uploaded and not dry_run:
        print(f"      publish: uploaded {len(uploaded)} file(s) "
              f"({total_bytes // 1024} KB total) to s3://{bucket}/{case_id}/")
    elif uploaded and dry_run:
        print(f"      publish: would upload {len(uploaded)} file(s) "
              f"({total_bytes // 1024} KB total)")
    else:
        print(f"      publish: no eligible files found under {out_dir}")
    return uploaded
