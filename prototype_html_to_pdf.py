"""Prototype: convert a rendered case-summary HTML to PDF via WeasyPrint.

Standalone smoke test before wiring HTML->PDF into the publish flow. The
goal is to give reviewers a format SharePoint will render inline for
everyone (PDFs render inline regardless of NoScript / Custom Script
policy; HTMLs only render for site collection admins).

Usage:
    pip install weasyprint        # one-time, ~3 MB + bundled deps on Windows

    py prototype_html_to_pdf.py output/7777777_EXPEDITESummary/0_7777777_EXPEDITESummary.html
        # writes 0_7777777_EXPEDITESummary.pdf next to the HTML

    py prototype_html_to_pdf.py <html_path> --out <pdf_path>
        # explicit output path

    py prototype_html_to_pdf.py <html_path> --base-url <url>
        # base URL for resolving any remaining relative URLs in the HTML
        # (CSS / images). For SharePoint-rewritten HTML this rarely matters
        # since all links are already absolute. Default: file's parent dir.

What to look at in the rendered PDF:
    - Does the 3-column case-summary table render correctly?
    - Are citation links clickable (right-click "Open link" should work)?
    - Does the sticky left sidebar layout collapse sanely for print?
      (Sidebar is meaningless on paged media; we'll add @media print
      rules in the next iteration if it's ugly.)
    - Page count + file size — reported on exit.

If WeasyPrint installs cleanly and produces an acceptable PDF, the
follow-up commit adds --also-pdf to publish_output.py so every case
gets HTML + PDF uploaded side-by-side.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("html_path", help="Path to an existing rendered HTML file")
    p.add_argument("--out", default=None,
                   help="Output PDF path (default: same dir + .pdf suffix)")
    p.add_argument("--base-url", default=None,
                   help="Base URL for resolving relative refs in the HTML. "
                        "Defaults to the HTML file's parent directory as a "
                        "file:// URL so local CSS/images resolve.")
    args = p.parse_args(argv[1:])

    html_path = Path(args.html_path).resolve()
    if not html_path.exists():
        print(f"ERROR: {html_path} does not exist", file=sys.stderr)
        return 2
    if not html_path.is_file():
        print(f"ERROR: {html_path} is not a file", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else html_path.with_suffix(".pdf")
    out_path = out_path.resolve()

    # base_url tells WeasyPrint how to resolve relative URLs (CSS, images,
    # the citation .pdf links). For SharePoint-rewritten HTML those are
    # already absolute, but for a locally-rendered HTML we want relative
    # paths to resolve against the HTML's own folder.
    base_url = args.base_url or html_path.parent.as_uri() + "/"

    try:
        from weasyprint import HTML
    except ImportError:
        print("ERROR: WeasyPrint is not installed.", file=sys.stderr)
        print("  Install with: pip install weasyprint", file=sys.stderr)
        print("  (Recent versions ship bundled native deps on Windows; "
              "no MSYS2 required.)", file=sys.stderr)
        return 3

    print(f"Reading  {html_path}")
    html_text = html_path.read_text(encoding="utf-8")
    html_size = len(html_text.encode("utf-8"))
    print(f"  size: {html_size // 1024} KB ({html_size:,} bytes)")
    print(f"  base_url: {base_url}")

    print(f"\nRendering -> {out_path}")
    t0 = time.time()
    pdf_bytes = HTML(string=html_text, base_url=base_url).write_pdf()
    elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)
    pdf_size = len(pdf_bytes)

    # Best-effort page count — read the bytes for /Type /Page markers.
    # Not 100% accurate (could appear in content streams) but close enough
    # for a smoke test. A real implementation would use pypdf.
    page_marker_count = pdf_bytes.count(b"/Type /Page\n") + \
                        pdf_bytes.count(b"/Type/Page\n") + \
                        pdf_bytes.count(b"/Type /Page ") + \
                        pdf_bytes.count(b"/Type/Page ")

    print(f"\nDone in {elapsed:.2f}s")
    print(f"  PDF size:  {pdf_size // 1024} KB ({pdf_size:,} bytes)")
    print(f"  ~pages:    {page_marker_count} (rough count; open the PDF to verify)")
    print(f"\nNext: open {out_path} in a PDF viewer and check:")
    print(f"  - 3-column case-summary table renders")
    print(f"  - Citation links are clickable")
    print(f"  - No layout disasters from the sticky sidebar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
