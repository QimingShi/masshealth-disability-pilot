"""Prototype: convert a rendered case-summary HTML to PDF via headless Edge.

Standalone smoke test before wiring HTML->PDF into the publish flow. The
goal is to give reviewers a format SharePoint will render inline for
everyone (PDFs render inline regardless of NoScript / Custom Script
policy; HTMLs only render for site collection admins).

Why headless Edge and not WeasyPrint / Playwright?
    Edge is pre-installed on every modern Windows box (Win 10 build 1809+,
    all Win 11). It supports Chromium's --print-to-pdf flag natively. So
    zero extra installs, zero IT approvals, zero new dependencies for the
    PHI environment. Trade-off: less programmatic control than Playwright
    (no per-page hooks, no JS evaluation), but for our static HTML output
    that's not a limitation.

Usage:
    py prototype_html_to_pdf.py output/7777777_EXPEDITESummary/0_7777777_EXPEDITESummary.html
        # writes 0_7777777_EXPEDITESummary.pdf next to the HTML

    py prototype_html_to_pdf.py <html_path> --out <pdf_path>
        # explicit output path

    py prototype_html_to_pdf.py <html_path> --edge "C:\\path\\to\\msedge.exe"
        # override auto-detected Edge location

What to look at in the rendered PDF:
    - Does the 3-column case-summary table render correctly?
    - Are citation links clickable (right-click "Open link" should work)?
    - Does the sticky left sidebar layout collapse sanely for print?
      (Sidebar is meaningless on paged media; we'll add @media print
      rules in the next iteration if it's ugly.)
    - Page count + file size — reported on exit.

If the output looks acceptable, the follow-up commit adds --also-pdf to
publish_output.py so every case gets HTML + PDF uploaded side-by-side.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Standard install locations for Microsoft Edge on Windows. msedge.exe
# is in PATH on most modern installs but not all — so probe these too.
_EDGE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    # Edge Beta / Dev channels — last resort, usually not present
    r"C:\Program Files (x86)\Microsoft\Edge Beta\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge Dev\Application\msedge.exe",
)


def _find_edge(override: str | None) -> str | None:
    """Return absolute path to msedge.exe, or None if not found."""
    if override:
        if Path(override).is_file():
            return override
        print(f"WARN: --edge {override} not found; falling back to auto-detect",
              file=sys.stderr)
    on_path = shutil.which("msedge")
    if on_path:
        return on_path
    for candidate in _EDGE_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("html_path", help="Path to an existing rendered HTML file")
    p.add_argument("--out", default=None,
                   help="Output PDF path (default: same dir + .pdf suffix)")
    p.add_argument("--edge", default=None,
                   help="Override path to msedge.exe (default: auto-detect)")
    p.add_argument("--keep-header-footer", action="store_true",
                   help="Don't pass --no-pdf-header-footer (Edge will add "
                        "page numbers + URL footer; usually you don't want this)")
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

    edge_exe = _find_edge(args.edge)
    if not edge_exe:
        print("ERROR: Microsoft Edge not found.", file=sys.stderr)
        print("  Tried PATH (msedge) and standard install locations:",
              file=sys.stderr)
        for c in _EDGE_CANDIDATES:
            print(f"    {c}", file=sys.stderr)
        print("  Pass --edge <path-to-msedge.exe> to override.", file=sys.stderr)
        return 3

    # Edge's --print-to-pdf=<path> writes to <path>. Some Windows shells
    # mangle the equals-sign form; use the more portable "--flag" "<val>"
    # split. Input must be a URL — convert the file path to file:// with
    # proper percent-encoding (handles spaces in "Desktop\EXPEDITE (26)").
    input_url = html_path.as_uri()

    # Edge writes to %USERPROFILE% by default; use an isolated user-data-dir
    # so we don't fight with the user's own Edge profile (avoids races if
    # Edge is already running, and skips first-run / sign-in prompts).
    with tempfile.TemporaryDirectory(prefix="msedge-pdf-") as user_data_dir:
        cmd = [
            edge_exe,
            "--headless=new",     # newer headless mode (better fidelity post-2022)
            "--disable-gpu",
            f"--user-data-dir={user_data_dir}",
            f"--print-to-pdf={out_path}",
        ]
        if not args.keep_header_footer:
            cmd.append("--no-pdf-header-footer")
        cmd.append(input_url)

        print(f"Reading  {html_path}")
        html_size = html_path.stat().st_size
        print(f"  size: {html_size // 1024} KB ({html_size:,} bytes)")
        print(f"  input URL: {input_url}")
        print(f"\nEdge:    {edge_exe}")
        print(f"Output:  {out_path}")
        print(f"\nRunning headless Edge...")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=60)
        except subprocess.TimeoutExpired:
            print("ERROR: Edge timed out after 60 seconds", file=sys.stderr)
            return 4

        elapsed = time.time() - t0
        # Edge writes a couple of warnings to stderr even on success
        # (DevTools listening, GPU-related messages). Don't treat those
        # as failures — just check the return code and the output file.
        if result.returncode != 0:
            print(f"ERROR: Edge exited with code {result.returncode}",
                  file=sys.stderr)
            if result.stdout:
                print(f"--- stdout ---\n{result.stdout}", file=sys.stderr)
            if result.stderr:
                print(f"--- stderr ---\n{result.stderr}", file=sys.stderr)
            return 5

    if not out_path.exists():
        print(f"ERROR: Edge exited 0 but {out_path} was not created",
              file=sys.stderr)
        if result.stderr:
            print(f"--- stderr ---\n{result.stderr}", file=sys.stderr)
        return 6

    pdf_size = out_path.stat().st_size

    # Best-effort page count — look for /Type /Page markers in the PDF
    # bytes. Imprecise (matches in content streams will inflate the count)
    # but fine for a smoke test. A real implementation would use pypdf.
    pdf_bytes = out_path.read_bytes()
    page_marker_count = (pdf_bytes.count(b"/Type /Page\n") +
                         pdf_bytes.count(b"/Type/Page\n") +
                         pdf_bytes.count(b"/Type /Page ") +
                         pdf_bytes.count(b"/Type/Page "))

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
