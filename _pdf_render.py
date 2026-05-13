"""Render selected pages of a PDF to PNG. Outputs to <out_dir>/page_NN.png.
Useful for manual transcription when real OCR isn't yet available.

Usage:
    python _pdf_render.py <pdf_path> [out_dir] [pages]

    pages: comma-separated list of 1-indexed page numbers, or "every:N" to
           sample every Nth page. Defaults to "every:5".

Examples:
    python _pdf_render.py mypacket.pdf
    python _pdf_render.py mypacket.pdf _preview
    python _pdf_render.py mypacket.pdf _preview 1,5,12,17,23
    python _pdf_render.py mypacket.pdf _preview every:3
"""
import os
import sys
import fitz


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    src = argv[1]
    out_dir = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(src) or ".", "_preview")
    pages_spec = argv[3] if len(argv) > 3 else "every:5"

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(src)

    if pages_spec.startswith("every:"):
        n = int(pages_spec.split(":", 1)[1])
        pages_0idx = list(range(0, doc.page_count, n))
    else:
        pages_0idx = [int(p) - 1 for p in pages_spec.split(",")]

    zoom = 2.0  # 144 DPI is enough to read scanned chart text
    mat = fitz.Matrix(zoom, zoom)
    for pi in pages_0idx:
        if pi < 0 or pi >= doc.page_count:
            continue
        pix = doc[pi].get_pixmap(matrix=mat)
        path = os.path.join(out_dir, f"page_{pi+1:02d}.png")
        pix.save(path)
        print(f"wrote {path} ({pix.width}x{pix.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
