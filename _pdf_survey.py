"""Survey a PDF: page count, dimensions, text content, image count per page.
Useful for deciding whether the PDF is OCRed, scanned-only, or mixed.

Usage:  python _pdf_survey.py path/to/file.pdf
"""
import sys
import fitz


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python _pdf_survey.py <pdf_path>", file=sys.stderr)
        return 2

    p = argv[1]
    doc = fitz.open(p)
    print(f"Pages: {doc.page_count}")
    print(f"Metadata: {doc.metadata}")
    print()
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        images = page.get_images()
        drawings = len(page.get_drawings())
        w, h = page.rect.width, page.rect.height
        preview = text[:120].replace("\n", " | ") if text else "(no embedded text)"
        print(f"p{i+1:>3}  {w:.0f}x{h:.0f}  text={len(text):>5}  imgs={len(images):>2}  draws={drawings:>3}  | {preview}")
        if i >= 50:
            print(f"... (stopping preview at 50, total {doc.page_count} pages)")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
