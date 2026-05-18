"""Generate an annotated copy of the source PDF with highlight rectangles at
the bbox of every cited chunk. Reviewers see the highlighted regions when they
click a citation in the HTML form.

Uses the bbox sidecar produced by pipeline.ingest_real (which carries Textract
WORD-geometry-derived bboxes per chunk). For each cited chunk_id, draws a
semi-transparent yellow rectangle at its bbox on every page the chunk appears.

The output PDF is itself a normal PDF — Chrome, Edge, Firefox, and Adobe Reader
all render the annotations when opened. No server, no JavaScript.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz


YELLOW = (1.0, 0.92, 0.0)


def annotate_pdf(
    source_pdf: Path,
    bbox_sidecar: Path,
    cited_chunk_ids: set[str],
    output_pdf: Path,
    *,
    color: tuple[float, float, float] = YELLOW,
    opacity: float = 0.35,
) -> tuple[Path, int]:
    """Create an annotated copy of source_pdf with highlight rectangles at the
    bboxes of every cited chunk.

    Args:
        source_pdf:      original PDF (read-only, not modified)
        bbox_sidecar:    _phi/<case_id>/chunks_with_bbox.json from ingest_real
        cited_chunk_ids: set of chunk_ids that appear as citations in any form
        output_pdf:      where to write the annotated copy
        color:           RGB 0-1 for the highlight box; default yellow
        opacity:         0-1 transparency for the fill; default 0.35

    Returns (output_path, number_of_annotations_drawn).
    """
    if not source_pdf.exists():
        raise FileNotFoundError(f"Source PDF not found: {source_pdf}")
    if not bbox_sidecar.exists():
        raise FileNotFoundError(f"Bbox sidecar not found: {bbox_sidecar}")

    bbox_data = json.loads(bbox_sidecar.read_text(encoding="utf-8"))
    bbox_by_id: dict[str, dict[str, list[float]]] = {
        c["chunk_id"]: c.get("bbox_by_page", {})
        for c in bbox_data["chunks"]
    }

    doc = fitz.open(source_pdf)
    annotation_count = 0
    missing_bbox: list[str] = []

    for chunk_id in cited_chunk_ids:
        bbox_by_page = bbox_by_id.get(chunk_id)
        if not bbox_by_page:
            missing_bbox.append(chunk_id)
            continue
        for page_key, bbox in bbox_by_page.items():
            page_num_1based = int(page_key)
            page_idx = page_num_1based - 1
            if page_idx < 0 or page_idx >= doc.page_count:
                continue

            page = doc[page_idx]
            # Textract bbox tuple: (left, top, right, bottom) normalized 0..1
            left, top, right, bottom = bbox
            rect = fitz.Rect(
                left * page.rect.width,
                top * page.rect.height,
                right * page.rect.width,
                bottom * page.rect.height,
            )

            # We use a "Square" rect annotation rather than the spec-correct
            # highlight annotation because (a) the source is a scanned PDF
            # with no underlying text layer, so highlight-as-underline doesn't
            # render meaningfully; (b) a filled rect is what reviewers expect
            # to see; (c) it renders consistently across Chrome/Edge/Adobe.
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=color, fill=color)
            annot.set_opacity(opacity)
            # Slight border so the box is visible even at low opacity
            annot.set_border(width=1.0)
            annot.update()
            annotation_count += 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf), garbage=4, deflate=True)
    doc.close()

    return output_pdf, annotation_count


def collect_cited_chunk_ids(leaf_results_by_listing: dict) -> set[str]:
    """Walk every listing's leaf_results dict and collect every chunk_id cited
    as evidence on any leaf. Returns the set we'd want to highlight."""
    cited: set[str] = set()
    for leaf_results in leaf_results_by_listing.values():
        for lr in leaf_results.values():
            for ev in lr.evidence:
                cited.add(ev.chunk_id)
    return cited
