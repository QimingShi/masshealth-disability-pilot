"""Chunk a Textract AnalyzeDocument response into pipeline chunks.

Template for the OCR ingestion path. Consumes the JSON from Textract
`AnalyzeDocument` (call with FeatureTypes=['LAYOUT','TABLES','FORMS']) and
produces chunks ready for embedding, structured-extraction, and DB insert.

Not yet wired into run.py — that still reads pre-chunked data/chunks.json.
When Textract is available, replace the chunks-loading step with:

    response = textract.analyze_document(...)  # or fetch from S3 / async result
    chunks = chunk_by_layout(response, doc_id="onc-note-1", doc_meta={...})

Scope of this file:
  - Layout-driven chunking only (one sub-document in, list of chunks out).
  - Sub-document boundary detection within a packet lives in a separate
    function (sketch comments at the bottom).
  - Structured extraction (ICD codes, lab table parsing) is downstream of
    this; this layer just produces clean chunks with metadata.

What Textract gives us (block types we care about):
  LAYOUT_TITLE              - document title at top of page
  LAYOUT_SECTION_HEADER     - section divider; starts a new chunk
  LAYOUT_TEXT               - paragraph body
  LAYOUT_LIST               - bulleted / numbered list
  LAYOUT_TABLE              - the table-as-a-region (cells live in TABLE)
  LAYOUT_KEY_VALUE          - form fields region
  LAYOUT_PAGE_NUMBER        - page footer; skip
  LAYOUT_FIGURE             - images, logos; skip
  TABLE / CELL              - structured table data
  LINE / WORD               - actual text with bboxes and confidence
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re
from typing import Any


# ---------- types ----------

Block = dict[str, Any]  # raw Textract block JSON


@dataclass
class BBox:
    """Union bounding box across one or more pages."""
    by_page: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    # {page: (left, top, right, bottom)} in Textract's normalized 0-1 coords

    def add_page(self, page: int, box: dict):
        """Merge a Textract Geometry.BoundingBox into the current per-page span."""
        left, top = box["Left"], box["Top"]
        right, bottom = left + box["Width"], top + box["Height"]
        if page in self.by_page:
            l, t, r, b = self.by_page[page]
            self.by_page[page] = (min(l, left), min(t, top), max(r, right), max(b, bottom))
        else:
            self.by_page[page] = (left, top, right, bottom)


@dataclass
class ChunkRecord:
    """One produced chunk, richer than the pipeline's existing Chunk dataclass.
    Use `to_pipeline_chunk()` to drop into the existing data flow, or write
    directly to DB using the full record."""
    chunk_id: str
    doc_id: str
    section: str
    text: str
    page_start: int
    page_end: int
    # rich metadata not yet consumed by the matcher but useful for the reviewer UI
    bbox: BBox = field(default_factory=BBox)
    ocr_confidence: float = 0.0       # avg word confidence in the chunk, 0-100
    layout_block_type: str = ""        # "LAYOUT_TEXT", "LAYOUT_TABLE", ...
    is_table: bool = False
    source_block_ids: list[str] = field(default_factory=list)
    # forwarded doc-level metadata (so chunks are self-contained for citation)
    doc_type: str | None = None
    doc_title: str | None = None
    encounter_date: str | None = None


# ---------- section normalization ----------

# Map of detected-header-text -> canonical section name. Keys are lowercased,
# punctuation-stripped. Extend this list as you see new chart styles.
SECTION_NORMALIZER = {
    "hpi": "HPI",
    "history of present illness": "HPI",
    "interval history": "Interval History",
    "ros": "Review of Systems",
    "review of systems": "Review of Systems",
    "past medical history": "Past Medical History",
    "pmh": "Past Medical History",
    "medications": "Medications",
    "current outpatient medications": "Medications",
    "allergies": "Allergies",
    "social history": "Social History",
    "family history": "Family History",
    "vitals": "Vitals",
    "exam": "Physical Exam",
    "physical exam": "Physical Exam",
    "assessment and plan": "Assessment and Plan",
    "impression": "Impression",
    "findings": "Findings",
    "visit diagnoses": "Visit Diagnoses",
    "diagnoses": "Visit Diagnoses",
    "problem list": "Problem List",
    "labs": "Labs",
    "results": "Labs",
    "linked episodes": "Linked Episodes",
    "medication changes": "Medication Changes",
    "orders placed": "Orders",
    "pathology report": "Pathology",
    "ct abdomen and pelvis": "CT Abdomen/Pelvis",
}

# Sections that benefit from splitting numbered findings into their own chunks.
# (Reviewer wants to cite "Impression #3" precisely, not the whole impression block.)
SECTIONS_SPLIT_NUMBERED = {"Impression", "Findings", "Assessment and Plan"}

# Sections that should be split into one chunk per list item (per ICD/per med).
# (Each line is independently citeable and queryable.)
SECTIONS_SPLIT_LIST = {"Visit Diagnoses", "Problem List", "Medications"}


def _normalize_section_name(raw_header: str) -> str:
    key = re.sub(r"[^\w\s/]", "", raw_header).strip().lower()
    key = re.sub(r"\s+", " ", key)
    return SECTION_NORMALIZER.get(key, raw_header.strip())


# ---------- text extraction from blocks ----------

def _index_blocks(blocks: list[Block]) -> dict[str, Block]:
    return {b["Id"]: b for b in blocks}


def _children(block: Block, kind: str = "CHILD") -> list[str]:
    for rel in block.get("Relationships") or []:
        if rel.get("Type") == kind:
            return rel.get("Ids", [])
    return []


def _word_blocks_under(block: Block, idx: dict[str, Block]) -> list[Block]:
    """Walk down to all WORD blocks reachable from this block, IN READING ORDER.

    Layout blocks point to LINEs; LINEs point to WORDs; TABLEs point to CELLs
    which point to WORDs. We use recursion (or a stack with reversed children)
    rather than a naive LIFO stack — a naive stack reverses traversal order,
    which produces backwards text. That bug would silently break table cells
    and any fallback text assembly."""
    out: list[Block] = []

    def visit(b: Block):
        if b.get("BlockType") == "WORD":
            out.append(b)
            return
        for cid in _children(b):
            cb = idx.get(cid)
            if cb is not None:
                visit(cb)

    visit(block)
    return out


def _text_of(block: Block, idx: dict[str, Block]) -> str:
    """Plaintext under this block, words joined with single spaces, lines
    separated by newlines. Preserves rough reading order from Textract."""
    line_ids = []
    table_ids = []
    for cid in _children(block):
        cb = idx.get(cid)
        if cb is None:
            continue
        t = cb.get("BlockType")
        if t == "LINE":
            line_ids.append(cid)
        elif t == "TABLE":
            table_ids.append(cid)
        elif t == "LAYOUT_TEXT" or (t and t.startswith("LAYOUT_")):
            # Nested layouts (rare but happens with multi-column docs)
            line_ids.append(cid)

    parts = []
    for lid in line_ids:
        line = idx[lid]
        if line.get("BlockType") == "LINE" and line.get("Text"):
            parts.append(line["Text"])
        else:
            # Fallback: walk to words and join
            words = _word_blocks_under(line, idx)
            if words:
                parts.append(" ".join(w.get("Text", "") for w in words))

    if table_ids:
        for tid in table_ids:
            parts.append(_render_table(idx[tid], idx))

    return "\n".join(p for p in parts if p.strip())


def _render_table(table_block: Block, idx: dict[str, Block]) -> str:
    """Render a TABLE block as a plaintext markdown-like table. Preserves row
    structure so downstream lab-parsing can split on lines."""
    # Group CELL blocks by (row, col).
    cells_by_rc: dict[tuple[int, int], list[Block]] = defaultdict(list)
    for cid in _children(table_block):
        cb = idx.get(cid)
        if cb and cb.get("BlockType") == "CELL":
            r, c = cb["RowIndex"], cb["ColumnIndex"]
            cells_by_rc[(r, c)].append(cb)

    if not cells_by_rc:
        return ""

    n_rows = max(r for r, _ in cells_by_rc)
    n_cols = max(c for _, c in cells_by_rc)

    rows: list[list[str]] = []
    for r in range(1, n_rows + 1):
        row: list[str] = []
        for c in range(1, n_cols + 1):
            cell_blocks = cells_by_rc.get((r, c), [])
            cell_text = ""
            if cell_blocks:
                words = _word_blocks_under(cell_blocks[0], idx)
                cell_text = " ".join(w.get("Text", "") for w in words).strip()
            row.append(cell_text)
        rows.append(row)

    # Render as pipe-separated table — easy for the embedder and human-readable.
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


def _avg_confidence(blocks: list[Block]) -> float:
    confs = [b.get("Confidence", 0.0) for b in blocks if b.get("Confidence") is not None]
    return sum(confs) / len(confs) if confs else 0.0


def _add_geometry_to_bbox(block: Block, bbox: BBox):
    geom = block.get("Geometry") or {}
    bb = geom.get("BoundingBox")
    if bb and "Page" in block:
        bbox.add_page(block["Page"], bb)


# ---------- main chunker ----------

def chunk_by_layout(
    response: dict,
    doc_id: str,
    doc_meta: dict | None = None,
) -> list[ChunkRecord]:
    """Chunk one sub-document's Textract response into ChunkRecords.

    Args:
        response: raw Textract AnalyzeDocument JSON for this sub-document
        doc_id: stable identifier (e.g. "onc-note-1")
        doc_meta: forwarded onto each chunk (encounter_date, doc_type, doc_title)
    """
    doc_meta = doc_meta or {}
    blocks: list[Block] = response.get("Blocks", [])
    idx = _index_blocks(blocks)

    # Pull layout blocks in reading order (page asc, then top-y asc).
    layout_blocks = [
        b for b in blocks
        if b.get("BlockType", "").startswith("LAYOUT_")
        and b["BlockType"] not in {"LAYOUT_PAGE_NUMBER", "LAYOUT_FIGURE"}
    ]
    layout_blocks.sort(key=lambda b: (
        b.get("Page", 0),
        b.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0.0),
    ))

    chunks: list[ChunkRecord] = []
    current_section = "Header"  # before the first SECTION_HEADER, content is header-like
    section_idx_counter: dict[str, int] = defaultdict(int)
    pending_blocks: list[Block] = []   # blocks accumulating under current section

    def flush_pending():
        """Emit accumulated blocks as one or more chunks for the current section."""
        if not pending_blocks:
            return
        section = current_section
        text = "\n".join(t for b in pending_blocks if (t := _text_of(b, idx)))
        if not text.strip():
            pending_blocks.clear()
            return

        # Decide whether to split this section into sub-chunks.
        sub_texts: list[str]
        if section in SECTIONS_SPLIT_NUMBERED and _has_numbered_items(text):
            sub_texts = _split_numbered(text)
        elif section in SECTIONS_SPLIT_LIST:
            sub_texts = _split_list_lines(text)
        else:
            sub_texts = [text]

        for sub_text in sub_texts:
            words = []
            for b in pending_blocks:
                words.extend(_word_blocks_under(b, idx))
            bbox = BBox()
            for b in pending_blocks:
                _add_geometry_to_bbox(b, bbox)
            page_start = min((b.get("Page", 1) for b in pending_blocks), default=1)
            page_end = max((b.get("Page", 1) for b in pending_blocks), default=page_start)

            chunk_idx = section_idx_counter[section]
            section_idx_counter[section] += 1
            chunk_id = f"{doc_id}:p{page_start}:{_safe(section)}:{chunk_idx}"

            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=section,
                text=sub_text.strip(),
                page_start=page_start,
                page_end=page_end,
                bbox=bbox,
                ocr_confidence=_avg_confidence(words),
                layout_block_type=pending_blocks[0].get("BlockType", ""),
                is_table=any(b.get("BlockType") == "LAYOUT_TABLE" for b in pending_blocks),
                source_block_ids=[b["Id"] for b in pending_blocks],
                doc_type=doc_meta.get("doc_type"),
                doc_title=doc_meta.get("doc_title"),
                encounter_date=doc_meta.get("encounter_date"),
            ))
        pending_blocks.clear()

    for b in layout_blocks:
        btype = b["BlockType"]

        if btype == "LAYOUT_SECTION_HEADER" or btype == "LAYOUT_TITLE":
            # Section boundary — emit what we've been collecting, then start a
            # new section using this block's text as the header.
            flush_pending()
            header_text = _text_of(b, idx)
            current_section = _normalize_section_name(header_text) or current_section

        elif btype == "LAYOUT_TABLE":
            # A table is its own chunk: flush prior text, emit table alone,
            # then continue accumulating under the same section.
            flush_pending()
            pending_blocks.append(b)
            flush_pending()

        else:
            # LAYOUT_TEXT, LAYOUT_LIST, LAYOUT_KEY_VALUE -> accumulate under section
            pending_blocks.append(b)

    flush_pending()
    return chunks


# ---------- splitters for sections that warrant finer granularity ----------

NUMBERED_RE = re.compile(r"(?ms)^\s*(\d+)\.\s+")


def _has_numbered_items(text: str) -> bool:
    # At least two numbered items to be worth splitting
    return len(NUMBERED_RE.findall(text)) >= 2


def _split_numbered(text: str) -> list[str]:
    """Split on lines that start with '1. ', '2. ', etc. Preserves any leading
    preamble (text before the first number) as the first chunk."""
    parts = NUMBERED_RE.split(text)
    if len(parts) <= 1:
        return [text]
    out: list[str] = []
    preamble = parts[0].strip()
    if preamble:
        out.append(preamble)
    # After split: ['preamble', '1', 'body1', '2', 'body2', ...]
    for i in range(1, len(parts), 2):
        num = parts[i]
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            out.append(f"{num}. {body}")
    return out


def _split_list_lines(text: str) -> list[str]:
    """Split a list-style section into one chunk per non-empty line.
    Filters out the "Visit Diagnoses:" label line if it stands alone."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Drop a header-style label line ("Visit Diagnoses:" or "Primary:") if it's first.
    out = []
    for ln in lines:
        if ln.endswith(":") and len(ln) < 30 and not out:
            continue
        out.append(ln)
    return out or [text]


# ---------- helpers ----------

def _safe(s: str) -> str:
    """Make a section name safe for use in a chunk_id (no spaces/punctuation)."""
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "Section"


# ---------- adapter for the existing pipeline ----------

def to_pipeline_chunks(records: list[ChunkRecord]) -> list[dict]:
    """Convert rich ChunkRecord -> the dict shape that data/chunks.json uses,
    so the existing loader and matcher can consume Textract output unchanged.
    Drops bbox/confidence/source_block_ids — those go in the DB but aren't
    needed for the matching pipeline."""
    return [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "section": r.section,
            "text": r.text,
            "page_start": r.page_start,
            "page_end": r.page_end,
        }
        for r in records
    ]


# ---------- sub-document boundary detection (sketch only) ----------
#
# Real-world packets bundle multiple sub-documents. Before calling
# chunk_by_layout on each, you have to detect the boundaries. Two reliable
# signals on the packets we've seen:
#
#   1. Page footers like "1/9", "4/12", "12/12" — the numerator resets when a
#      new sub-document starts. Textract LAYOUT_PAGE_NUMBER blocks contain the
#      footer text; parse with re.match(r"(\d+)/(\d+)$", text). When the
#      numerator drops or the denominator changes, you crossed a boundary.
#
#   2. Institution/title headers re-appearing — "MASSACHUSETTS GENERAL
#      HOSPITAL", "After Visit Summary", etc. LAYOUT_TITLE on a new page is a
#      reliable boundary signal even without footers.
#
# Suggested approach: scan pages in order; for each page extract its
# LAYOUT_PAGE_NUMBER (if any) and LAYOUT_TITLE. Start a new sub-document
# whenever (a) the page-footer numerator resets to 1, or (b) the previous
# page's footer was the last of its sub-doc ("M/M"), or (c) a LAYOUT_TITLE
# matching a known institution header appears at the top of the page.
# Then call chunk_by_layout once per sub-document, slicing the Blocks list
# by Page.
#
# This function is not implemented here — keep it deterministic, no LLM, and
# test it against your real packets when Textract output is available.


# ---------- demo / smoke test ----------

if __name__ == "__main__":
    # Minimal synthetic input so the file is exercisable without real Textract.
    fake_response = {
        "Blocks": [
            # Section: HPI
            {"Id": "h1", "BlockType": "LAYOUT_SECTION_HEADER", "Page": 1,
             "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.10, "Width": 0.5, "Height": 0.02}},
             "Relationships": [{"Type": "CHILD", "Ids": ["h1_line"]}]},
            {"Id": "h1_line", "BlockType": "LINE", "Text": "HPI", "Confidence": 99.0, "Page": 1,
             "Relationships": [{"Type": "CHILD", "Ids": ["h1_w"]}]},
            {"Id": "h1_w", "BlockType": "WORD", "Text": "HPI", "Confidence": 99.0, "Page": 1},

            {"Id": "t1", "BlockType": "LAYOUT_TEXT", "Page": 1,
             "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.13, "Width": 0.8, "Height": 0.05}},
             "Relationships": [{"Type": "CHILD", "Ids": ["t1_line"]}]},
            {"Id": "t1_line", "BlockType": "LINE",
             "Text": "Patient with metastatic CRC on protocol 24-471.",
             "Confidence": 97.5, "Page": 1,
             "Relationships": [{"Type": "CHILD", "Ids": []}]},

            # Section: Impression (with numbered findings, should split into 2 chunks)
            {"Id": "h2", "BlockType": "LAYOUT_SECTION_HEADER", "Page": 1,
             "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.25, "Width": 0.5, "Height": 0.02}},
             "Relationships": [{"Type": "CHILD", "Ids": ["h2_line"]}]},
            {"Id": "h2_line", "BlockType": "LINE", "Text": "Impression", "Confidence": 99.0, "Page": 1,
             "Relationships": [{"Type": "CHILD", "Ids": []}]},

            {"Id": "t2", "BlockType": "LAYOUT_TEXT", "Page": 1,
             "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.28, "Width": 0.8, "Height": 0.10}},
             "Relationships": [{"Type": "CHILD", "Ids": ["t2_l1", "t2_l2"]}]},
            {"Id": "t2_l1", "BlockType": "LINE",
             "Text": "1. Liver dome metastasis, decreased conspicuity.",
             "Confidence": 96.0, "Page": 1, "Relationships": [{"Type": "CHILD", "Ids": []}]},
            {"Id": "t2_l2", "BlockType": "LINE",
             "Text": "2. Stable rectosigmoid mass, 2.7 cm.",
             "Confidence": 96.0, "Page": 1, "Relationships": [{"Type": "CHILD", "Ids": []}]},
        ]
    }

    records = chunk_by_layout(
        fake_response,
        doc_id="demo-doc",
        doc_meta={"doc_type": "imaging_report", "encounter_date": "2026-02-11"},
    )
    import json
    print(json.dumps(
        [{"chunk_id": r.chunk_id, "section": r.section, "text": r.text,
          "page": (r.page_start, r.page_end),
          "ocr_confidence": round(r.ocr_confidence, 1),
          "is_table": r.is_table} for r in records],
        indent=2,
    ))
