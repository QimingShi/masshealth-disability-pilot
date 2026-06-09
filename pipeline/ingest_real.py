"""Real Textract ingestion: PDF in S3 -> chunks.json for the pipeline.

Replaces the manual hand-transcription path. Uses Textract async
AnalyzeDocument with LAYOUT/TABLES/FORMS, paginates results, detects
sub-document boundaries from page-footer patterns, runs chunk_by_layout
per sub-doc, and writes chunks.json that the rest of the pipeline consumes
unchanged.

Outputs land in:
  data/<case_id>/chunks.json          (tracked-shape; minimal metadata)
  _phi/<case_id>/textract_raw.json    (full Textract response — PHI, gitignored)
  _phi/<case_id>/source.pdf           (downloaded copy for HTML citation links)
  _phi/<case_id>/chunks_with_bbox.json (chunks with bboxes, for future
                                        annotated-PDF highlight generation)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import boto3
import fitz

from .ingest_textract import (
    Block,
    BBox,
    ChunkRecord,
    chunk_by_layout,
    to_pipeline_chunks,
)


# =============================================================================
# Textract API: start / poll / paginate
# =============================================================================

def start_analysis(bucket: str, key: str, *, session=None) -> str:
    """Kick off async Textract analysis with LAYOUT/TABLES/FORMS. Returns JobId."""
    session = session or boto3.Session()
    textract = session.client("textract")
    job = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["LAYOUT", "TABLES", "FORMS"],
    )
    return job["JobId"]


def wait_for_analysis(job_id: str, *, session=None, poll_seconds: int = 5) -> str:
    """Poll until job finishes. Returns final status: 'SUCCEEDED' or 'FAILED'.
    Prints progress so the user can see something is happening."""
    session = session or boto3.Session()
    textract = session.client("textract")
    while True:
        result = textract.get_document_analysis(JobId=job_id, MaxResults=1)
        status = result["JobStatus"]
        if status in ("SUCCEEDED", "FAILED"):
            return status
        print(f"  status: {status}  (sleeping {poll_seconds}s)")
        time.sleep(poll_seconds)


def fetch_all_blocks(job_id: str, *, session=None) -> list[Block]:
    """Page through GetDocumentAnalysis results and concatenate all blocks.
    Textract returns blocks in batches; without NextToken handling you lose
    everything past the first page of results."""
    session = session or boto3.Session()
    textract = session.client("textract")
    blocks: list[Block] = []
    next_token = None
    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token
        result = textract.get_document_analysis(**kwargs)
        blocks.extend(result.get("Blocks", []))
        next_token = result.get("NextToken")
        if not next_token:
            break
    return blocks


# =============================================================================
# Sub-document boundary detection
# =============================================================================
# Packets bundle multiple sub-documents stapled together. Reliable signals:
#   1. Page-footer "N/M" pattern (numerator resets when a new sub-doc starts)
#   2. LAYOUT_TITLE re-appearance at the top of a page
# This implementation uses (1) primarily and falls back to (2) if footers
# are missing.

# Footer patterns we recognize. The packet uses several variants:
#   "Page 1 of 4"    "Page 1 of 1"    "1 / 4"    "-2-"    bare integers ("8")
# Most reliable: explicit "Page N of M". Bare integers are typically packet-level
# numbering (the whole bundle's page count, NOT per-sub-doc) so we ignore them.
_FOOTER_PAGE_OF_RE = re.compile(r"\bpage\s*(\d+)\s*of\s*(\d+)", re.IGNORECASE)
_FOOTER_NUM_SLASH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def _children(block: Block) -> list[str]:
    for rel in block.get("Relationships") or []:
        if rel.get("Type") == "CHILD":
            return rel.get("Ids", [])
    return []


def _layout_block_text(block: Block, idx: dict[str, Block]) -> str:
    """Get correct-order text from a LAYOUT_* block by collecting child LINE.Text
    values. Far more reliable than walking to WORDs ourselves — Textract
    pre-assembles LINE.Text in reading order."""
    parts: list[str] = []
    for cid in _children(block):
        cb = idx.get(cid)
        if cb and cb.get("BlockType") == "LINE" and cb.get("Text"):
            parts.append(cb["Text"])
    return " ".join(parts)


def _parse_page_of(text: str) -> tuple[int, int] | None:
    """Extract (current_page, total_pages) from text like 'Page 2 of 4' or '2/4'.
    Returns None if no match or denom is 0."""
    if not text:
        return None
    m = _FOOTER_PAGE_OF_RE.search(text)
    if not m:
        m = _FOOTER_NUM_SLASH_RE.search(text)
    if not m:
        return None
    cur, total = int(m.group(1)), int(m.group(2))
    if total == 0:
        return None
    return cur, total


def _page_summary(blocks: list[Block], idx: dict[str, Block]) -> dict[int, dict]:
    """Per-page summary: LAYOUT_TITLE text, LAYOUT_HEADER text, LAYOUT_FOOTER text,
    and any parsed 'Page N of M' footer. Used as the basis for sub-doc detection."""
    summary: dict[int, dict] = {}
    for b in blocks:
        bt = b.get("BlockType", "")
        p = b.get("Page", 0)
        if not p:
            continue
        s = summary.setdefault(p, {"titles": [], "headers": [], "footers": [], "page_of": None})
        if bt == "LAYOUT_TITLE":
            s["titles"].append(_layout_block_text(b, idx))
        elif bt == "LAYOUT_HEADER":
            s["headers"].append(_layout_block_text(b, idx))
        elif bt == "LAYOUT_FOOTER":
            s["footers"].append(_layout_block_text(b, idx))
        elif bt == "LAYOUT_PAGE_NUMBER":
            txt = _layout_block_text(b, idx)
            parsed = _parse_page_of(txt)
            if parsed and not s["page_of"]:
                s["page_of"] = parsed
    # Also check footers for "Page N of M" patterns (the packet sometimes
    # encodes them as LAYOUT_FOOTER, not LAYOUT_PAGE_NUMBER).
    for p, s in summary.items():
        if not s["page_of"]:
            for f in s["footers"]:
                parsed = _parse_page_of(f)
                if parsed:
                    s["page_of"] = parsed
                    break
    return summary


def _title_changed(prev: str | None, curr: str | None) -> bool:
    """Heuristic: is the current page's title meaningfully different from the
    prev page's effective title? Returns True when we believe the title indicates
    a new sub-document."""
    if not curr:
        return False
    if not prev:
        return True  # First title we've seen
    # Exact match -> same sub-doc
    if prev.strip().lower() == curr.strip().lower():
        return False
    # Prefix overlap -> same form continuing across pages
    a, b = prev.strip().lower(), curr.strip().lower()
    if a.startswith(b) or b.startswith(a):
        return False
    # Token overlap >= 60% suggests same sub-doc
    a_tokens = set(re.findall(r"\w+", a))
    b_tokens = set(re.findall(r"\w+", b))
    if a_tokens and b_tokens:
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
        if overlap >= 0.6:
            return False
    return True


def detect_sub_documents(blocks: list[Block]) -> list[dict]:
    """Group pages into sub-documents using LAYOUT_TITLE as the primary signal,
    with 'Page 1 of M' footers as a secondary marker.

    Algorithm:
      - Build per-page summary (titles, footers, parsed page-of-M).
      - Walk pages in order. Start a new sub-doc when:
        * Current page has a 'Page 1 of M' footer (very strong signal), OR
        * Current page's title is meaningfully different from the active title
          (carried forward across page-without-title gaps).
      - Pages without a title inherit the prior sub-doc.

    Returns list of {doc_id, page_start, page_end, blocks, derived_title}.
    """
    idx = {b["Id"]: b for b in blocks}
    max_page = max((b.get("Page", 0) for b in blocks), default=1)
    summary = _page_summary(blocks, idx)

    boundaries: list[int] = [1]
    active_title: str | None = None  # last title we saw

    for p in range(2, max_page + 1):
        s = summary.get(p, {})
        page_of = s.get("page_of")
        titles = s.get("titles", [])
        curr_title = titles[0] if titles else None

        new_subdoc = False

        # Strong signal #1: "Page 1 of M" footer
        if page_of and page_of[0] == 1:
            new_subdoc = True

        # Strong signal #2: title changed
        if not new_subdoc and _title_changed(active_title, curr_title):
            new_subdoc = True

        if new_subdoc:
            boundaries.append(p)
            active_title = curr_title or active_title
        else:
            if curr_title:
                active_title = curr_title

    boundaries.append(max_page + 1)  # sentinel for slicing

    out: list[dict] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] - 1
        if end < start:
            continue
        sd_blocks = [b for b in blocks if start <= b.get("Page", 0) <= end]

        # Pick a representative title from this range (first non-empty title)
        derived_title = ""
        for pp in range(start, end + 1):
            titles = summary.get(pp, {}).get("titles", [])
            if titles and titles[0]:
                derived_title = titles[0]
                break

        out.append({
            "doc_id": f"doc-{i+1:02d}",
            "page_start": start,
            "page_end": end,
            "blocks": sd_blocks,
            "derived_title": derived_title,
        })
    return out


# =============================================================================
# Doc-level metadata extraction (best-effort, rule-based)
# =============================================================================
# Rule-based extraction of doc_type, doc_title, encounter_date from each
# sub-document's header region. Production system would tune these heuristics.

_DATE_RE = re.compile(r"(?:Date of Encounter|Encounter Date|Collection Time|Service Date)[:\s]+(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:\s+\d{1,2}:\d{2})?)", re.IGNORECASE)
_DATE_FALLBACK = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")

_DOC_TYPE_PATTERNS = [
    # Disability application packet forms (MEPERS-style)
    (r"\bmemorandum\b", "memorandum"),
    (r"intake summary form|disability retirement intake", "intake_form"),
    (r"application for disability retirement", "application_form"),
    (r"application checklist", "application_checklist"),
    (r"consent form authorizing release", "consent_release_of_info"),
    (r"consent form designating", "consent_designate_representative"),
    (r"\bconsent form\b", "consent_form"),
    (r"authorization.*release.*healthcare", "consent_release_healthcare"),
    (r"frequently asked questions|faqs?\b", "faqs"),
    (r"new application interview|application interview", "applicant_interview"),
    (r"employer interview", "employer_interview"),
    (r"healthcare provider'?s request", "provider_request"),
    (r"continuing health insurance", "health_insurance_notice"),
    # Medical records
    (r"primary care office note", "primary_care_note"),
    (r"physical examination|physical exam", "physical_exam"),
    (r"oncology.*(?:progress|follow.?up).*note", "oncology_progress_note"),
    (r"after.visit.summary|visit.bundle", "visit_bundle"),
    (r"pathology.*report", "pathology_report"),
    (r"\* ?final report ?\*|final report", "imaging_report"),
    (r"ct\s+(?:abdomen|chest|head|pelvis)|computed tomography", "ct_imaging_report"),
    (r"mri\b|magnetic resonance", "mri_imaging_report"),
    (r"x.?ray|radiograph", "xray_report"),
    (r"mammogram|breast imaging", "mammogram"),
    (r"ecg|ekg|electrocardiogram", "ecg_report"),
    (r"laboratory|lab\s+result", "lab_report"),
    (r"discharge\s+summary", "discharge_summary"),
    (r"history\s+and\s+physical|h&p", "history_and_physical"),
    (r"operative\s+(?:note|report)", "operative_note"),
    (r"echocardiogram", "echocardiogram"),
    (r"emergency\s+department|ed\s+visit", "ed_visit"),
    # Institution-letterhead fallbacks for medical records when no explicit type marker
    (r"northern light health", "medical_record_northern_light"),
    (r"epic systems|mychart", "medical_record_epic"),
    (r"cerner|powerchart", "medical_record_cerner"),
]

# EPIC-style structured marker in medical-record headers: "Document Type: X"
# Lazy match up to known terminator keywords so we don't gobble adjacent fields.
_DOC_TYPE_EXPLICIT_RE = re.compile(
    r"Document\s+Type\s*:\s*(.{3,80}?)"
    r"\s*(?:,|\||\n|"
    r"\s+(?:Service\s+Date|Result\s+status|Template|Performed|Verified|"
    r"Encounter|Author|Date\s+of|Provider|Patient)\b|$)",
    re.IGNORECASE,
)


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"\W+", "_", text.lower())).strip("_") or "unknown"


def _extract_doc_metadata(
    sd_blocks: list[Block],
    idx: dict[str, Block],
    derived_title: str = "",
) -> dict:
    """Best-effort doc_type / doc_title / encounter_date from a sub-doc's
    header region. Uses correct-order text via _layout_block_text + LINE.Text."""
    if not sd_blocks:
        return {"doc_type": "unknown", "doc_title": derived_title or "Untitled", "encounter_date": None}

    first_page = min((b.get("Page", 99999) for b in sd_blocks), default=0)

    # Collect first-page LAYOUT_TITLE / LAYOUT_HEADER text
    title_texts: list[str] = []
    header_texts: list[str] = []
    for b in sd_blocks:
        if b.get("Page") != first_page:
            continue
        bt = b.get("BlockType")
        if bt == "LAYOUT_TITLE":
            t = _layout_block_text(b, idx)
            if t:
                title_texts.append(t)
        elif bt == "LAYOUT_HEADER":
            t = _layout_block_text(b, idx)
            if t:
                header_texts.append(t)

    title_blob = " | ".join(title_texts)
    header_blob = " | ".join(header_texts)
    full_blob = f"{title_blob} | {header_blob}"[:3000]

    # 1) Explicit EPIC-style "Document Type: X" wins if present
    doc_type = "unknown"
    m = _DOC_TYPE_EXPLICIT_RE.search(header_blob)
    if m:
        doc_type = _slug(m.group(1))
    else:
        for pattern, type_name in _DOC_TYPE_PATTERNS:
            if re.search(pattern, full_blob, re.IGNORECASE):
                doc_type = type_name
                break

    # 2) Title: prefer derived_title (from sub-doc detection) > first LAYOUT_TITLE
    doc_title = derived_title or (title_texts[0] if title_texts else "")
    if not doc_title and header_texts:
        doc_title = header_texts[0]
    doc_title = (doc_title or "Untitled sub-document")[:240]

    # 3) Encounter date: search LINE.Text on first 2 pages of the sub-doc
    pages_sorted = sorted({b.get("Page", 0) for b in sd_blocks})
    search_pages = set(pages_sorted[:2])
    page_text = " ".join(
        b["Text"]
        for b in sd_blocks
        if b.get("BlockType") == "LINE" and b.get("Text") and b.get("Page") in search_pages
    )[:6000]
    date_match = _DATE_RE.search(page_text) or _DATE_FALLBACK.search(page_text)
    encounter_date = date_match.group(1) if date_match else None

    return {
        "doc_type": doc_type,
        "doc_title": doc_title,
        "encounter_date": encounter_date,
    }


# =============================================================================
# Allegation auto-extraction (best-effort, rule-based)
# =============================================================================
# Pulls a starter set of allegations from chart sections (chief complaint /
# visit diagnoses / PMH) AND from disability-supplement form patterns (reason
# for visit, conditions claimed, why I cannot work). All extractions pass a
# garbage filter that rejects column headers, section labels, timestamps, etc.
# Review allegations manually before relying on them for matching.

_ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?\b")

# Allegations that are obviously form labels / boilerplate / OCR fragments.
# Centralized so the same stoplist applies to every extraction path.
_ALLEGATION_GARBAGE_WORDS = frozenset({
    # Generic form labels / column headers
    "date", "diagnosis", "diagnoses", "provider", "providers", "author",
    "patient", "status", "time", "specimen", "reason", "type", "year",
    "name", "signature", "description", "address", "phone", "test",
    "yes", "no", "n/a", "none", "n / a", "not applicable", "n.a.",
    "history", "results", "result", "value", "ref range", "reference",
    "encounter", "visit", "service", "department", "facility", "unit",
    "section", "part", "page", "form", "blank", "tbd", "tba", "see above",
    "see below", "as needed", "prn", "unknown", "see hpi", "see ros",
    "n/k", "nkda", "none documented", "no known",
    # Chart section names (not allegations even if extractor sees them
    # without their trailing colon)
    "past medical history", "past surgical history", "surgical history",
    "family history", "social history", "review of systems",
    "history of present illness", "present illness", "hpi", "ros",
    "physical exam", "physical examination", "exam", "assessment",
    "assessment and plan", "impression", "plan", "vital signs", "vitals",
    "allergies", "medications", "current medications", "chief complaint",
    "linked episodes", "medication changes", "visit diagnoses",
    "orders", "orders placed", "interval history", "problem list",
    "procedure/surgical history", "initiating author",
    "health maintenance", "health status",
})

# Regex patterns that flag garbage entries even if the word stoplist misses them.
_ALLEGATION_GARBAGE_PATTERNS = [
    re.compile(r"^received\s+\d", re.IGNORECASE),       # "Received 3/10/2026..."
    re.compile(r"\bdes[0-9a-f]{8,}", re.IGNORECASE),     # DES tracking codes
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+\d",
               re.IGNORECASE),                            # "Mar. 10, 2026..."
    re.compile(r"^\d+\s*(am|pm|edt|est|cst|pst|utc)\b", re.IGNORECASE),
    re.compile(r"^no\.?\s*\d{3,}\b", re.IGNORECASE),      # fax form numbers
    re.compile(r"^p\.\s*\d+\s*/", re.IGNORECASE),          # "P.001/006" fax marks
    re.compile(r"^\(?fax\)?\b", re.IGNORECASE),
    re.compile(r"^mads-?\w+_\d", re.IGNORECASE),           # MassHealth form codes
    re.compile(r"^\W*$"),                                  # punctuation/whitespace only
    re.compile(r"^\d+(\.\d+)?\s*$"),                       # bare numbers
]


def _is_garbage_allegation(text: str) -> bool:
    """Reject form labels, column headers, timestamps, OCR noise that the
    extractor sometimes picks up as 'allegations'. Centralized so every
    extraction code path uses the same rules."""
    t = text.strip()
    if len(t) < 4 or len(t) > 200:
        return True
    if t.endswith(":"):
        return True
    # Bare label / single-word column header
    low = t.lower().rstrip(",.;:")
    if low in _ALLEGATION_GARBAGE_WORDS:
        return True
    # Pattern blocklist
    for pat in _ALLEGATION_GARBAGE_PATTERNS:
        if pat.search(t):
            return True
    # Must contain at least one alphabetic character (filters timestamps, IDs)
    if not re.search(r"[A-Za-z]{3,}", t):
        return True
    # Filter "all caps no vowels" which is usually a header acronym
    if t.isupper() and not re.search(r"[AEIOUaeiou]", t):
        return True
    return False


# Supplement-form patterns: where the member states their claimed conditions.
# Covers MassHealth Adult Disability Supplement, SSA-3368, and similar forms.
_SUPPLEMENT_REASON_RE = re.compile(
    r"(?:reason\s+for\s+visit|diagnoses?\s+i\s+am\s+claiming|"
    r"conditions?\s+i\s+am\s+applying\s+for|conditions?\s+being\s+claimed|"
    r"why\s+i\s+cannot\s+work|disabling\s+conditions?|"
    r"please\s+list\s+(?:all\s+)?conditions?|complaints?\s+leading\s+to)"
    r"\s*[:\-]?\s*([^\n]{4,200})",
    re.IGNORECASE,
)

# Generic "condition" hint phrases. Restricted to phrases that strongly imply
# the member is making a claim about a specific condition. We deliberately
# DON'T include bare "history of" here — that's a section name far more often
# than it's an allegation phrase.
_FOLLOWING_CONDITION_RE = re.compile(
    r"(?:complications?\s+following\s+(?:a\s+|an\s+)?|"
    r"status\s+post\s+|recently\s+diagnosed\s+with\s+|"
    r"currently\s+being\s+treated\s+for\s+|"
    r"applying\s+(?:for\s+)?(?:disability\s+(?:benefits?\s+)?)?(?:due\s+to|because\s+of)\s+|"
    r"unable\s+to\s+work\s+(?:due\s+to|because\s+of)\s+)"
    r"([^.\n]{4,150})",
    re.IGNORECASE,
)


def _is_supplement_section(section: str, text: str) -> bool:
    """Does this chunk look like it's part of a disability supplement / SSA-3368?"""
    s = section.lower()
    t = text.lower()[:400]
    return (
        "supplement" in s
        or "part 1" in s or "part 2" in s or "part 6" in s
        or "disability supplement" in t
        or "ssa-3368" in t
        or "function report" in s
    )


# Header-row keyword sets that identify the two supplement tables.
# We match on column-header substrings (Textract preserves the printed
# header text in the first row of a TABLE block).
_PART1_HEADER_KEYWORDS = (
    "list your medical",
    "describe the symptoms",
    "health problems",
)
_PART2_HEADER_KEYWORDS = (
    "reason for visit",
    "name of medical",
    "name of medical and mental health providers",
)


def _split_table_row(line: str) -> list[str]:
    """Split a Textract-rendered table row on the pipe separator."""
    return [cell.strip() for cell in line.split("|")]


def _extract_supplement_table_allegations(table_text: str) -> list[tuple[str, str]]:
    """Parse a pipe-separated table chunk; pull allegations from PART 1 / PART 2.

    Returns a list of (allegation_text, source_tag) tuples where source_tag is:
      - "supplement_part1"  → patient's listed health problem (Part 1 col 1)
      - "supplement_part2"  → reason for provider visit (Part 2 col 2)

    Tagging Part 1 vs Part 2 separately lets downstream filters (e.g. the
    no-listings fallback report) include only the patient's diagnoses
    section without the provider-reason entries that tend to be terse and
    less diagnostically useful (e.g. "Kidneys", "Cancer Center").

    Empty list if the table doesn't look like a known supplement section.
    """
    lines = [ln for ln in table_text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return []

    header_lc = lines[0].lower()

    is_part1 = any(kw in header_lc for kw in _PART1_HEADER_KEYWORDS)
    is_part2 = any(kw in header_lc for kw in _PART2_HEADER_KEYWORDS)
    if not (is_part1 or is_part2):
        return []

    out: list[tuple[str, str]] = []
    for line in lines[1:]:
        cells = _split_table_row(line)
        if not cells:
            continue
        if is_part1:
            if cells and cells[0]:
                out.append((cells[0], "supplement_part1"))
        elif is_part2:
            if len(cells) >= 2 and cells[1]:
                out.append((cells[1], "supplement_part2"))
    return out


def auto_extract_allegations(chunks: list[dict]) -> list[dict]:
    """Best-effort allegation extraction. Combines chart-style patterns
    (chief complaint, visit diagnoses, PMH) with supplement-form patterns
    (reason for visit, conditions claimed, MassHealth Part 1/Part 2 tables).
    All candidate allegations pass through _is_garbage_allegation() before
    being kept."""
    allegations: list[dict] = []
    seen: set[str] = set()

    def add(text: str, source: str, chunk_id: str):
        text = text.strip().rstrip(",.;")
        if _is_garbage_allegation(text):
            return
        key = re.sub(r"\W+", " ", text).strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        allegations.append({
            "text": text,
            "source": source,
            "source_chunk_id": chunk_id,
        })

    for c in chunks:
        section = c.get("section", "")
        section_lc = section.lower()
        text = c.get("text", "")

        # ---- MassHealth Adult Disability Supplement: PART 1 / PART 2 tables ----
        # Textract renders TABLE blocks as pipe-separated lines (see
        # ingest_textract._render_table). The supplement form has two
        # multi-row tables we care about:
        #
        #   PART 1: column 1 is the patient's listed diagnoses.
        #     header keywords: "list your medical", "health problems"
        #   PART 2: column 2 is the reason for each provider visit.
        #     header keywords: "reason for visit", "name of medical"
        #
        # For PART 1 we pull column 1 from every body row.
        # For PART 2 we pull column 2.
        # Pre-printed example rows (e.g. "Depression / Very tired all the
        # time. April 2010 / None") will be included — they're filtered by
        # _is_garbage_allegation only if they happen to match a known
        # garbage pattern. Treating them as real allegations costs nothing
        # in practice: if the chart has matching evidence the listing was
        # already going to come up; if not, Claude returns 'insufficient'.
        if c.get("is_table") and text:
            allegs_from_table = _extract_supplement_table_allegations(text)
            for alleg_text, source_tag in allegs_from_table:
                add(alleg_text, source_tag, c["chunk_id"])

        # ---- Supplement-form patterns (high precision) ----
        # SKIP this regex on table chunks. _SUPPLEMENT_REASON_RE captures
        # everything after "reason for visit" up to the next newline; in a
        # Part 2 table chunk the literal "Reason for visit" appears in the
        # COLUMN HEADER line, so the regex captures the rest of the header
        # text ("| Was this visit in the past year?") as a "claim". The
        # per-row table extractor above already handles supplement tables
        # via column-aware extraction.
        if _is_supplement_section(section, text) and not c.get("is_table"):
            for m in _SUPPLEMENT_REASON_RE.finditer(text):
                add(m.group(1), "supplement_form", c["chunk_id"])

        # ---- Chief complaint / chart header diagnosis line ----
        if any(k in section_lc for k in ("header", "chief", "complaint", "hpi")):
            for m in re.finditer(
                r"(?i)(?:chief\s+complaint|diagnosis/chief\s+complaint|diagnosis)\s*[:\-]\s*([^.\n]{4,150})",
                text,
            ):
                add(m.group(1), "chief_complaint", c["chunk_id"])

        # ---- Visit Diagnoses: one allegation per non-trivial line ----
        if "diagnos" in section_lc:
            for line in text.split("\n"):
                cleaned = re.sub(
                    r"^\s*(visit\s+diagnoses|primary|secondary|active|other)\s*[:\-]?\s*",
                    "", line, flags=re.IGNORECASE,
                )
                # Strip ICD-10 codes
                cleaned = _ICD_RE.sub("", cleaned).strip().rstrip(",.;:")
                add(cleaned, "visit_diagnoses", c["chunk_id"])

        # ---- PMH: bulleted past medical history ----
        if "past medical history" in section_lc or section_lc == "pmh":
            for line in re.split(r"[\n•·;]", text):
                cleaned = re.sub(r"^[-\s*•·]+", "", line).strip().rstrip(",.;:")
                cleaned = re.sub(
                    r"^past\s+medical\s+history\s*[:\-]?\s*",
                    "", cleaned, flags=re.IGNORECASE,
                )
                add(cleaned, "past_medical_history", c["chunk_id"])

        # ---- Free-text "complications following ...", "history of ...", etc.
        # These often appear in supplement comment sections or interview narratives.
        if _is_supplement_section(section, text) or "complaint" in section_lc:
            for m in _FOLLOWING_CONDITION_RE.finditer(text):
                add(m.group(1), "narrative_phrase", c["chunk_id"])

    return allegations


# =============================================================================
# Main entry point
# =============================================================================

def ingest_packet(
    bucket: str,
    key: str,
    case_id: str,
    *,
    profile_name: str = "user",
    region_name: str = "us-east-1",
    project_root: Path | None = None,
    download_pdf: bool = True,
) -> Path:
    """End-to-end ingestion: PDF in S3 -> chunks.json ready for the pipeline.

    Side effects:
      data/<case_id>/chunks.json            (tracked-shape chunks)
      _phi/<case_id>/textract_raw.json      (full raw response — PHI)
      _phi/<case_id>/chunks_with_bbox.json  (full ChunkRecords with bbox)
      _phi/<case_id>/source.pdf             (downloaded PDF for citation links)

    Returns the chunks.json path.
    """
    project_root = project_root or Path(__file__).parent.parent
    session = boto3.Session(profile_name=profile_name, region_name=region_name)

    data_dir = project_root / "data" / case_id
    phi_dir = project_root / "_phi" / case_id
    data_dir.mkdir(parents=True, exist_ok=True)
    phi_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = data_dir / "chunks.json"
    raw_path = phi_dir / "textract_raw.json"
    bbox_path = phi_dir / "chunks_with_bbox.json"
    pdf_path = phi_dir / "source.pdf"

    # === 1. Kick off Textract job ===
    print(f"[1/5] Starting Textract analysis: s3://{bucket}/{key}")
    job_id = start_analysis(bucket, key, session=session)
    print(f"      JobId: {job_id}")

    # === 2. Poll until done ===
    print("[2/5] Polling for completion...")
    status = wait_for_analysis(job_id, session=session)
    if status != "SUCCEEDED":
        raise RuntimeError(f"Textract job did not succeed: {status}")

    # === 3. Fetch all paginated blocks ===
    print("[3/5] Fetching all blocks (paginated)...")
    blocks = fetch_all_blocks(job_id, session=session)
    print(f"      total blocks: {len(blocks)}")
    raw_path.write_text(json.dumps({"Blocks": blocks}), encoding="utf-8")
    print(f"      raw response saved to {raw_path}")

    # === 4. (Optional) download PDF for citation links ===
    if download_pdf:
        print("[4/5] Downloading PDF for citation links...")
        s3 = session.client("s3")
        s3.download_file(bucket, key, str(pdf_path))
        print(f"      saved: {pdf_path}")
    else:
        print("[4/5] (skipping PDF download)")

    # === 5. Detect sub-docs, chunk, write outputs ===
    print("[5/5] Detecting sub-documents and chunking...")
    sub_docs = detect_sub_documents(blocks)
    print(f"      detected {len(sub_docs)} sub-document(s)")

    idx_all = {b["Id"]: b for b in blocks}

    documents: list[dict] = []
    pipeline_chunks: list[dict] = []
    full_records: list[ChunkRecord] = []

    for sd in sub_docs:
        meta = _extract_doc_metadata(sd["blocks"], idx_all, derived_title=sd.get("derived_title", ""))
        documents.append({
            "doc_id": sd["doc_id"],
            "doc_type": meta["doc_type"],
            "doc_title": meta["doc_title"],
            "encounter_date": meta["encounter_date"],
            "page_range_in_packet": [sd["page_start"], sd["page_end"]],
        })
        print(
            f"        {sd['doc_id']}  pp.{sd['page_start']:>2}-{sd['page_end']:<2}  "
            f"type={meta['doc_type']:<25}  date={meta['encounter_date']}"
        )

        records = chunk_by_layout(
            {"Blocks": sd["blocks"]},
            doc_id=sd["doc_id"],
            doc_meta=meta,
        )
        full_records.extend(records)
        pipeline_chunks.extend(to_pipeline_chunks(records))

    # Auto-allegations (review manually)
    allegations = auto_extract_allegations(pipeline_chunks)

    # Tracked-shape chunks.json (no bbox, used by pipeline)
    out_data = {
        "case_id": case_id,
        "source_pdf": f"_phi/{case_id}/source.pdf" if download_pdf else "",
        "_note": (
            f"Auto-generated by Textract ingestion from s3://{bucket}/{key}. "
            f"Allegations auto-extracted from chief complaint / visit diagnoses / PMH; "
            f"review before relying on for matching."
        ),
        "documents": documents,
        "chunks": pipeline_chunks,
        "allegations": allegations,
    }
    chunks_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"      wrote {chunks_path}")
    print(f"      chunks: {len(pipeline_chunks)}  documents: {len(documents)}  "
          f"allegations: {len(allegations)}")

    # Full chunks with bbox/confidence (sidecar in _phi, for future
    # annotated-PDF generation when we want highlighted citation regions)
    bbox_out = {
        "case_id": case_id,
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "text": r.text,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "bbox_by_page": r.bbox.by_page,
                "ocr_confidence": r.ocr_confidence,
                "layout_block_type": r.layout_block_type,
                "is_table": r.is_table,
                "source_block_ids": r.source_block_ids,
            }
            for r in full_records
        ],
    }
    bbox_path.write_text(json.dumps(bbox_out, indent=2), encoding="utf-8")
    print(f"      bbox sidecar: {bbox_path}")

    return chunks_path


# =============================================================================
# Multi-PDF case ingestion (allegation supplement + medical records, etc.)
# =============================================================================

def ingest_multi_pdf_case(
    case_id: str,
    pdfs: list[dict],
    *,
    profile_name: str = "user",
    region_name: str = "us-east-1",
    project_root: Path | None = None,
) -> Path:
    """Ingest multiple S3 PDFs into a single combined case.

    Each PDF is OCR'd separately, then the PDFs are concatenated into one
    combined source PDF and Textract block Page numbers are remapped so that
    every chunk's page reference points into the combined PDF. The rest of
    the matcher pipeline (sub-doc detection, chunking, annotated PDF, HTML
    citations) consumes the combined output unchanged.

    Args:
        case_id: case identifier — outputs land in data/<case_id>/ and _phi/<case_id>/
        pdfs: list of {bucket, key, role} dicts.
              role is one of {"allegation_source", "medical_evidence"}
              (currently informational; matcher reads chunks uniformly).
              PDFs are processed in list order; concatenation follows the
              same order so allegations come first by convention.

    Side effects produced:
      data/<case_id>/chunks.json            — combined chunks for the matcher
      _phi/<case_id>/source.pdf             — combined PDF for citation links
      _phi/<case_id>/source_NN.pdf          — per-PDF downloaded copies
      _phi/<case_id>/textract_raw.json      — combined raw blocks (page-remapped)
      _phi/<case_id>/chunks_with_bbox.json  — combined bbox sidecar
      _phi/<case_id>/ingest_manifest.json   — record of which PDFs were
                                              ingested, with role + page ranges
    """
    project_root = project_root or Path(__file__).parent.parent
    session = boto3.Session(profile_name=profile_name, region_name=region_name)

    data_dir = project_root / "data" / case_id
    phi_dir = project_root / "_phi" / case_id
    data_dir.mkdir(parents=True, exist_ok=True)
    phi_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = data_dir / "chunks.json"
    raw_path = phi_dir / "textract_raw.json"
    bbox_path = phi_dir / "chunks_with_bbox.json"
    combined_pdf_path = phi_dir / "source.pdf"
    manifest_path = phi_dir / "ingest_manifest.json"

    s3 = session.client("s3")

    all_blocks: list[Block] = []
    page_offset = 0
    per_pdf_paths: list[Path] = []
    manifest: list[dict] = []

    for i, pdf_spec in enumerate(pdfs):
        bucket = pdf_spec["bucket"]
        key = pdf_spec["key"]
        role = pdf_spec.get("role", "evidence")

        print(f"\n=== PDF {i+1}/{len(pdfs)} [{role}]: {key} ===")

        # 1. Kick off Textract analysis
        print("[1/4] Starting Textract analysis...")
        job_id = start_analysis(bucket, key, session=session)
        print(f"      JobId: {job_id}")

        # 2. Wait + fetch blocks
        print("[2/4] Polling for completion...")
        status = wait_for_analysis(job_id, session=session)
        if status != "SUCCEEDED":
            raise RuntimeError(f"Textract job did not succeed for {key}: {status}")
        print("[3/4] Fetching paginated blocks...")
        blocks = fetch_all_blocks(job_id, session=session)
        print(f"      total blocks: {len(blocks)}")

        # 3. Download local copy of this PDF
        local_pdf = phi_dir / f"source_{i+1:02d}.pdf"
        print(f"[4/4] Downloading PDF -> {local_pdf.name}...")
        s3.download_file(bucket, key, str(local_pdf))
        per_pdf_paths.append(local_pdf)

        # Determine page count so we can remap subsequent PDFs' Page numbers
        with fitz.open(local_pdf) as src_doc:
            page_count = src_doc.page_count

        # 4. Remap Page values in this PDF's blocks by the running offset
        if page_offset > 0:
            print(f"      remapping page numbers +{page_offset}")
            for b in blocks:
                if b.get("Page") is not None:
                    b["Page"] = b["Page"] + page_offset

        # Track manifest entry for this PDF
        manifest.append({
            "index": i + 1,
            "role": role,
            "bucket": bucket,
            "key": key,
            "local_path": f"_phi/{case_id}/source_{i+1:02d}.pdf",
            "combined_pages": [page_offset + 1, page_offset + page_count],
            "page_count": page_count,
        })

        all_blocks.extend(blocks)
        page_offset += page_count

    # Save the combined raw response for cheap re-chunking iteration
    raw_path.write_text(json.dumps({"Blocks": all_blocks}), encoding="utf-8")
    print(f"\nCombined raw response saved: {raw_path}")

    # Concatenate all source PDFs into the combined source.pdf
    print(f"Concatenating {len(per_pdf_paths)} PDF(s) into {combined_pdf_path.name}...")
    combined = fitz.open()
    for pdf_path in per_pdf_paths:
        with fitz.open(pdf_path) as src:
            combined.insert_pdf(src)
    combined.save(str(combined_pdf_path), garbage=4, deflate=True)
    combined.close()
    print(f"      {combined_pdf_path} ({sum(m['page_count'] for m in manifest)} pages total)")

    # Write manifest for traceability
    manifest_path.write_text(
        json.dumps({"case_id": case_id, "pdfs": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"      manifest: {manifest_path}")

    # Sub-doc detection + chunking against the combined (remapped) blocks
    print("Detecting sub-documents and chunking...")
    sub_docs = detect_sub_documents(all_blocks)
    print(f"      detected {len(sub_docs)} sub-document(s)")

    idx_all = {b["Id"]: b for b in all_blocks}
    documents: list[dict] = []
    pipeline_chunks: list[dict] = []
    full_records: list[ChunkRecord] = []

    # Map a combined-PDF page number to the local_path of the source PDF
    # that contributed it. Used below to attribute each sub-document to its
    # originating source PDF (populates documents.source_pdf_id in DB).
    def _source_pdf_for_page(page_num: int) -> str | None:
        for m in manifest:
            lo, hi = m["combined_pages"]
            if lo <= page_num <= hi:
                return m["local_path"]
        return None

    for sd in sub_docs:
        meta = _extract_doc_metadata(sd["blocks"], idx_all, derived_title=sd.get("derived_title", ""))
        documents.append({
            "doc_id": sd["doc_id"],
            "doc_type": meta["doc_type"],
            "doc_title": meta["doc_title"],
            "encounter_date": meta["encounter_date"],
            "page_range_in_packet": [sd["page_start"], sd["page_end"]],
            "source_pdf_local_path": _source_pdf_for_page(sd["page_start"]),
        })
        print(
            f"        {sd['doc_id']}  pp.{sd['page_start']:>2}-{sd['page_end']:<2}  "
            f"type={meta['doc_type']:<32}  date={meta['encounter_date']}"
        )
        records = chunk_by_layout(
            {"Blocks": sd["blocks"]},
            doc_id=sd["doc_id"],
            doc_meta=meta,
        )
        full_records.extend(records)
        pipeline_chunks.extend(to_pipeline_chunks(records))

    allegations = auto_extract_allegations(pipeline_chunks)

    out_data = {
        "case_id": case_id,
        "source_pdf": f"_phi/{case_id}/source.pdf",
        "_note": (
            f"Multi-PDF ingestion ({len(pdfs)} source PDFs concatenated). "
            f"Pages remapped to point at combined PDF. See ingest_manifest.json "
            f"for per-source-PDF page ranges."
        ),
        "documents": documents,
        "chunks": pipeline_chunks,
        "allegations": allegations,
    }
    chunks_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nWrote {chunks_path}")
    print(f"      chunks: {len(pipeline_chunks)}  documents: {len(documents)}  "
          f"allegations: {len(allegations)}")

    # Bbox sidecar (for annotated-PDF generation)
    bbox_out = {
        "case_id": case_id,
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "text": r.text,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "bbox_by_page": r.bbox.by_page,
                "ocr_confidence": r.ocr_confidence,
                "layout_block_type": r.layout_block_type,
                "is_table": r.is_table,
                "source_block_ids": r.source_block_ids,
            }
            for r in full_records
        ],
    }
    bbox_path.write_text(json.dumps(bbox_out, indent=2), encoding="utf-8")
    print(f"      bbox sidecar: {bbox_path}")

    return chunks_path


# =============================================================================
# Notebook-friendly convenience: re-chunk from saved raw response without
# re-running Textract (and re-spending money)
# =============================================================================

def rechunk_from_raw(
    case_id: str,
    *,
    project_root: Path | None = None,
) -> Path:
    """Re-run the chunking step against a previously-saved textract_raw.json.
    Useful when you tune chunk_by_layout or sub-doc detection and want to
    iterate without paying for another Textract run.

    Also rewrites the bbox sidecar so confidence/bbox lookups stay in sync
    with the new chunk_ids.
    """
    project_root = project_root or Path(__file__).parent.parent
    raw_path = project_root / "_phi" / case_id / "textract_raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"No saved raw response at {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    blocks = raw["Blocks"]

    chunks_path = project_root / "data" / case_id / "chunks.json"
    bbox_path = project_root / "_phi" / case_id / "chunks_with_bbox.json"
    pdf_path = project_root / "_phi" / case_id / "source.pdf"

    sub_docs = detect_sub_documents(blocks)
    idx_all = {b["Id"]: b for b in blocks}

    documents = []
    pipeline_chunks = []
    full_records = []
    for sd in sub_docs:
        meta = _extract_doc_metadata(sd["blocks"], idx_all, derived_title=sd.get("derived_title", ""))
        documents.append({
            "doc_id": sd["doc_id"],
            "doc_type": meta["doc_type"],
            "doc_title": meta["doc_title"],
            "encounter_date": meta["encounter_date"],
            "page_range_in_packet": [sd["page_start"], sd["page_end"]],
        })
        records = chunk_by_layout(
            {"Blocks": sd["blocks"]},
            doc_id=sd["doc_id"],
            doc_meta=meta,
        )
        full_records.extend(records)
        pipeline_chunks.extend(to_pipeline_chunks(records))

    allegations = auto_extract_allegations(pipeline_chunks)
    out_data = {
        "case_id": case_id,
        "source_pdf": f"_phi/{case_id}/source.pdf" if pdf_path.exists() else "",
        "_note": "Re-chunked from saved Textract response.",
        "documents": documents,
        "chunks": pipeline_chunks,
        "allegations": allegations,
    }
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"  wrote {chunks_path}: {len(pipeline_chunks)} chunks, "
          f"{len(documents)} docs, {len(allegations)} allegations")

    # Refresh bbox sidecar so confidence/bbox lookups stay consistent with the
    # current chunk_ids. Otherwise downstream inspectors see chunk_id mismatches.
    bbox_out = {
        "case_id": case_id,
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "text": r.text,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "bbox_by_page": r.bbox.by_page,
                "ocr_confidence": r.ocr_confidence,
                "layout_block_type": r.layout_block_type,
                "is_table": r.is_table,
                "source_block_ids": r.source_block_ids,
            }
            for r in full_records
        ],
    }
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_path.write_text(json.dumps(bbox_out, indent=2), encoding="utf-8")
    print(f"  bbox sidecar refreshed: {bbox_path}")

    return chunks_path
