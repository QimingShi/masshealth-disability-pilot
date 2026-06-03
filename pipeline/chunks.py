"""Core dataclasses (Chunk, Allegation, Case, Listing) + ICD-10 extraction.

Used across the ingest pipeline (chunks come out of Textract via
ingest_real.py), the DB layer (pipeline/db.py persists them), and the matcher
output (pipeline/output.py renders citations from them).

The Listing dataclass mirrors the SSA JSON listing shape; the
.leaves() iterator walks its rule_json criterion tree yielding every
testable atomic criterion.
"""
from dataclasses import dataclass, field
import re
from typing import Iterator


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    page_start: int
    page_end: int
    encounter_date: str | None = None
    doc_type: str | None = None
    doc_title: str | None = None


@dataclass
class Allegation:
    text: str
    source: str
    source_chunk_id: str


@dataclass
class Case:
    case_id: str
    chunks: list[Chunk]
    allegations: list[Allegation]
    documents_by_id: dict[str, dict] = field(default_factory=dict)


@dataclass
class Listing:
    code: str
    title: str
    body_system: str
    summary: str
    synonyms: dict[str, list[str]]
    rule_json: dict

    def leaves(self) -> Iterator[dict]:
        """Yield every leaf node (children-less node) from the rule tree."""
        def walk(node):
            if "children" not in node:
                yield node
            else:
                for c in node["children"]:
                    yield from walk(c)
        yield from walk(self.rule_json)


# ----- structured extractions -----

ICD10_PATTERN = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)\b")


def extract_icd_codes(chunks: list[Chunk]) -> list[dict]:
    """Pull ICD-10 codes from chunks. Returns one dict per (code, chunk_id) hit.

    Scans only chunks whose section name suggests coded diagnoses (Diagnoses,
    Visit, PMH) -- full-text scanning produces false positives from
    cycle/timepoint notation like "C1D1".
    """
    hits = []
    for c in chunks:
        if "Diagnoses" not in c.section and "Visit" not in c.section and "PMH" not in c.section:
            continue
        for m in ICD10_PATTERN.finditer(c.text):
            hits.append({
                "code": m.group(1),
                "chunk_id": c.chunk_id,
                "context": c.text[max(0, m.start() - 40): m.end() + 20],
            })
    return hits
