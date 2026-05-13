"""Load case chunks (mock OCR output) and SSA listing JSONs."""
from dataclasses import dataclass, field
from pathlib import Path
import json
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


def load_case(path: Path) -> Case:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs_by_id = {d["doc_id"]: d for d in data.get("documents", [])}
    chunks = []
    for c in data["chunks"]:
        doc = docs_by_id.get(c["doc_id"], {})
        chunks.append(Chunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            section=c["section"],
            text=c["text"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            encounter_date=doc.get("encounter_date"),
            doc_type=doc.get("doc_type"),
            doc_title=doc.get("doc_title"),
        ))
    allegations = [Allegation(**a) for a in data.get("allegations", [])]
    return Case(
        case_id=data["case_id"],
        chunks=chunks,
        allegations=allegations,
        documents_by_id=docs_by_id,
    )


def load_listings(listings_dir: Path) -> list[Listing]:
    """Recursively load every listing JSON under listings_dir/<body_system>/*.json."""
    listings = []
    for jf in listings_dir.rglob("*.json"):
        # Skip bundle file and any non-listing JSONs that lack the required shape.
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "rule_json" not in data:
            continue
        listings.append(Listing(
            code=data["code"],
            title=data["title"],
            body_system=data["body_system"],
            summary=data["summary"],
            synonyms=data.get("synonyms", {}),
            rule_json=data["rule_json"],
        ))
    return listings


# ----- structured extractions -----

ICD10_PATTERN = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)\b")


def extract_icd_codes(chunks: list[Chunk]) -> list[dict]:
    """Pull ICD-10 codes from chunks. Returns one dict per (code, chunk_id) hit."""
    hits = []
    for c in chunks:
        # Only scan chunks likely to contain coded diagnoses; full-text scan
        # produces false positives from e.g. "C1D1" cycle notation.
        if "Diagnoses" not in c.section and "Visit" not in c.section and "PMH" not in c.section:
            continue
        for m in ICD10_PATTERN.finditer(c.text):
            hits.append({
                "code": m.group(1),
                "chunk_id": c.chunk_id,
                "context": c.text[max(0, m.start() - 40): m.end() + 20],
            })
    return hits


# ICD-10 prefix → SSA body_system.
ICD_TO_BODY_SYSTEM = {
    "A": "immune",        # certain infectious diseases — closest SSA mapping
    "B": "immune",        # HIV etc
    "C": "cancer",        # neoplasms
    "D": "hematological", # blood disorders (D5-D8); D0-D4 also cancer
    "E": "endocrine",     # endocrine, nutritional, metabolic
    "F": "mental",        # mental disorders
    "G": "neurological",
    "H": "special_senses",
    "I": "cardiovascular",
    "J": "respiratory",
    "K": "digestive",
    "L": "skin",
    "M": "musculoskeletal",
    "N": "genitourinary",
    "Q": "congenital_multisystem",
}


def icd_to_body_system(code: str) -> str | None:
    """Map an ICD-10 code's first letter to an SSA body_system."""
    if not code:
        return None
    first = code[0].upper()
    # D00-D49 are neoplasms; D50-D89 are blood disorders.
    if first == "D":
        try:
            n = int(code[1:3])
        except ValueError:
            return "hematological"
        return "cancer" if n <= 49 else "hematological"
    return ICD_TO_BODY_SYSTEM.get(first)
