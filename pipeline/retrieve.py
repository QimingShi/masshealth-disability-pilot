"""Per-leaf retrieval: three variants unioned and deduped.

  Variant 1: embed(criterion_text) -> top-K chunks by cosine
  Variant 2: keyword scan of leaf's keywords[] across all chunks
  Variant 3: synonym-expanded scan using the listing's synonyms map
"""
from dataclasses import dataclass
import re

from .chunks import Chunk, Listing
from .embed import ChunkIndex


@dataclass
class RetrievedChunk:
    chunk: Chunk
    sources: list[str]  # which variant(s) surfaced it: "embed", "keyword", "synonym"
    score: float


def retrieve_for_leaf(
    leaf: dict,
    listing: Listing,
    chunk_index: ChunkIndex,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    surfaced: dict[str, RetrievedChunk] = {}

    # --- Variant 1: criterion text embedding ---
    if criterion := leaf.get("criterion"):
        for chunk, score in chunk_index.top_k(criterion, k=top_k):
            r = surfaced.setdefault(
                chunk.chunk_id,
                RetrievedChunk(chunk=chunk, sources=[], score=0.0),
            )
            r.sources.append(f"embed({score:.2f})")
            r.score = max(r.score, score)

    # --- Variant 2: keyword scan ---
    keywords = [k.lower() for k in leaf.get("keywords", [])]
    if keywords:
        for chunk in chunk_index.chunks:
            tlower = chunk.text.lower()
            hits = [kw for kw in keywords if _term_in(kw, tlower)]
            if not hits:
                continue
            r = surfaced.setdefault(
                chunk.chunk_id,
                RetrievedChunk(chunk=chunk, sources=[], score=0.0),
            )
            r.sources.append(f"keyword({','.join(hits)})")
            # Keyword hit floor of 0.6 — a literal hit is high-confidence retrieval.
            r.score = max(r.score, 0.6 + 0.05 * len(hits))

    # --- Variant 3: synonym-expanded scan ---
    # For each keyword, find any synonym variants in the listing's synonyms map
    # whose canonical term matches the keyword, then scan for those variants.
    expansions = []
    for kw in keywords:
        for canonical, variants in listing.synonyms.items():
            if canonical.lower() == kw or kw in canonical.lower():
                expansions.extend(v.lower() for v in variants)
    if expansions:
        expansions = list(set(expansions) - set(keywords))  # don't double-count
        for chunk in chunk_index.chunks:
            tlower = chunk.text.lower()
            hits = [e for e in expansions if _term_in(e, tlower)]
            if not hits:
                continue
            r = surfaced.setdefault(
                chunk.chunk_id,
                RetrievedChunk(chunk=chunk, sources=[], score=0.0),
            )
            r.sources.append(f"synonym({','.join(hits[:3])})")
            r.score = max(r.score, 0.55 + 0.05 * len(hits))

    out = sorted(surfaced.values(), key=lambda r: -r.score)[:top_k]
    return out


def _term_in(term: str, text_lower: str) -> bool:
    """Word-boundary check for short terms, substring for long ones."""
    if len(term) <= 5:
        return bool(re.search(rf"\b{re.escape(term)}\b", text_lower))
    return term in text_lower
