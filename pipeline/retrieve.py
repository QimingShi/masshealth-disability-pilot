"""Per-leaf retrieval: three variants unioned, deduped, and adaptively cut.

  Variant 1: embed(criterion_text) -> top-K chunks by cosine
  Variant 2: keyword scan of leaf's keywords[] across all chunks
  Variant 3: synonym-expanded scan using the listing's synonyms map

The three variants return a combined set of (chunk, score) pairs. The final
selection step takes the top-scoring chunks using an adaptive strategy:
  - always include at least K_FLOOR chunks (so the LLM has baseline context
    even on leaves where evidence is weak)
  - add more chunks beyond the floor only while their score >= RELEVANCE_THRESHOLD
  - never exceed K_CEILING (caps per-leaf LLM input cost)

Why adaptive: Textract packets produce ~10x as many chunks as hand-transcribed
ones, and Textract over-splits content (each med on its own line, each lab row,
etc.). With a fixed top-K, key evidence chunks get displaced by form
boilerplate that incidentally matches a criterion keyword. Adaptive K
guarantees recall on cases with broad signal while bounding cost on cases
where the relevant content is concentrated.
"""
from dataclasses import dataclass
import re

from .chunks import Chunk, Listing
from .embed import ChunkIndex


# ---- Adaptive selection parameters ----
#
# Tuned empirically against the colorectal Textract case (~282 chunks, 39 pages):
#   - K_FLOOR=10 because key semantic-only chunks (positive embedding but no
#     keyword hit) often score 0.35-0.45 cosine, which is just below where
#     keyword-hit chunks cluster (0.6+). With K_FLOOR=5, those semantically
#     relevant chunks were getting outranked by keyword-positive but
#     semantically-opposing chunks ("no metastasis" outranking the actual
#     metastasis impression because both contain the word).
#   - RELEVANCE_THRESHOLD=0.35 captures the embedding-only "this looks like
#     it's about the criterion" chunks that don't keyword-match.
#   - K_CEILING=25 caps per-leaf LLM input cost at ~3.7K tokens.
K_FLOOR = 10
K_CEILING = 25
RELEVANCE_THRESHOLD = 0.35


@dataclass
class RetrievedChunk:
    chunk: Chunk
    sources: list[str]  # which variant(s) surfaced it: "embed", "keyword", "synonym"
    score: float


def retrieve_for_leaf(
    leaf: dict,
    listing: Listing,
    chunk_index: ChunkIndex,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return retrieved chunks for a leaf criterion.

    If top_k is None (default), uses adaptive selection bounded by K_FLOOR
    and K_CEILING with a RELEVANCE_THRESHOLD gate. If top_k is an integer,
    falls back to fixed top-K (kept for back-compat / explicit caller control).
    """
    # Internal retrieval pool: fetch up to K_CEILING from each variant in
    # adaptive mode so we have headroom to expand. In fixed mode, fetch top_k.
    fetch_n = K_CEILING if top_k is None else max(top_k, K_FLOOR)

    surfaced: dict[str, RetrievedChunk] = {}

    # --- Variant 1: criterion text embedding ---
    if criterion := leaf.get("criterion"):
        for chunk, score in chunk_index.top_k(criterion, k=fetch_n):
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

    # Sort by score desc
    ranked = sorted(surfaced.values(), key=lambda r: -r.score)

    # ---- Final selection ----
    if top_k is not None:
        return ranked[:top_k]
    return _adaptive_select(ranked)


def _adaptive_select(ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Take the floor unconditionally; add more chunks while they clear the
    relevance threshold; cap at the ceiling."""
    if not ranked:
        return []
    # Always include the top K_FLOOR (even if some are below threshold —
    # the LLM still benefits from seeing the best available context)
    out = list(ranked[:K_FLOOR])
    # Expand past the floor only while chunks remain above the threshold
    for r in ranked[K_FLOOR:K_CEILING]:
        if r.score >= RELEVANCE_THRESHOLD:
            out.append(r)
        else:
            # Scores are sorted descending; once we drop below, the rest are
            # all below too. Stop early.
            break
    return out


def _term_in(term: str, text_lower: str) -> bool:
    """Word-boundary check for short terms, substring for long ones."""
    if len(term) <= 5:
        return bool(re.search(rf"\b{re.escape(term)}\b", text_lower))
    return term in text_lower
