"""Embed chunks and queries with sentence-transformers; in-memory cosine search.

Model: all-MiniLM-L6-v2 (~80MB). Fast, local, no API key. Quality is adequate
for ~tens-to-hundreds of chunks. Swap to a clinical-tuned model in V2 if recall
on conceptual criteria proves weak.
"""
from dataclasses import dataclass
import numpy as np

from .chunks import Chunk, Listing


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return normalized embeddings, shape (n, d)."""
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs


@dataclass
class ChunkIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray  # (n_chunks, d), L2-normalized

    def top_k(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = embed_texts([query])[0]
        scores = self.embeddings @ q  # cosine since both normalized
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


def build_chunk_index(chunks: list[Chunk]) -> ChunkIndex:
    # Prepend the section header to each chunk's text — gives the embedder
    # context about what KIND of content this is (Labs vs HPI vs Impression).
    texts = [f"[{c.section}] {c.text}" for c in chunks]
    return ChunkIndex(chunks=chunks, embeddings=embed_texts(texts))


@dataclass
class ListingIndex:
    listings: list[Listing]
    embeddings: np.ndarray  # (n_listings, d)

    def top_k(self, query: str, k: int = 10) -> list[tuple[Listing, float]]:
        q = embed_texts([query])[0]
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:k]
        return [(self.listings[i], float(scores[i])) for i in idx]


def build_listing_index(listings: list[Listing]) -> ListingIndex:
    # Embed summary alone — that's what the schema is designed for and what
    # `identify_candidate_listings` in the production design uses.
    texts = [lst.summary for lst in listings]
    return ListingIndex(listings=listings, embeddings=embed_texts(texts))
