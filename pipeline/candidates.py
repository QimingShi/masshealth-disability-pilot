"""Find candidate listings via UNION of three signals:
  (a) allegation embedding -> listing summary embedding cosine
  (b) ICD codes -> body_system filter, then keyword refinement within system
  (c) chunk text -> leaf keyword / synonym hits
"""
from dataclasses import dataclass, field
from collections import defaultdict
import re

from .chunks import Chunk, Listing, Allegation, extract_icd_codes, icd_to_body_system
from .embed import ChunkIndex, ListingIndex


@dataclass
class SignalEvidence:
    matched: bool = False
    score: float = 0.0
    detail: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    listing: Listing
    score: float = 0.0
    allegation_signal: SignalEvidence = field(default_factory=SignalEvidence)
    icd_signal: SignalEvidence = field(default_factory=SignalEvidence)
    keyword_signal: SignalEvidence = field(default_factory=SignalEvidence)

    def reasoning(self) -> list[str]:
        out = []
        if self.allegation_signal.matched:
            out.append(f"allegation: {'; '.join(self.allegation_signal.detail)}")
        if self.icd_signal.matched:
            out.append(f"ICD: {'; '.join(self.icd_signal.detail)}")
        if self.keyword_signal.matched:
            out.append(f"keywords: {'; '.join(self.keyword_signal.detail)}")
        return out


# ---- signal (a): allegation -> listing summary ----

def _signal_allegations(
    allegations: list[Allegation],
    listing_index: ListingIndex,
    cos_threshold: float = 0.35,
    top_k_per_allegation: int = 5,
) -> dict[str, SignalEvidence]:
    out: dict[str, SignalEvidence] = defaultdict(SignalEvidence)
    for a in allegations:
        hits = listing_index.top_k(a.text, k=top_k_per_allegation)
        for lst, score in hits:
            if score < cos_threshold:
                continue
            ev = out[lst.code]
            ev.matched = True
            ev.score = max(ev.score, score)
            ev.detail.append(f"'{a.text}' ~ summary ({score:.2f})")
    return out


# ---- signal (b): ICD codes -> body_system ----

def _signal_icd(
    chunks: list[Chunk],
    listings: list[Listing],
) -> tuple[dict[str, SignalEvidence], list[dict]]:
    icd_hits = extract_icd_codes(chunks)
    out: dict[str, SignalEvidence] = defaultdict(SignalEvidence)

    # For each ICD code, find its body system and surface a small number of
    # listings in that system. Without an ICD->listing map, we use the
    # body_system filter alone: every listing in that system becomes a weak
    # candidate (will be confirmed or culled by keyword + leaf eval).
    for hit in icd_hits:
        bs = icd_to_body_system(hit["code"])
        if not bs:
            continue
        for lst in listings:
            if lst.body_system != bs:
                continue
            ev = out[lst.code]
            ev.matched = True
            ev.score = max(ev.score, 0.4)  # body-system-only is a weak signal
            ev.detail.append(f"{hit['code']} -> {bs}")
    return out, icd_hits


# ---- signal (c): keyword / synonym hits ----

def _collect_keywords(listing: Listing) -> dict[str, set[str]]:
    """Return {keyword_lower: {leaf_paths_that_use_it}} for a listing."""
    kw_to_paths: dict[str, set[str]] = defaultdict(set)
    for leaf in listing.leaves():
        for kw in leaf.get("keywords", []):
            kw_to_paths[kw.lower()].add(leaf.get("path", "?"))
    # Add synonyms scoped to this listing (canonical and all variants).
    for canonical, variants in listing.synonyms.items():
        for term in [canonical, *variants]:
            kw_to_paths[term.lower()].add("synonym")
    return kw_to_paths


def _signal_keywords(
    chunks: list[Chunk],
    listings: list[Listing],
    min_distinct_keywords: int = 1,
) -> dict[str, SignalEvidence]:
    out: dict[str, SignalEvidence] = defaultdict(SignalEvidence)
    # Lowercase once per chunk; do plain substring lookups. Cheap and explicit.
    chunk_text_lower = [c.text.lower() for c in chunks]

    for lst in listings:
        kw_to_paths = _collect_keywords(lst)
        # Sort by length desc to prefer longest matches (e.g. "perianal fistula"
        # ahead of "fistula") and avoid double-counting overlapping hits.
        kws_sorted = sorted(kw_to_paths.keys(), key=len, reverse=True)

        hit_counts: dict[str, int] = defaultdict(int)
        for ctext in chunk_text_lower:
            for kw in kws_sorted:
                # Use word-boundary regex for terms <= 5 chars to avoid
                # spurious substring hits like "CD" matching inside "CADD".
                if len(kw) <= 5:
                    if re.search(rf"\b{re.escape(kw)}\b", ctext):
                        hit_counts[kw] += 1
                elif kw in ctext:
                    hit_counts[kw] += 1

        if not hit_counts:
            continue
        if len(hit_counts) < min_distinct_keywords:
            continue

        ev = out[lst.code]
        ev.matched = True
        # Score: log-ish on total hits, capped.
        total_hits = sum(hit_counts.values())
        ev.score = min(0.95, 0.3 + 0.1 * len(hit_counts) + 0.02 * total_hits)
        # Detail: top 5 keywords by count.
        top = sorted(hit_counts.items(), key=lambda kv: -kv[1])[:5]
        ev.detail.append(", ".join(f"'{k}'×{v}" for k, v in top))
    return out


# ---- union + rank ----

def identify_candidates(
    chunks: list[Chunk],
    allegations: list[Allegation],
    listings: list[Listing],
    listing_index: ListingIndex,
    top_k: int = 10,
) -> tuple[list[Candidate], list[dict]]:
    listings_by_code = {l.code: l for l in listings}

    aleg_sig = _signal_allegations(allegations, listing_index)
    icd_sig, icd_hits = _signal_icd(chunks, listings)
    kw_sig = _signal_keywords(chunks, listings)

    all_codes = set(aleg_sig) | set(icd_sig) | set(kw_sig)
    candidates = []
    for code in all_codes:
        lst = listings_by_code.get(code)
        if lst is None:
            continue
        c = Candidate(
            listing=lst,
            allegation_signal=aleg_sig.get(code, SignalEvidence()),
            icd_signal=icd_sig.get(code, SignalEvidence()),
            keyword_signal=kw_sig.get(code, SignalEvidence()),
        )
        # Combined score: weighted sum of signals (rank only, not probability).
        c.score = (
            0.5 * c.allegation_signal.score
            + 0.2 * c.icd_signal.score
            + 0.5 * c.keyword_signal.score
        )
        candidates.append(c)

    candidates.sort(key=lambda c: -c.score)
    return candidates[:top_k], icd_hits
