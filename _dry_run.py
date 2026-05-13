"""Dry-run: exercises everything except the LLM call. Confirms candidate
identification and retrieval before we spend API tokens."""
from pathlib import Path
from pipeline.chunks import load_case, load_listings
from pipeline.embed import build_chunk_index, build_listing_index
from pipeline.candidates import identify_candidates
from pipeline.retrieve import retrieve_for_leaf

HERE = Path(__file__).parent

print("Loading case + listings...")
case = load_case(HERE / "data" / "chunks.json")
listings = load_listings(HERE / "SSA JSON" / "disability-eval-listings" / "disability-eval" / "data" / "listings")
print(f"  case={case.case_id}  chunks={len(case.chunks)}  allegations={len(case.allegations)}  listings={len(listings)}")

print("\nBuilding embedding indexes (one-time download of all-MiniLM-L6-v2 if first run)...")
chunk_index = build_chunk_index(case.chunks)
listing_index = build_listing_index(listings)
print(f"  chunk embeddings shape: {chunk_index.embeddings.shape}")
print(f"  listing embeddings shape: {listing_index.embeddings.shape}")

print("\nIdentifying candidates...")
candidates, icd_hits = identify_candidates(
    case.chunks, case.allegations, listings, listing_index, top_k=5,
)
print(f"  ICD hits: {sorted({h['code'] for h in icd_hits})}")
print(f"\n  Top {len(candidates)} candidate listings:")
for c in candidates:
    print(f"    {c.listing.code:>6}  score={c.score:.2f}  {c.listing.title[:70]}")
    for r in c.reasoning():
        print(f"        - {r}")

print("\n  Per-leaf retrieval for top candidate:")
if candidates:
    top = candidates[0].listing
    print(f"    Listing: {top.code} — {top.title}")
    for leaf in top.leaves():
        retrieved = retrieve_for_leaf(leaf, top, chunk_index, top_k=5)
        path = leaf.get('path', '?')
        crit = (leaf.get('criterion') or '')[:60]
        print(f"\n    leaf {path}: \"{crit}...\"")
        for r in retrieved:
            print(f"      [{r.score:.2f}] {r.chunk.chunk_id}")
            print(f"          sources: {' + '.join(r.sources)}")
            print(f"          text: {r.chunk.text[:120]}...")
