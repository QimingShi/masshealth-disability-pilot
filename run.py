"""End-to-end pipeline stub.

Loads the mock-OCR chunks for this packet, embeds them, finds candidate listings
(UNION of allegation / ICD / keyword signals), evaluates each candidate's
criterion tree leaf-by-leaf via Claude with citation guardrails, consolidates
the AND/OR tree, and writes a populated Matched_Listing form per candidate.

Usage:
    set ANTHROPIC_API_KEY=...
    python run.py
"""
from pathlib import Path
import os
import sys
import time

from pipeline.chunks import load_case, load_listings
from pipeline.embed import build_chunk_index, build_listing_index
from pipeline.candidates import identify_candidates
from pipeline.retrieve import retrieve_for_leaf
from pipeline.evaluate import evaluate_leaf, LeafResult
from pipeline.consolidate import consolidate, form_verdict
from pipeline.output import render_form, render_form_html, write_form


HERE = Path(__file__).parent
DEFAULT_CASE_PATH = HERE / "data" / "chunks.json"
LISTINGS_DIR = HERE / "SSA JSON" / "disability-eval-listings" / "disability-eval" / "data" / "listings"
OUTPUT_ROOT = HERE / "output"
TOP_K_CANDIDATES = 5      # how many listings to fully evaluate
TOP_K_CHUNKS_PER_LEAF = 5  # how many chunks to feed the LLM per leaf


def main(argv: list[str]) -> int:
    mock_mode = os.environ.get("MOCK_EVAL") == "1"
    if mock_mode:
        print("[MOCK MODE] Using canned LLM responses (no API calls). "
              "Set MOCK_EVAL=0 and ANTHROPIC_API_KEY for real evaluation.")
    elif "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY env var not set.", file=sys.stderr)
        print("       Set it before running:  $env:ANTHROPIC_API_KEY = '...'", file=sys.stderr)
        print("       Or run in mock mode:    $env:MOCK_EVAL = '1'", file=sys.stderr)
        return 2

    # Allow `python run.py path/to/chunks.json` to override the default case.
    case_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CASE_PATH
    if not case_path.is_absolute():
        case_path = HERE / case_path

    print(f"[1/6] Loading case from {case_path.name} + listings...")
    case = load_case(case_path)
    listings = load_listings(LISTINGS_DIR)
    out_dir = OUTPUT_ROOT / case.case_id
    print(f"      case={case.case_id}  chunks={len(case.chunks)}  "
          f"allegations={len(case.allegations)}  listings={len(listings)}")

    print("[2/6] Building embedding indexes (chunks + listings)...")
    t0 = time.time()
    chunk_index = build_chunk_index(case.chunks)
    listing_index = build_listing_index(listings)
    print(f"      embedded in {time.time() - t0:.1f}s")

    print("[3/6] Identifying candidate listings...")
    candidates, icd_hits = identify_candidates(
        case.chunks, case.allegations, listings, listing_index, top_k=TOP_K_CANDIDATES,
    )
    print(f"      ICD codes found: {sorted({h['code'] for h in icd_hits})}")
    print(f"      Top {len(candidates)} candidates:")
    for c in candidates:
        print(f"        {c.listing.code:>6}  score={c.score:.2f}  {c.listing.title[:60]}")
        for r in c.reasoning():
            print(f"           - {r}")

    chunks_by_id = {c.chunk_id: c for c in case.chunks}

    print(f"[4/6] Evaluating leaf criteria for {len(candidates)} candidate(s)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        listing = cand.listing
        print(f"\n  === {listing.code} {listing.title[:60]} ===")
        leaf_results: dict[str, LeafResult] = {}
        for leaf in listing.leaves():
            retrieved = retrieve_for_leaf(leaf, listing, chunk_index, top_k=TOP_K_CHUNKS_PER_LEAF)
            print(f"    leaf {leaf.get('path','?'):<20} retrieved={len(retrieved):<2} ", end="", flush=True)
            lr = evaluate_leaf(leaf, listing, retrieved)
            leaf_results[lr.leaf_path] = lr
            print(f"-> {lr.verdict} ({len(lr.evidence)} evidence)")

        print(f"[5/6] Consolidating tree for {listing.code}...")
        root = consolidate(listing.rule_json, leaf_results)
        print(f"      VERDICT: {form_verdict(root)}")

        print(f"[6/6] Writing form for {listing.code}...")
        content = render_form(
            listing=listing,
            root_verdict=root,
            leaf_results=leaf_results,
            case_id=case.case_id,
            chunks_by_id=chunks_by_id,
        )
        out_path = out_dir / f"{listing.code}.md"
        write_form(out_path, content)
        print(f"      -> {out_path}")

        # Resolve source PDF path for clickable citations in HTML output.
        source_pdf_path = None
        import json
        try:
            case_meta = json.loads(case_path.read_text(encoding="utf-8"))
            if pdf_name := case_meta.get("source_pdf"):
                candidate = HERE / pdf_name
                if candidate.exists():
                    source_pdf_path = candidate
        except Exception:
            pass

        html = render_form_html(
            listing=listing,
            root_verdict=root,
            leaf_results=leaf_results,
            case_id=case.case_id,
            chunks_by_id=chunks_by_id,
            source_pdf_path=source_pdf_path,
        )
        html_path = out_dir / f"{listing.code}.html"
        write_form(html_path, html)
        print(f"      -> {html_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
