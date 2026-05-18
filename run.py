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

import json

from pipeline.chunks import load_case, load_listings
from pipeline.embed import build_chunk_index, build_listing_index
from pipeline.candidates import identify_candidates
from pipeline.retrieve import retrieve_for_leaf
from pipeline.evaluate import evaluate_leaf, LeafResult
from pipeline.consolidate import consolidate, form_verdict
from pipeline.output import render_form, render_form_html, write_form
from pipeline.annotate_pdf import annotate_pdf, collect_cited_chunk_ids


HERE = Path(__file__).parent
DEFAULT_CASE_PATH = HERE / "data" / "chunks.json"
LISTINGS_DIR = HERE / "SSA JSON" / "disability-eval-listings" / "disability-eval" / "data" / "listings"
OUTPUT_ROOT = HERE / "output"
TOP_K_CANDIDATES = 5  # how many listings to fully evaluate
# Per-leaf chunk count is adaptive (see pipeline/retrieve.py for the
# K_FLOOR / RELEVANCE_THRESHOLD / K_CEILING parameters).


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

    # Resolve source-PDF and bbox-sidecar paths up front so HTML citations can
    # point at the (eventually-generated) annotated PDF.
    source_pdf_path = None
    bbox_sidecar_path = None
    annotated_pdf_path = None
    try:
        case_meta = json.loads(case_path.read_text(encoding="utf-8"))
        if pdf_name := case_meta.get("source_pdf"):
            candidate = HERE / pdf_name
            if candidate.exists():
                source_pdf_path = candidate
                # Bbox sidecar lives under _phi/<case_id>/ — produced by the
                # Textract ingest path. If it exists, we'll generate an
                # annotated PDF and point HTML citations there.
                bbox_sidecar_path = HERE / "_phi" / case.case_id / "chunks_with_bbox.json"
                if bbox_sidecar_path.exists():
                    annotated_pdf_path = HERE / "_phi" / case.case_id / "source_annotated.pdf"
    except Exception as e:
        print(f"      (warning: could not resolve source PDF: {e})")
    # HTML citations point at the annotated PDF if we'll generate one,
    # otherwise at the original.
    html_pdf_target = annotated_pdf_path or source_pdf_path
    if annotated_pdf_path:
        print(f"      bbox sidecar found; will produce annotated PDF: {annotated_pdf_path.name}")
    elif source_pdf_path:
        print(f"      no bbox sidecar (hand-transcribed case?); citations will be page-level only")

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
    # Track every (listing -> leaf_results) so we can build the annotated PDF
    # after the loop with all cited chunks at once.
    leaf_results_by_listing: dict[str, dict[str, LeafResult]] = {}
    for cand in candidates:
        listing = cand.listing
        print(f"\n  === {listing.code} {listing.title[:60]} ===")
        leaf_results: dict[str, LeafResult] = {}
        for leaf in listing.leaves():
            # Pass top_k=None to use adaptive selection
            retrieved = retrieve_for_leaf(leaf, listing, chunk_index)
            print(f"    leaf {leaf.get('path','?'):<20} retrieved={len(retrieved):<3}", end="", flush=True)
            lr = evaluate_leaf(leaf, listing, retrieved)
            leaf_results[lr.leaf_path] = lr
            print(f"-> {lr.verdict} ({len(lr.evidence)} evidence)")

        print(f"[5/6] Consolidating tree for {listing.code}...")
        root = consolidate(listing.rule_json, leaf_results)
        verdict_label = form_verdict(root)
        print(f"      VERDICT: {verdict_label}")

        # Filename prefix sorts forms by reviewer-attention priority:
        #   1_MEETS         — needs reviewer verification + signature
        #   2_INSUFFICIENT  — AI couldn't decide; reviewer must investigate
        #   3_DOES_NOT_MEET — AI confident the listing is not met; skim/skip
        # The leading digit guarantees correct alphabetical sort in any file browser.
        priority_prefix = {
            "Meets": "1_MEETS",
            "Insufficient evidence (review chart)": "2_INSUFFICIENT",
            "Does not meet/equal": "3_DOES_NOT_MEET",
        }.get(verdict_label, "9_UNKNOWN")
        file_stem = f"{priority_prefix}_{listing.code}"

        print(f"[6/6] Writing form for {listing.code}...")
        content = render_form(
            listing=listing,
            root_verdict=root,
            leaf_results=leaf_results,
            case_id=case.case_id,
            chunks_by_id=chunks_by_id,
        )
        out_path = out_dir / f"{file_stem}.md"
        write_form(out_path, content)
        print(f"      -> {out_path}")

        html = render_form_html(
            listing=listing,
            root_verdict=root,
            leaf_results=leaf_results,
            case_id=case.case_id,
            chunks_by_id=chunks_by_id,
            source_pdf_path=html_pdf_target,
        )
        html_path = out_dir / f"{file_stem}.html"
        write_form(html_path, html)
        print(f"      -> {html_path}")

        leaf_results_by_listing[listing.code] = leaf_results

    # Annotated PDF: one yellow box per cited chunk's bbox. Reviewers click a
    # citation in the HTML form and land on the page with the cited region
    # already highlighted.
    if annotated_pdf_path and bbox_sidecar_path and source_pdf_path:
        cited = collect_cited_chunk_ids(leaf_results_by_listing)
        print(f"\nAnnotating PDF with {len(cited)} cited chunks...")
        out_path, n = annotate_pdf(
            source_pdf=source_pdf_path,
            bbox_sidecar=bbox_sidecar_path,
            cited_chunk_ids=cited,
            output_pdf=annotated_pdf_path,
        )
        print(f"      drew {n} highlight rectangles -> {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
