"""End-to-end pipeline stub.

Two routing modes:
  python run.py path/to/chunks.json    JSON path (back-compat with synthetic
                                        demo + already-ingested-to-disk cases)
  python run.py --from-db <case_id>    DB path — reads everything from
                                        Postgres, uses SQL pgvector for
                                        candidate identification + per-leaf
                                        retrieval, persists results in the
                                        chunk_leaf_matches table.

Both paths share the same per-leaf LLM evaluation, AND/OR consolidation, and
form-output stages; only the data-loading / retrieval layers differ.

Usage:
    set ANTHROPIC_API_KEY=...
    python run.py                              # default JSON case
    python run.py data/<case>/chunks.json      # specific JSON case
    python run.py --from-db redacted-001-fresh # DB case
"""
from pathlib import Path
import os
import sys
import time

import json

from pipeline.chunks import load_case, load_listings, Chunk, Listing
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
              "Set MOCK_EVAL=0 for real evaluation via Bedrock.")
    else:
        # Real evaluation goes through Amazon Bedrock via boto3. We don't
        # validate creds here — boto3 will raise a clear error at first
        # invoke_model() if SSO is stale or the profile is wrong. Users hit
        # that with `aws sso login --profile user` in a fresh shell.
        print(f"[BEDROCK] Eval via inference profile (AWS_PROFILE="
              f"{os.environ.get('AWS_PROFILE', 'user')}, "
              f"region={os.environ.get('AWS_REGION', 'us-east-1')})")

    # ---- Route to DB or JSON path ----
    if len(argv) >= 3 and argv[1] == "--from-db":
        return run_from_db(argv[2])

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


def run_from_db(case_id_str: str) -> int:
    """DB-resident pipeline path.

    Reads chunks/allegations/listings from Postgres, uses pgvector cosine
    similarity for both candidate identification and per-leaf retrieval,
    persists retrieval results to chunk_leaf_matches, then runs the same
    per-leaf LLM evaluation / consolidation / form rendering as the JSON
    path.

    Args:
        case_id_str: the human-readable case_id (e.g. "redacted-001-fresh")

    Returns:
        0 on success, non-zero on error (mirrors the JSON-path main()).
    """
    # Late imports — keep the DB dependency out of the JSON path's import graph
    from pipeline import db as db_mod
    from pipeline.db_matcher import (
        find_candidates_sql,
        retrieve_chunks_for_leaf_sql,
        persist_chunk_leaf_matches,
        persist_leaf_assessment,
        persist_listing_assessment,
        persist_case_summary,
        to_chunk,
        to_retrieved_chunks,
    )
    from pipeline.groundedness import (
        evidence_strength as compute_evidence_strength,
        groundedness_for_listing,
        decision_summary as build_decision_summary,
        case_summary_text,
        overall_groundedness,
    )
    from pipeline.evaluate import MODEL as EVAL_MODEL
    from datetime import datetime, timezone

    run_started_at = datetime.now(timezone.utc)
    t_start = time.time()

    print(f"[1/6] Connecting to Postgres + loading case {case_id_str!r}...")
    with db_mod.transaction() as conn:
        case_pk = db_mod.get_case_pk_by_case_id(conn, case_id_str)
        if case_pk is None:
            print(f"ERROR: no case row with case_id={case_id_str!r}. "
                  f"Did you run db/load_case_to_db.py?", file=sys.stderr)
            return 3

        # Load chunks + allegations once for downstream renderers and
        # output-side chunks_by_id lookups.
        chunk_rows = db_mod.get_chunks_for_case(conn, case_pk, with_embedding=False)
        allegation_rows = db_mod.get_allegations_for_case(conn, case_pk)
        print(f"      case_pk={case_pk}  chunks={len(chunk_rows)}  "
              f"allegations={len(allegation_rows)}")

        # Build a chunks_by_id mapping in the dataclass shape the renderer
        # expects (chunks.Chunk). The DB rows already have all the fields.
        chunks_by_id: dict[str, Chunk] = {
            r["chunk_id"]: to_chunk(r) for r in chunk_rows
        }
        # str chunk_id -> int chunks.id (PK). Layer 2's leaf_assessment_evidence
        # table FKs on the int PK, but evaluate.py cites by string chunk_id;
        # persist_leaf_assessment uses this map to resolve the FK.
        chunk_pk_by_str_id: dict[str, int] = {
            r["chunk_id"]: r["id"] for r in chunk_rows
        }

        out_dir = OUTPUT_ROOT / case_id_str
        out_dir.mkdir(parents=True, exist_ok=True)

        # Source-PDF + bbox sidecar — same convention as the JSON path. The
        # ingest pipeline writes _phi/<case_id>/source.pdf and chunks_with_bbox.json
        # alongside each ingested case, regardless of whether it's been loaded
        # to the DB. So we just probe those local paths.
        source_pdf_path = None
        bbox_sidecar_path = None
        annotated_pdf_path = None
        phi_dir = HERE / "_phi" / case_id_str
        candidate_pdf = phi_dir / "source.pdf"
        if candidate_pdf.exists():
            source_pdf_path = candidate_pdf
            candidate_bbox = phi_dir / "chunks_with_bbox.json"
            if candidate_bbox.exists():
                bbox_sidecar_path = candidate_bbox
                annotated_pdf_path = phi_dir / "source_annotated.pdf"
        html_pdf_target = annotated_pdf_path or source_pdf_path
        if annotated_pdf_path:
            print(f"      bbox sidecar found; will produce annotated PDF: "
                  f"{annotated_pdf_path.name}")
        elif source_pdf_path:
            print("      no bbox sidecar; citations will be page-level only")
        else:
            print(f"      no source.pdf at {phi_dir}; HTML citations will not "
                  f"be hyperlinked")

        print("[2/6] (skipped: embeddings already in DB)")

        print("[3/6] Identifying candidate listings via SQL pgvector...")
        candidates = find_candidates_sql(
            conn, case_pk, top_k=TOP_K_CANDIDATES,
        )
        if not candidates:
            print("      No candidates surfaced. Did the embedding workers run "
                  "for both allegations and ssa_listings?", file=sys.stderr)
            return 4
        print(f"      Top {len(candidates)} candidates:")
        for c in candidates:
            print(f"        {c.listing.code:>6}  score={c.score:.2f}  "
                  f"{c.listing.title[:60]}")
            for r in c.reasoning[:3]:
                print(f"           - {r}")

        print(f"[4/6] Evaluating leaf criteria for {len(candidates)} candidate(s)...")
        # leaf_db_id -> [retrieved_dict, ...]  (for persist_chunk_leaf_matches)
        retrieved_per_leaf: dict[int, list[dict]] = {}
        # listing.code -> {leaf_path: LeafResult}  (for PDF annotation)
        leaf_results_by_listing: dict[str, dict[str, LeafResult]] = {}
        # Per-listing rollups for Layer 3 (case_summaries)
        listing_outcomes: list[dict] = []

        for cand in candidates:
            listing_pk = cand.listing_pk
            # Re-fetch rule_json with _db_id annotations so we can map each
            # leaf node to its ssa_listing_criteria.id for retrieval + persist.
            rule_json = db_mod.get_listing_criteria_tree(conn, listing_pk)
            # Re-fetch synonyms — the SQL candidate query didn't fill these in;
            # they're not needed for SQL retrieval (no keyword/synonym scan)
            # but the output renderer touches listing.synonyms.
            synonyms = db_mod.get_listing_synonyms(conn, listing_pk)
            listing = Listing(
                code=cand.listing.code,
                title=cand.listing.title,
                body_system=cand.listing.body_system,
                summary=cand.listing.summary,
                synonyms=synonyms,
                rule_json=rule_json,
            )

            print(f"\n  === {listing.code} {listing.title[:60]} ===")
            leaf_results: dict[str, LeafResult] = {}
            # leaf_path -> retrieval list (groundedness scoring uses path key)
            retrieved_by_path: dict[str, list[dict]] = {}
            for leaf in listing.leaves():
                leaf_path = leaf.get("path", "?")
                leaf_db_id = leaf.get("_db_id")

                if leaf_db_id is None:
                    # Internal node masquerading as a leaf, or a leaf with no
                    # criterion row in DB (shouldn't happen, but be defensive).
                    print(f"    leaf {leaf_path:<20} no DB id; skipping")
                    leaf_results[leaf_path] = LeafResult(
                        leaf_path=leaf_path,
                        criterion=leaf.get("criterion", ""),
                        verdict="insufficient",
                        rationale="No criterion_embedding in DB for this leaf.",
                        evidence=[],
                        retrieval_debug=[],
                    )
                    retrieved_by_path[leaf_path] = []
                    continue

                retrieved_dicts = retrieve_chunks_for_leaf_sql(
                    conn, case_pk, leaf_db_id,
                )
                retrieved_per_leaf[leaf_db_id] = retrieved_dicts
                retrieved_by_path[leaf_path] = retrieved_dicts
                retrieved = to_retrieved_chunks(retrieved_dicts)

                print(f"    leaf {leaf_path:<20} retrieved={len(retrieved):<3}",
                      end="", flush=True)
                t_leaf = time.time()
                lr = evaluate_leaf(leaf, listing, retrieved)
                leaf_elapsed = time.time() - t_leaf
                leaf_results[lr.leaf_path] = lr

                # ---- Layer 2 persist: leaf_assessments + evidence rows ----
                top_sim = retrieved_dicts[0]["similarity"] if retrieved_dicts else None
                strength = compute_evidence_strength(
                    verdict=lr.verdict,
                    n_evidence=len(lr.evidence),
                    top_similarity=top_sim,
                )
                persist_leaf_assessment(
                    conn,
                    case_pk=case_pk,
                    listing_pk=listing_pk,
                    leaf_db_id=leaf_db_id,
                    leaf_path=leaf_path,
                    criterion_text=leaf.get("criterion", ""),
                    leaf_result=lr,
                    n_chunks_retrieved=len(retrieved_dicts),
                    top_chunk_similarity=top_sim,
                    evidence_strength=strength,
                    chunk_pk_by_str_id=chunk_pk_by_str_id,
                    model=EVAL_MODEL,
                    elapsed_seconds=leaf_elapsed,
                )
                print(f"-> {lr.verdict} ({len(lr.evidence)} ev, {strength})")

            print(f"[5/6] Consolidating tree for {listing.code}...")
            root = consolidate(listing.rule_json, leaf_results)
            verdict_label = form_verdict(root)
            print(f"      VERDICT: {verdict_label}")

            # ---- Layer 3 persist: listing_assessments row ----
            grounded = groundedness_for_listing(leaf_results, retrieved_by_path)
            cand_rank = len(listing_outcomes) + 1   # candidates are processed in rank order
            decision_text = build_decision_summary(
                listing_code=listing.code,
                listing_title=listing.title,
                form_verdict=verdict_label,
                grounded=grounded,
            )
            persist_listing_assessment(
                conn,
                case_pk=case_pk,
                listing_pk=listing_pk,
                form_verdict=verdict_label,
                root_verdict=root.verdict,
                grounded=grounded,
                candidate_score=cand.score,
                candidate_rank=cand_rank,
                decision_summary=decision_text,
            )
            print(f"      groundedness={grounded['groundedness_score']:.2f}  "
                  f"({grounded['n_met']}M/{grounded['n_not_met']}N/"
                  f"{grounded['n_insufficient']}I of {grounded['n_leaves']})")
            listing_outcomes.append({
                "listing_pk":          listing_pk,
                "listing_code":        listing.code,
                "listing_title":       listing.title,
                "form_verdict":        verdict_label,
                "groundedness_score":  grounded["groundedness_score"],
                "candidate_rank":      cand_rank,
            })

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
                case_id=case_id_str,
                chunks_by_id=chunks_by_id,
            )
            out_path = out_dir / f"{file_stem}.md"
            write_form(out_path, content)
            print(f"      -> {out_path}")

            html = render_form_html(
                listing=listing,
                root_verdict=root,
                leaf_results=leaf_results,
                case_id=case_id_str,
                chunks_by_id=chunks_by_id,
                source_pdf_path=html_pdf_target,
            )
            html_path = out_dir / f"{file_stem}.html"
            write_form(html_path, html)
            print(f"      -> {html_path}")

            leaf_results_by_listing[listing.code] = leaf_results

        # Persist all retrieved chunks for this case's matcher run. Idempotent
        # (DELETE WHERE case_id then INSERT) so re-runs replace, never append.
        if retrieved_per_leaf:
            n_persisted = persist_chunk_leaf_matches(
                conn, case_pk, retrieved_per_leaf,
            )
            print(f"\nPersisted {n_persisted} chunk_leaf_matches rows "
                  f"({len(retrieved_per_leaf)} leaves x avg "
                  f"{n_persisted // max(1, len(retrieved_per_leaf))} chunks)")

        # ---- Layer 3 persist: one case_summaries row ----
        elapsed_total = time.time() - t_start
        overall_g = overall_groundedness(listing_outcomes)
        case_text = case_summary_text(case_id_str, listing_outcomes)
        persist_case_summary(
            conn,
            case_pk=case_pk,
            n_candidates=len(candidates),
            listing_outcomes=listing_outcomes,
            summary_text=case_text,
            overall_groundedness=overall_g,
            started_at=run_started_at,
            elapsed_seconds=elapsed_total,
        )
        print(f"\nCase summary: {case_text}")
        print(f"Overall groundedness: {overall_g:.2f}  (elapsed {elapsed_total:.1f}s)")

        # Optional: annotate the local source PDF with cited-chunk bboxes
        if annotated_pdf_path and bbox_sidecar_path and source_pdf_path:
            cited = collect_cited_chunk_ids(leaf_results_by_listing)
            print(f"\nAnnotating PDF with {len(cited)} cited chunks...")
            out_pdf, n = annotate_pdf(
                source_pdf=source_pdf_path,
                bbox_sidecar=bbox_sidecar_path,
                cited_chunk_ids=cited,
                output_pdf=annotated_pdf_path,
            )
            print(f"      drew {n} highlight rectangles -> {out_pdf}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
