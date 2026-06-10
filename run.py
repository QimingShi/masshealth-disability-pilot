"""End-to-end matcher entry point.

Reads a case from Postgres (chunks + allegations + listings, all already
embedded by the Titan v2 workers), identifies candidate listings via
pgvector cosine, runs per-leaf retrieval + Bedrock Claude evaluation,
consolidates AND/OR verdicts, persists the three-layer audit trail, and
renders the reviewer-facing HTML (per-listing forms + a case-summary
report with sticky sidebar nav).

Usage:
    aws sso login --profile user
    set DATABASE_URL=postgresql://<user>@<host>:5432/<db>
    set PGPASSWORD=<password>
    python run.py --from-db <case_id>
"""
from pathlib import Path
import os
import sys
import time

import json

from pipeline.chunks import Chunk, Listing
from pipeline.evaluate import evaluate_leaf, LeafResult
from pipeline.consolidate import consolidate, form_verdict, load_bearing_leaf_paths
from pipeline.output import render_form, render_form_html, render_case_summary_html, write_form
from pipeline.output_docx import render_form_docx, find_template, LISTINGS_WITHOUT_TEMPLATE
from pipeline.annotate_pdf import annotate_pdf, collect_cited_chunk_ids


HERE = Path(__file__).parent
OUTPUT_ROOT = HERE / "output"
TEMPLATES_DIR = HERE / "templates" / "listings"   # hand-crafted UMass DES forms
# Candidate-shortlist sizes — split by allegation source so chart-derived
# diagnoses (typically the larger, more authoritative pool) and patient-
# supplement claims both get representation. The two buckets are deduped
# on listing_id before the LLM prefilter, so the effective max sent for
# review is MEDICAL_TOP_K + SUPPLEMENT_TOP_K but the actual count is
# usually lower (overlapping listings get merged).
MEDICAL_TOP_K     = 10    # top listings from chart-derived allegations
SUPPLEMENT_TOP_K  = 5     # top listings from supplement allegations
TOP_K_CANDIDATES  = MEDICAL_TOP_K + SUPPLEMENT_TOP_K   # legacy alias

# Which `allegations.source` values count as which bucket.
MEDICAL_ALLEGATION_SOURCES = [
    "visit_diagnoses",
    "past_medical_history",
    "problem_list",
    "chief_complaint",
    "narrative_phrase",
]
SUPPLEMENT_ALLEGATION_SOURCES = [
    "supplement_part1",
    "supplement_part2",
    "supplement_form",
]


def main(argv: list[str]) -> int:
    """CLI dispatcher.

    Usage:
        python run.py --from-db <case_id>
        python run.py --from-db <case_id> --include 1.18
        python run.py --from-db <case_id> --include 1.18,2.04,3.02

    --include force-evaluates specific listing codes that didn't make the
    top-K candidate shortlist. Useful when the reviewer knows a case
    should be checked against a particular listing regardless of how well
    the allegation embeddings matched.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="run.py",
        description=("Run the matcher on a case already loaded into Postgres."),
    )
    p.add_argument(
        "--from-db", dest="case_id", required=True,
        help="case_id (human-readable, as in cases.case_id) to evaluate",
    )
    p.add_argument(
        "--include", default="",
        help=("comma-separated listing codes to force-evaluate even if they "
              "didn't make the top-K shortlist, e.g. --include 1.18,2.04"),
    )
    p.add_argument(
        "--no-prefilter", action="store_true",
        help=("skip the LLM relevance pre-filter (~$0.05/case). The "
              "pre-filter asks Claude to drop candidates that aren't "
              "plausibly related to any allegation (e.g. Anxiety -> 8.09)."),
    )
    args = p.parse_args(argv[1:])
    include_codes = [c.strip() for c in args.include.split(",") if c.strip()]
    if include_codes:
        print(f"--include codes received: {include_codes}")
    if args.no_prefilter:
        print("--no-prefilter: LLM relevance pre-filter disabled")
    return run_from_db(
        args.case_id,
        include_codes=include_codes,
        prefilter=not args.no_prefilter,
    )


def run_from_db(case_id_str: str, *,
                 include_codes: list[str] | None = None,
                 prefilter: bool = True) -> int:
    """Run the matcher against a case that's already loaded into Postgres.

    Reads chunks, allegations, and the listing reference data from the DB,
    identifies candidate listings via pgvector cosine similarity (allegation
    embedding -> listing summary embedding), runs per-leaf retrieval (leaf
    criterion embedding -> chunk embedding), evaluates each leaf with
    Bedrock Claude under verbatim-citation guardrails, consolidates the
    AND/OR criterion tree, persists the three-layer audit trail
    (chunk_leaf_matches / leaf_assessments + evidence / listing_assessments
    + case_summaries), and writes per-listing reviewer forms plus a
    case-level summary HTML with sticky sidebar nav.

    Args:
        case_id_str: the human-readable case_id (e.g. "1234567")

    Returns:
        0 on success, non-zero on error.
    """
    # Late imports — keep psycopg2/pgvector out of the CLI dispatcher's import
    # graph so `python run.py --help` works without DB drivers installed.
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

        # Bundle layout: output/<case_id>_EXPEDITESummary/ — the case_id +
        # project tag makes the folder name self-identifying when downloaded
        # standalone (e.g. on SharePoint, where the path context above the
        # folder is often hidden in the file listing).
        out_dir = OUTPUT_ROOT / f"{case_id_str}_EXPEDITESummary"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Source PDF + bbox sidecar. The matcher cites chunks by chunks.page_start,
        # so the HTML's "source.pdf#page=N" links need a PDF whose pagination
        # matches what the matcher recorded. We bundle that PDF into out_dir
        # itself (via pipeline/output_bundle.ensure_source_pdf) so the case
        # folder is fully self-contained: zip it, upload to SharePoint, share —
        # citations resolve relatively as long as source.pdf sits next to the
        # HTML files.
        #
        # ensure_source_pdf resolution order:
        #   1. out_dir/source.pdf already cached  (re-runs no-op)
        #   2. _phi/<case>/source.pdf on local disk → copied into out_dir
        #   3. Fetch each source_pdfs row from S3 → combine into out_dir/source.pdf
        # Returns None if the case has no S3 records AND no local PHI copy
        # (rare: only synthetic/hand-loaded cases).
        from pipeline.output_bundle import ensure_source_pdf
        phi_dir = HERE / "_phi" / case_id_str
        source_pdf_path = ensure_source_pdf(
            conn,
            case_pk=case_pk,
            out_dir=out_dir,
            phi_dir=phi_dir,
        )

        # Bbox sidecar — drives the yellow-highlight annotation step. We
        # prefer the on-disk sidecar (_phi/<case>/chunks_with_bbox.json,
        # written at ingest time) when it's available because it carries the
        # exact original bbox JSON; otherwise we reconstruct an equivalent
        # sidecar from chunks.bbox in the DB. Either way, downstream code
        # consumes the same JSON file.
        bbox_sidecar_path = None
        annotated_pdf_path = None
        if source_pdf_path is not None:
            candidate_bbox = phi_dir / "chunks_with_bbox.json"
            if candidate_bbox.exists():
                bbox_sidecar_path = candidate_bbox
            else:
                from pipeline.output_bundle import reconstruct_bbox_sidecar
                rebuilt = out_dir / "_chunks_with_bbox.json"
                bbox_sidecar_path = reconstruct_bbox_sidecar(
                    conn, case_pk=case_pk, output_path=rebuilt,
                )
                if bbox_sidecar_path:
                    print(f"      bundle: reconstructed bbox sidecar from DB "
                          f"-> {bbox_sidecar_path.name}")
            if bbox_sidecar_path is not None:
                annotated_pdf_path = out_dir / "source_annotated.pdf"

        # If we produced an annotated PDF, prefer linking to that (highlights).
        # Otherwise link to the plain combined source.pdf (page-level only).
        html_pdf_target = annotated_pdf_path or source_pdf_path
        if annotated_pdf_path:
            print(f"      bbox sidecar found; will produce annotated PDF: "
                  f"{annotated_pdf_path.name}")
        elif source_pdf_path:
            print(f"      source.pdf bundled into out_dir; citations will be "
                  f"page-level (no bbox highlights)")
        else:
            print(f"      WARNING: no source.pdf available (case has no S3 records "
                  f"and no _phi copy); HTML citations will not be hyperlinked")

        print("[2/6] (skipped: embeddings already in DB)")

        print("[3/6] Identifying candidate listings via SQL pgvector "
              "(split by allegation source: medical + supplement)...")
        # Run TWO separate top-K queries — one bucket per allegation source
        # family — then merge by listing_id. Keeps supplement-claimed
        # listings from getting pushed out of the shortlist by the
        # typically-larger chart-derived pool, while still letting strong
        # chart signals come through.
        medical_cands = find_candidates_sql(
            conn, case_pk,
            top_k=MEDICAL_TOP_K,
            allegation_sources=MEDICAL_ALLEGATION_SOURCES,
        )
        supplement_cands = find_candidates_sql(
            conn, case_pk,
            top_k=SUPPLEMENT_TOP_K,
            allegation_sources=SUPPLEMENT_ALLEGATION_SOURCES,
        )
        print(f"      medical bucket    ({MEDICAL_TOP_K} max): "
              f"{len(medical_cands)} candidate(s)")
        print(f"      supplement bucket ({SUPPLEMENT_TOP_K} max): "
              f"{len(supplement_cands)} candidate(s)")

        # Merge: dedupe on listing_pk, combine reasoning, keep max score.
        by_pk: dict = {}
        for c in medical_cands:
            c.reasoning = [f"[medical] {r}" for r in c.reasoning]
            by_pk[c.listing_pk] = c
        for c in supplement_cands:
            tagged = [f"[supplement] {r}" for r in c.reasoning]
            if c.listing_pk in by_pk:
                existing = by_pk[c.listing_pk]
                existing.score = max(existing.score, c.score)
                existing.reasoning = existing.reasoning + tagged
            else:
                c.reasoning = tagged
                by_pk[c.listing_pk] = c
        candidates = sorted(by_pk.values(), key=lambda x: -x.score)
        overlap = (len(medical_cands) + len(supplement_cands)) - len(candidates)
        print(f"      merged unique listings: {len(candidates)} "
              f"(overlap deduped: {overlap})")

        # LLM relevance pre-filter: kill candidates that embedding similarity
        # surfaced but that don't plausibly match any allegation (e.g. the
        # classic "Anxiety -> 8.09 Chronic skin conditions" cosine miss).
        # One batched Bedrock call per case; the resulting decisions are
        # logged for transparency.
        #
        # Runs BEFORE --include so reviewer-forced codes always bypass the
        # filter (the operator's explicit override wins).
        prefilter_decisions: dict[str, dict] = {}
        if prefilter and candidates:
            print(f"[3.5/6] LLM relevance pre-filter "
                  f"(checking {len(candidates)} candidate(s))...")
            from pipeline.prefilter import prefilter_candidates_with_llm
            before = candidates
            candidates, prefilter_decisions = prefilter_candidates_with_llm(
                candidates, allegation_rows,
            )
            kept = {c.listing.code for c in candidates}
            for c in before:
                d = prefilter_decisions.get(c.listing.code, {})
                tag = "KEEP" if c.listing.code in kept else "DROP"
                reason = d.get("reason", "?")
                matched = d.get("matched_allegations") or []
                if matched:
                    print(f"      [prefilter] {tag}  {c.listing.code:>6}  "
                          f"matched={matched}  ({reason})")
                else:
                    print(f"      [prefilter] {tag}  {c.listing.code:>6}  "
                          f"({reason})")
            dropped = [c.listing.code for c in before if c.listing.code not in kept]
            print(f"      [prefilter] kept {len(candidates)}/{len(before)}; "
                  f"dropped {dropped or 'none'}")

        # If candidates list is empty at this point (either find_candidates_sql
        # returned nothing, OR the LLM pre-filter dropped everything), AND
        # the reviewer didn't force-include anything, fall back to an
        # allegation-vs-chart evidence report instead of running expensive
        # per-leaf eval on nothing.
        if not candidates and not include_codes:
            # Use ONLY supplement Part 1 entries — the "Your health problems"
            # table where the patient lists their actual claimed diagnoses.
            # Skip Part 2 (provider names / reasons-for-visit which are terse
            # and less diagnostic, e.g. "Kidneys", "Cancer Center") and all
            # chart-derived sources (PMH / visit_diagnoses / chief_complaint /
            # narrative_phrase / supplement_form regex matches).
            supplement_allegations = [
                a for a in allegation_rows
                if a.get("source") == "supplement_part1"
            ]
            print(f"      No candidate listings to evaluate. Falling back to "
                  f"allegation-vs-chart evidence report ("
                  f"{len(supplement_allegations)} supplement Part 1 "
                  f"allegation(s), {len(allegation_rows) - len(supplement_allegations)} "
                  f"other-sourced allegations skipped)...")
            from pipeline.db_matcher import retrieve_chunks_for_allegation_sql
            from pipeline.output import render_no_listings_fallback_html
            allegation_chunks: list[dict] = []
            for alleg in supplement_allegations:
                chunks = retrieve_chunks_for_allegation_sql(
                    conn, case_pk,
                    allegation_id=alleg["id"],
                    top_k=5,
                )
                allegation_chunks.append({
                    "allegation": alleg,
                    "chunks":     chunks,
                })
                print(f"      [fallback] {alleg.get('text','?')[:60]:<60}  "
                      f"-> {len(chunks)} chunk(s)")
            # IMPORTANT: link citations to the PLAIN source.pdf, not the
            # annotated one — in the fallback path we early-return before
            # the annotate_pdf step, so source_annotated.pdf never gets
            # written and any link to it would 404. source_pdf_path is the
            # combined chart that ensure_source_pdf bundled into out_dir.
            summary_html = render_no_listings_fallback_html(
                case_id=case_id_str,
                allegation_chunks=allegation_chunks,
                source_pdf_path=source_pdf_path,
            )
            summary_path = out_dir / f"0_{case_id_str}_EXPEDITESummary.html"
            write_form(summary_path, summary_html)
            print(f"\nCase-summary (fallback): {summary_path}")
            persist_case_summary(
                conn,
                case_pk=case_pk,
                n_candidates=0,
                listing_outcomes=[],
                summary_text=("No SSA listings matched; allegation-vs-chart "
                              "evidence report rendered instead."),
                overall_groundedness=None,
                started_at=run_started_at,
                elapsed_seconds=time.time() - t_start,
            )
            publish_bucket = os.environ.get("OUTPUT_PUBLISH_BUCKET")
            if publish_bucket:
                try:
                    from pipeline.output_publish import publish_to_s3
                    print(f"\n[publish] Uploading fallback bundle to "
                          f"s3://{publish_bucket}/{case_id_str}/ ...")
                    publish_to_s3(
                        case_id=case_id_str,
                        out_dir=out_dir,
                        bucket=publish_bucket,
                        sharepoint_base_url=os.environ.get(
                            "SHAREPOINT_PUBLISH_BASE_URL"),
                    )
                except Exception as e:
                    print(f"[publish] WARNING — upload failed: {e}")
            print("\nDone.")
            return 0

        # Force-include any listings the operator passed via --include that
        # didn't already make the shortlist. Synthetic score=0.0 and reasoning
        # tag flags them so the printed output makes it obvious they're
        # reviewer-forced, not similarity-driven. Bypasses the pre-filter.
        if include_codes:
            print(f"      [include] processing codes: {include_codes}")
            from pipeline.db import get_listings_by_codes
            from pipeline.db_matcher import DBCandidate
            already = {c.listing.code for c in candidates}
            print(f"      [include] already in top-{TOP_K_CANDIDATES}: "
                  f"{sorted(already)}")
            to_add  = [code for code in include_codes if code not in already]
            already_kept = [c for c in include_codes if c in already]
            if already_kept:
                print(f"      [include] already shortlisted (no-op): "
                      f"{already_kept}")
            print(f"      [include] codes to fetch from DB: {to_add}")
            if to_add:
                forced_rows = get_listings_by_codes(conn, to_add)
                fetched = [r["code"] for r in forced_rows]
                print(f"      [include] DB lookup returned: {fetched}")
                missing = [c for c in to_add if c not in fetched]
                if missing:
                    print(f"      [include] WARNING: codes not found in "
                          f"ssa_listings: {missing}", file=sys.stderr)
                for row in forced_rows:
                    candidates.append(DBCandidate(
                        listing_pk=row["id"],
                        listing=Listing(
                            code=row["code"],
                            title=row["title"],
                            body_system=row["body_system"],
                            summary=row["summary"],
                            synonyms={},
                            rule_json=row["rule_json"],
                        ),
                        score=0.0,
                        reasoning=[f"forced via --include (not in top-{TOP_K_CANDIDATES})"],
                    ))
                print(f"      [include] appended {len(forced_rows)} forced "
                      f"candidate(s); total candidates now {len(candidates)}")

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
        # listing.code -> Listing dataclass / NodeVerdict  (for case-summary HTML)
        listings_by_code: dict[str, Listing] = {}
        root_verdicts_by_code: dict[str, "NodeVerdict"] = {}

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
            # Load-bearing leaves drove the consolidated verdict. Pass them
            # to groundedness so avg_top_similarity averages only over the
            # leaves that mattered, not over irrelevant OR-pathways.
            lb_paths = load_bearing_leaf_paths(listing.rule_json, root)
            grounded = groundedness_for_listing(
                leaf_results, retrieved_by_path, load_bearing_paths=lb_paths,
            )
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
                  f"{grounded['n_insufficient']}I of {grounded['n_leaves']}; "
                  f"load-bearing={grounded['n_load_bearing']})")
            listing_outcomes.append({
                "listing_pk":          listing_pk,
                "listing_code":        listing.code,
                "listing_title":       listing.title,
                "form_verdict":        verdict_label,
                "groundedness_score":  grounded["groundedness_score"],
                "candidate_rank":      cand_rank,
            })

            # Per-listing HTML + MD files used to be written here. They've
            # been removed — the case-summary HTML (0_<case>_EXPEDITESummary.html)
            # embeds every listing's full form inline, so the per-listing
            # files were just duplicate content cluttering the bundle.
            # Reviewer sees everything in the single summary file.
            #
            # The reviewer-facing DOCX forms (one per listing, filled-in
            # UMass Blue Book templates) ARE still written below — those
            # are a different artifact (printable forms, not browseable
            # HTML) and are the reviewer's signoff document.

            print(f"[6/6] Writing docx form for {listing.code}...")

            # Render the UMass Chan DES Word form by filling the hand-crafted
            # template under templates/listings/<code> *.docx. The docx is the
            # artifact reviewers actually print + sign; HTML/MD are the
            # browseable + diff-able views of the same data. Filename mirrors
            # the template's basename (e.g. "1.15 Disorders of the Skeletal
            # Spine ...docx") so reviewers recognize it.
            #
            # 4 listings (12.05 / 5.11 / 5.12 / 8.09) have no UMass template;
            # we skip docx for those and fall back to the HTML/MD just written.
            if listing.code in LISTINGS_WITHOUT_TEMPLATE:
                print(f"      (skipping docx: no UMass template for {listing.code})")
            else:
                tpl_path = find_template(listing.code, TEMPLATES_DIR)
                if tpl_path is None:
                    print(f"      WARNING: no template found for {listing.code} "
                          f"under {TEMPLATES_DIR}; skipping docx")
                else:
                    docx_path = out_dir / tpl_path.name
                    render_form_docx(
                        listing=listing,
                        root_verdict=root,
                        leaf_results=leaf_results,
                        case_id=case_id_str,
                        chunks_by_id=chunks_by_id,
                        template_path=tpl_path,
                        output_path=docx_path,
                    )
                    print(f"      -> {docx_path}")

            leaf_results_by_listing[listing.code] = leaf_results
            listings_by_code[listing.code] = listing
            root_verdicts_by_code[listing.code] = root

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

        # ---- Reviewer-facing case-summary HTML (3-column top table + all forms) ----
        # Single document the reviewer opens to navigate the entire case. The
        # top table maps allegations -> medical evidence -> SSI listing met;
        # detail sections below mirror the per-listing forms with anchor links.
        summary_html = render_case_summary_html(
            case_id=case_id_str,
            allegations=allegation_rows,
            candidates=candidates,
            leaf_results_by_listing=leaf_results_by_listing,
            listing_outcomes=listing_outcomes,
            chunks_by_id=chunks_by_id,
            listings_by_code=listings_by_code,
            root_verdicts_by_code=root_verdicts_by_code,
            source_pdf_path=html_pdf_target,
        )
        summary_path = out_dir / f"{case_id_str}_EXPEDITESummary.html"
        write_form(summary_path, summary_html)
        print(f"\nCase-summary HTML: {summary_path}")

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

            # If we reconstructed the bbox sidecar from the DB (filename
            # starts with "_"), it was a build-time artifact — delete so
            # the SharePoint bundle only ships the reviewer-facing files.
            if bbox_sidecar_path.parent == out_dir and \
               bbox_sidecar_path.name.startswith("_"):
                try:
                    bbox_sidecar_path.unlink()
                except OSError:
                    pass   # non-fatal — re-runs will overwrite anyway

        # ---- Optional: publish the rendered bundle to outgoing S3 ----
        # Opt-in by setting OUTPUT_PUBLISH_BUCKET in the shell environment.
        # When set, every successful matcher run also pushes out_dir to
        # s3://<bucket>/<case_id>/ for downstream consumers (SharePoint
        # sync, reviewer self-serve download, etc.). Skipped silently when
        # the env var is unset — won't trigger AWS calls or block run.py.
        publish_bucket = os.environ.get("OUTPUT_PUBLISH_BUCKET")
        if publish_bucket:
            try:
                from pipeline.output_publish import publish_to_s3
                print(f"\n[publish] Uploading bundle to "
                      f"s3://{publish_bucket}/{case_id_str}/ ...")
                publish_to_s3(
                    case_id=case_id_str,
                    out_dir=out_dir,
                    bucket=publish_bucket,
                    sharepoint_base_url=os.environ.get(
                        "SHAREPOINT_PUBLISH_BASE_URL"),
                )
            except Exception as e:
                # Don't fail the whole run if publish hits a transient AWS
                # error — the local bundle is fine; reviewer can re-publish
                # later via py publish_output.py <case_id>.
                print(f"[publish] WARNING — upload failed: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
