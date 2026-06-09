"""SQL-based candidate identification and per-leaf retrieval, using pgvector.

This is the DB-resident counterpart to pipeline/candidates.py and
pipeline/retrieve.py. Where those modules build in-memory numpy indexes and
do cosine in Python, this module pushes the same work down to Postgres via
pgvector's `<=>` operator. The per-leaf retrieval results are persisted to
the chunk_leaf_matches table for downstream analytics.

What stays the same:
  - The Chunk / Listing dataclasses (we adapt DB rows into them when needed)
  - evaluate.py (LLM per-leaf evaluation) — takes the same RetrievedChunk shape
  - consolidate.py + output.py — operate on leaf_results dicts, no change

What's different from the in-memory path:
  - No build_chunk_index() / build_listing_index() upfront — each query hits DB
  - Candidate identification is one SQL query (allegation→listing cosine)
  - Per-leaf retrieval is one SQL query per leaf (criterion→chunk cosine)
  - chunk_leaf_matches table caches the retrieval for the matcher run
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from uuid import UUID

from .chunks import Chunk, Listing

# Type alias for SSA-side PKs. ssa_listings.id and ssa_listing_criteria.id
# are both UUID in db/schema_v1.sql. psycopg2 returns them as uuid.UUID
# instances by default — Python comparison, hashing, and parameter binding
# all work transparently.
ListingPK   = UUID
LeafPK      = UUID
# Case-side PKs (cases, chunks, documents) are SERIAL/INT.
CasePK      = int
ChunkPK     = int


# ---------------------------------------------------------------------------
#  Candidate identification (SQL, allegation embedding vs listing summary)
# ---------------------------------------------------------------------------

@dataclass
class DBCandidate:
    """A candidate listing surfaced via SQL similarity. Mirrors the
    pipeline.candidates.Candidate dataclass enough that downstream code can
    treat them interchangeably; carries DB pk for follow-up queries."""
    listing_pk: ListingPK            # UUID — ssa_listings.id
    listing: Listing
    score: float
    reasoning: list[str]


def find_candidates_sql(conn, case_pk: CasePK, *,
                        top_k: int = 5,
                        per_allegation_k: int = 5,
                        min_similarity: float = 0.35
                        ) -> list[DBCandidate]:
    """For each allegation, find the top-K closest listings by cosine; UNION
    across allegations, weight by max similarity, return the global top-K.

    This replaces the embedding-cosine portion of pipeline/candidates.py
    (signal A: allegation → listing summary). For now keyword/synonym/ICD
    signals stay in the Python path — they can move to SQL later if needed,
    but embedding-based is the dominant signal in practice.
    """
    cur = conn.cursor()
    # Note on the query shape:
    #   ssa_listings.rule_json is column type JSON (not JSONB). PostgreSQL's
    #   JSON type has no equality operator, so rule_json can't appear in
    #   GROUP BY. We aggregate on listing_id alone (it's the PK and
    #   functionally determines all other listing columns) and then JOIN
    #   ssa_listings back in to fetch the metadata + rule_json. Cheap — the
    #   join hits the PK index for at most top_k rows.
    cur.execute("""
        WITH allegation_listing_pairs AS (
            SELECT a.id        AS allegation_id,
                   a.text      AS allegation_text,
                   l.id        AS listing_id,
                   1 - (a.embedding <=> l.summary_embedding) AS similarity
            FROM allegations a
            CROSS JOIN ssa_listings l
            WHERE a.case_id = %(case_pk)s
              AND a.embedding IS NOT NULL
              AND l.summary_embedding IS NOT NULL
        ),
        top_per_allegation AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY allegation_id
                       ORDER BY similarity DESC
                   ) AS rank
            FROM allegation_listing_pairs
        ),
        kept AS (
            SELECT *
            FROM top_per_allegation
            WHERE rank <= %(per_allegation_k)s
              AND similarity >= %(min_similarity)s
        ),
        aggregated AS (
            SELECT listing_id,
                   MAX(similarity)                                AS best_similarity,
                   COUNT(DISTINCT allegation_id)                  AS n_allegations,
                   array_agg(DISTINCT
                       allegation_text || ' ~ summary (' ||
                       to_char(similarity, 'FM0.00') || ')'
                   )                                              AS reasoning_bits
            FROM kept
            GROUP BY listing_id
        )
        SELECT ag.listing_id, l.code, l.title, l.body_system, l.summary, l.rule_json,
               ag.best_similarity, ag.n_allegations, ag.reasoning_bits
        FROM aggregated ag
        JOIN ssa_listings l ON l.id = ag.listing_id
        ORDER BY ag.best_similarity DESC, ag.n_allegations DESC
        LIMIT %(top_k)s
    """, {
        "case_pk":           case_pk,
        "per_allegation_k":  per_allegation_k,
        "min_similarity":    min_similarity,
        "top_k":             top_k,
    })

    out: list[DBCandidate] = []
    for row in cur.fetchall():
        (listing_id, code, title, body_system, summary, rule_json,
         best_sim, n_alleg, reasoning_bits) = row
        listing = Listing(
            code=code,
            title=title,
            body_system=body_system,
            summary=summary,
            synonyms={},   # filled in lazily if needed
            rule_json=rule_json,
        )
        out.append(DBCandidate(
            listing_pk=listing_id,
            listing=listing,
            score=float(best_sim),
            reasoning=list(reasoning_bits) if reasoning_bits else [],
        ))
    return out


def retrieve_chunks_for_allegation_sql(conn, case_pk: CasePK,
                                        allegation_id: int, *,
                                        top_k: int = 5,
                                        ) -> list[dict]:
    """Find the top-K chart chunks most similar to a specific allegation.

    Used by run.py's no-listings fallback: when find_candidates_sql
    returns nothing, we surface allegation-by-allegation evidence so the
    reviewer can see what the chart actually says about each claim and
    decide which listings (if any) to manually --include.

    Returns chunk dicts with enough metadata for citation rendering:
    provider (doc_title), date, section, page range, text snippet, and
    similarity score.
    """
    cur = conn.cursor()
    cur.execute("""
        WITH allegation_vec AS (
            SELECT embedding
            FROM allegations
            WHERE id = %(allegation_id)s
              AND embedding IS NOT NULL
        )
        SELECT c.id, c.chunk_id, d.doc_id, d.doc_type, d.doc_title,
               d.encounter_date, c.section, c.page_start, c.page_end,
               c.text, c.bbox,
               1 - (c.embedding <=> av.embedding) AS similarity
        FROM chunks c
        JOIN documents d  ON d.id = c.document_id
        CROSS JOIN allegation_vec av
        WHERE c.case_id   = %(case_pk)s
          AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> av.embedding
        LIMIT %(top_k)s
    """, {
        "allegation_id": allegation_id,
        "case_pk":       case_pk,
        "top_k":         top_k,
    })
    cols = ("id", "chunk_id", "doc_id", "doc_type", "doc_title",
            "encounter_date", "section", "page_start", "page_end",
            "text", "bbox", "similarity")
    out = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        # psycopg2 returns DATE columns as datetime.date; downstream html
        # rendering expects a string. Match the convention in
        # pipeline.db.get_chunks_for_case.
        if d.get("encounter_date") and hasattr(d["encounter_date"], "isoformat"):
            d["encounter_date"] = d["encounter_date"].isoformat()
        out.append(d)
    return out


# ---------------------------------------------------------------------------
#  Per-leaf retrieval (SQL, criterion embedding vs chunk embedding)
# ---------------------------------------------------------------------------

def retrieve_chunks_for_leaf_sql(conn, case_pk: CasePK, leaf_db_id: LeafPK, *,
                                  top_k: int = 25,
                                  min_similarity: float = 0.0
                                  ) -> list[dict]:
    """Find the top-K chunks for this case most similar to the leaf's
    criterion_embedding. Returns a list of dicts ordered by similarity desc.

    Each dict carries enough metadata for the LLM evaluator to do citation:
      id (chunk pk), chunk_id (string), doc_id, section, text, page_start,
      page_end, bbox, ocr_confidence, doc_type, doc_title, similarity, rank.

    Internal/no-criterion leaves return [] — we never embedded them.
    """
    cur = conn.cursor()
    cur.execute("""
        WITH leaf AS (
            SELECT criterion_embedding AS qvec
            FROM ssa_listing_criteria
            WHERE id = %(leaf_id)s
        )
        SELECT c.id, c.chunk_id, d.doc_id, c.section, c.text,
               c.page_start, c.page_end, c.bbox, c.ocr_confidence,
               d.doc_type, d.doc_title,
               1 - (c.embedding <=> leaf.qvec) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN leaf
        WHERE c.case_id = %(case_pk)s
          AND c.embedding IS NOT NULL
          AND leaf.qvec IS NOT NULL
          AND 1 - (c.embedding <=> leaf.qvec) >= %(min_similarity)s
        ORDER BY c.embedding <=> leaf.qvec
        LIMIT %(top_k)s
    """, {
        "leaf_id":         leaf_db_id,
        "case_pk":         case_pk,
        "min_similarity":  min_similarity,
        "top_k":           top_k,
    })

    out: list[dict] = []
    for rank, row in enumerate(cur.fetchall(), start=1):
        (chunk_pk, chunk_id, doc_id, section, text, page_start, page_end,
         bbox, ocr_conf, doc_type, doc_title, similarity) = row
        out.append({
            "id":               chunk_pk,        # DB pk, needed for chunk_leaf_matches FK
            "chunk_id":         chunk_id,        # string id, for citations
            "doc_id":           doc_id,
            "section":          section,
            "text":             text,
            "page_start":       page_start,
            "page_end":         page_end,
            "bbox":             bbox,
            "ocr_confidence":   ocr_conf,
            "doc_type":         doc_type,
            "doc_title":        doc_title,
            "similarity":       float(similarity),
            "rank":             rank,
        })
    return out


# ---------------------------------------------------------------------------
#  Persist retrieval results for downstream analytics
# ---------------------------------------------------------------------------

def persist_chunk_leaf_matches(conn, case_pk: CasePK,
                               retrieved_per_leaf: dict[LeafPK, list[dict]]) -> int:
    """Idempotent write to chunk_leaf_matches for one matcher run.

    Args:
        case_pk: cases.id
        retrieved_per_leaf: {leaf_db_id: [retrieved_chunk_dict, ...]}
            each retrieved_chunk_dict has 'id' (chunks.id), 'similarity',
            'rank' from retrieve_chunks_for_leaf_sql.

    Returns: number of rows inserted (= sum of len(v) for v in retrieved_per_leaf.values()).
    """
    cur = conn.cursor()
    # Clear the case's previous retrieval cache so re-runs don't accumulate
    cur.execute(
        "DELETE FROM chunk_leaf_matches WHERE case_id = %s",
        (case_pk,),
    )
    inserted = 0
    for leaf_db_id, retrieved in retrieved_per_leaf.items():
        for rec in retrieved:
            cur.execute("""
                INSERT INTO chunk_leaf_matches
                    (case_id, leaf_id, chunk_id, similarity, rank_in_leaf)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                case_pk,
                leaf_db_id,
                rec["id"],
                rec["similarity"],
                rec["rank"],
            ))
            inserted += 1
    return inserted


# ---------------------------------------------------------------------------
#  Layer 2 + 3 persistence: criterion assessments, listing assessments,
#  case summary. Each is idempotent per (case, ...) — re-runs replace.
# ---------------------------------------------------------------------------

def persist_leaf_assessment(conn, *,
                            case_pk: CasePK,
                            listing_pk: ListingPK,
                            leaf_db_id: LeafPK,
                            leaf_path: str,
                            criterion_text: str,
                            leaf_result,                          # evaluate.LeafResult
                            n_chunks_retrieved: int,
                            top_chunk_similarity: float | None,
                            evidence_strength: str,
                            chunk_pk_by_str_id: dict[str, int],
                            model: str | None = None,
                            elapsed_seconds: float | None = None,
                            ) -> int:
    """Insert one row into leaf_assessments + N rows into
    leaf_assessment_evidence (one per cited evidence item).

    Idempotent per (case_id, leaf_id) — drops any prior assessment for that
    pair before inserting. Returns leaf_assessments.id.
    """
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM leaf_assessments WHERE case_id = %s AND leaf_id = %s",
        (case_pk, leaf_db_id),
    )
    cur.execute("""
        INSERT INTO leaf_assessments
            (case_id, listing_id, leaf_id, leaf_path, criterion_text,
             verdict, rationale, evidence_strength,
             n_evidence_cited, n_chunks_retrieved, top_chunk_similarity,
             model, elapsed_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        case_pk, listing_pk, leaf_db_id, leaf_path, criterion_text,
        leaf_result.verdict, leaf_result.rationale, evidence_strength,
        len(leaf_result.evidence), n_chunks_retrieved, top_chunk_similarity,
        model, elapsed_seconds,
    ))
    assessment_id = cur.fetchone()[0]

    for i, ev in enumerate(leaf_result.evidence, start=1):
        chunk_pk = chunk_pk_by_str_id.get(ev.chunk_id)
        if chunk_pk is None:
            # evaluate.py already validates that every cited chunk_id is in
            # the input set, so this branch should be unreachable. Log and
            # skip rather than crash if the chart somehow drifts.
            continue
        cur.execute("""
            INSERT INTO leaf_assessment_evidence
                (assessment_id, chunk_id, cite_order, quote, relevance)
            VALUES (%s, %s, %s, %s, %s)
        """, (assessment_id, chunk_pk, i, ev.quote, ev.relevance))
    return assessment_id


def persist_listing_assessment(conn, *,
                               case_pk: CasePK,
                               listing_pk: ListingPK,
                               form_verdict: str,
                               root_verdict: str,
                               grounded: dict,
                               candidate_score: float | None = None,
                               candidate_rank: int | None = None,
                               decision_summary: str | None = None,
                               ) -> int:
    """Insert/replace one row in listing_assessments for (case, listing).

    grounded must have the keys produced by
    pipeline/groundedness.py::groundedness_for_listing.
    Returns listing_assessments.id.
    """
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM listing_assessments WHERE case_id = %s AND listing_id = %s",
        (case_pk, listing_pk),
    )
    cur.execute("""
        INSERT INTO listing_assessments
            (case_id, listing_id, form_verdict, root_verdict,
             groundedness_score, frac_leaves_decided, avg_top_similarity,
             avg_evidence_per_leaf,
             n_leaves, n_load_bearing, n_met, n_not_met, n_insufficient,
             candidate_score, candidate_rank, decision_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        case_pk, listing_pk, form_verdict, root_verdict,
        grounded["groundedness_score"],
        grounded["frac_leaves_decided"],
        grounded["avg_top_similarity"],
        grounded["avg_evidence_per_leaf"],
        grounded["n_leaves"],
        grounded.get("n_load_bearing"),   # NULL-safe for callers using legacy dict
        grounded["n_met"],
        grounded["n_not_met"],
        grounded["n_insufficient"],
        candidate_score, candidate_rank, decision_summary,
    ))
    return cur.fetchone()[0]


def persist_case_summary(conn, *,
                         case_pk: CasePK,
                         n_candidates: int,
                         listing_outcomes: list[dict],
                         summary_text: str | None,
                         overall_groundedness: float,
                         started_at=None,
                         elapsed_seconds: float | None = None,
                         ) -> int:
    """Insert/replace the single case_summaries row for this case.

    listing_outcomes: list of {listing_pk, listing_code, listing_title,
        form_verdict, groundedness_score, candidate_rank}.

    top_listing_id / top_form_verdict are the "headline" candidate -- picked
    by verdict priority (Meets > Does not meet > Insufficient), not by SQL
    similarity rank. If any candidate met, that's what surfaces here, even
    if a higher-similarity candidate came back insufficient. See
    pipeline/groundedness.py::pick_headline_outcome for the tiebreaker.
    """
    # Late import to keep this module's import graph minimal at module load.
    from .groundedness import pick_headline_outcome

    cur = conn.cursor()
    cur.execute("DELETE FROM case_summaries WHERE case_id = %s", (case_pk,))

    headline = pick_headline_outcome(listing_outcomes)
    n_meets = sum(1 for o in listing_outcomes
                  if o["form_verdict"] == "Meets")
    n_dnm = sum(1 for o in listing_outcomes
                if o["form_verdict"] == "Does not meet/equal")
    n_insuf = sum(1 for o in listing_outcomes
                  if "Insufficient" in o["form_verdict"])

    cur.execute("""
        INSERT INTO case_summaries
            (case_id, top_listing_id, top_form_verdict, any_meets,
             n_candidates_evaluated, n_meets, n_does_not_meet, n_insufficient,
             overall_groundedness, summary_text,
             started_at, elapsed_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        case_pk,
        headline["listing_pk"] if headline else None,
        headline["form_verdict"] if headline else None,
        n_meets > 0,
        n_candidates, n_meets, n_dnm, n_insuf,
        overall_groundedness, summary_text,
        started_at, elapsed_seconds,
    ))
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
#  Adapter: DB-shape chunk dicts → pipeline.chunks.Chunk dataclass + a
#  RetrievedChunk-compatible shape that evaluate.py understands.
# ---------------------------------------------------------------------------

def to_chunk(rec: dict) -> Chunk:
    """Build a Chunk dataclass from a get_chunks_for_case() row."""
    return Chunk(
        chunk_id=rec["chunk_id"],
        doc_id=rec["doc_id"],
        section=rec.get("section", ""),
        text=rec["text"],
        page_start=rec.get("page_start") or 0,
        page_end=rec.get("page_end") or 0,
        encounter_date=rec.get("encounter_date"),
        doc_type=rec.get("doc_type"),
        doc_title=rec.get("doc_title"),
    )


def to_retrieved_chunks(retrieved_dicts: list[dict]):
    """Adapt SQL retrieval rows to the RetrievedChunk shape evaluate.py expects.

    evaluate.py's RetrievedChunk has: chunk (Chunk), sources (list[str]), score.
    """
    # Late import to keep the module-load graph minimal
    from .evaluate import RetrievedChunk
    out: list[RetrievedChunk] = []
    for rec in retrieved_dicts:
        out.append(RetrievedChunk(
            chunk=to_chunk(rec),
            sources=[f"embed({rec['similarity']:.2f})"],
            score=rec["similarity"],
        ))
    return out
