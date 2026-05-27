"""Evidence-strength bucketing + groundedness scoring for matcher output.

Pure functions: take LeafResult + retrieval similarities as input, return
strings / floats / dicts. No DB access here -persistence lives in
pipeline/db_matcher.py (persist_leaf_assessment, persist_listing_assessment,
persist_case_summary).

The two outputs:
  1. evidence_strength: categorical bucket per leaf (strong/moderate/weak/none).
     Used to sort reviewer attention -"which leaves does the AI sound
     confident about, and which need human investigation?"
  2. groundedness_score: 0..1 composite per listing.
     Composite formula:
        0.5 * frac_leaves_decided
      + 0.3 * avg_top_similarity
      + 0.2 * min(avg_evidence_per_leaf, 2) / 2
     The three components are also persisted separately in
     listing_assessments so analytics can re-derive or re-weight.
"""
from __future__ import annotations

from .evaluate import LeafResult


# ---- Evidence-strength thresholds --------------------------------------------
#
# Tuned for the colorectal Textract case (~282 chunks, 39 pages). On a real
# signal, rank-1 chunk similarity tends to hit 0.55-0.75; on form boilerplate
# that incidentally matches a criterion keyword it's 0.4-0.5; on cases with no
# matching content at all rank-1 sits ≤0.35.
STRONG_SIM_FLOOR = 0.50
STRONG_EVIDENCE_FLOOR = 2     # ≥ this many cited evidence items = "strong"
WEAK_SIM_FLOOR = 0.40         # below this, "insufficient" with no nearby chunks = "none"


def evidence_strength(verdict: str,
                      n_evidence: int,
                      top_similarity: float | None) -> str:
    """Bucketed per-leaf signal -one of:
       'strong'   decisive verdict + ≥2 evidence items + top sim ≥ 0.5
       'moderate' decisive verdict + ≥1 evidence item
       'weak'     insufficient verdict but retrieval surfaced something near
       'none'     insufficient AND retrieval found nothing close
    The thresholds live at the top of this module for tuning."""
    sim = top_similarity or 0.0
    if verdict in ("met", "not_met"):
        if n_evidence >= STRONG_EVIDENCE_FLOOR and sim >= STRONG_SIM_FLOOR:
            return "strong"
        if n_evidence >= 1:
            return "moderate"
        # Decisive verdict but no citations -shouldn't happen because
        # evaluate.py rejects citation-less met/not_met responses, but be
        # defensive in case the prompt allows a slip.
        return "weak"
    # insufficient verdict
    if sim >= WEAK_SIM_FLOOR:
        return "weak"
    return "none"


# ---- Per-listing groundedness ------------------------------------------------

def groundedness_for_listing(
    leaf_results: dict[str, LeafResult],
    retrieved_per_leaf: dict[str, list[dict]],
) -> dict:
    """Compute groundedness signals for ONE listing's full leaf set.

    Args:
        leaf_results: {leaf_path: LeafResult} from evaluate_leaf
        retrieved_per_leaf: {leaf_path: [retrieved_chunk_dict, ...]}
            Each retrieved dict carries 'similarity' (used for avg-top-sim).
            Empty list / missing key both treated as no retrieval.

    Returns a dict matching the listing_assessments columns:
        groundedness_score, frac_leaves_decided, avg_top_similarity,
        avg_evidence_per_leaf, n_leaves, n_met, n_not_met, n_insufficient.
    """
    n_leaves = len(leaf_results)
    if n_leaves == 0:
        return _empty_groundedness()

    n_met = sum(1 for lr in leaf_results.values() if lr.verdict == "met")
    n_not_met = sum(1 for lr in leaf_results.values() if lr.verdict == "not_met")
    n_insufficient = n_leaves - n_met - n_not_met

    top_sims: list[float] = []
    for path in leaf_results.keys():
        recs = retrieved_per_leaf.get(path) or []
        if recs:
            top_sims.append(float(recs[0].get("similarity", 0.0)))
    avg_top_sim = sum(top_sims) / len(top_sims) if top_sims else 0.0

    total_evidence = sum(len(lr.evidence) for lr in leaf_results.values())
    avg_evidence = total_evidence / n_leaves

    frac_decided = (n_met + n_not_met) / n_leaves
    # Cap evidence-density contribution: above ~2 cited items per leaf,
    # extra citations don't add proportional confidence.
    evidence_density = min(avg_evidence, 2.0) / 2.0

    composite = (
        0.5 * frac_decided
        + 0.3 * avg_top_sim
        + 0.2 * evidence_density
    )

    return {
        "groundedness_score":    round(composite, 3),
        "frac_leaves_decided":   round(frac_decided, 3),
        "avg_top_similarity":    round(avg_top_sim, 3),
        "avg_evidence_per_leaf": round(avg_evidence, 3),
        "n_leaves":              n_leaves,
        "n_met":                 n_met,
        "n_not_met":             n_not_met,
        "n_insufficient":        n_insufficient,
    }


def _empty_groundedness() -> dict:
    return {
        "groundedness_score":    0.0,
        "frac_leaves_decided":   0.0,
        "avg_top_similarity":    0.0,
        "avg_evidence_per_leaf": 0.0,
        "n_leaves":              0,
        "n_met":                 0,
        "n_not_met":             0,
        "n_insufficient":        0,
    }


# ---- Per-listing reviewer summary --------------------------------------------

def decision_summary(listing_code: str,
                     listing_title: str,
                     form_verdict: str,
                     grounded: dict) -> str:
    """Generate a deterministic 1-2 sentence reviewer-facing summary.

    Suitable for the listing_assessments.decision_summary column and the
    case-summary line items. Kept deterministic so two runs on the same
    leaf_results produce identical text (helpful for diff-based regression
    testing). An LLM-generated rich summary can be a separate column later.
    """
    g = grounded
    short_title = (listing_title[:60] + "…") if len(listing_title) > 60 else listing_title

    if form_verdict == "Meets":
        return (
            f"{listing_code} ({short_title}) - MEETS. "
            f"{g['n_met']}/{g['n_leaves']} criteria satisfied; "
            f"top-evidence similarity avg {g['avg_top_similarity']:.2f}; "
            f"groundedness {g['groundedness_score']:.2f}."
        )
    if form_verdict == "Does not meet/equal":
        return (
            f"{listing_code} ({short_title}) - DOES NOT MEET. "
            f"{g['n_not_met']}/{g['n_leaves']} criteria documented as not present; "
            f"groundedness {g['groundedness_score']:.2f}."
        )
    # Insufficient
    return (
        f"{listing_code} ({short_title}) - INSUFFICIENT, reviewer must investigate. "
        f"{g['n_insufficient']}/{g['n_leaves']} criteria lack chart evidence; "
        f"groundedness {g['groundedness_score']:.2f}."
    )


# ---- Case-level summary ------------------------------------------------------

def case_summary_text(case_id_str: str,
                      listing_outcomes: list[dict]) -> str:
    """Generate a deterministic 1-3 sentence case-level summary.

    Args:
        case_id_str: human-readable case id
        listing_outcomes: list of {listing_code, listing_title, form_verdict,
            groundedness_score, candidate_rank}, ordered however -we'll
            sort by candidate_rank.
    """
    if not listing_outcomes:
        return f"Case {case_id_str}: no candidate listings surfaced."

    ordered = sorted(listing_outcomes, key=lambda o: o.get("candidate_rank", 99))
    top = ordered[0]
    n_meets = sum(1 for o in ordered if o["form_verdict"] == "Meets")

    lead = (
        f"Case {case_id_str}: evaluated {len(ordered)} candidate listing(s); "
        f"top candidate {top['listing_code']} -> {top['form_verdict']} "
        f"(groundedness {top['groundedness_score']:.2f})."
    )
    if n_meets > 0:
        lead += f" {n_meets} listing(s) appear MET; reviewer should verify."
    else:
        lead += " No listing surfaced as MET; reviewer should confirm."
    return lead


def overall_groundedness(listing_outcomes: list[dict]) -> float:
    """Mean of per-listing groundedness scores. Used in case_summaries."""
    scores = [o["groundedness_score"] for o in listing_outcomes
              if o.get("groundedness_score") is not None]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 3)
