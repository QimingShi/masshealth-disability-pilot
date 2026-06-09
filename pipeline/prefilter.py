"""LLM-based relevance pre-filter for the candidate shortlist.

Embedding-only candidate identification (find_candidates_sql) can surface
spurious matches when allegation text is short and listing summaries are
generic — e.g. 'Anxiety' cosine-matching 8.09 'Chronic conditions of the
skin or mucous membranes' above the threshold.

This module asks Claude (via Bedrock) one batched question per case:
"given these allegations and these candidate listings, which candidates
are plausibly relevant to any allegation?" The LLM returns a per-candidate
keep/drop decision; we filter and continue.

One LLM call per case (not per candidate), so the cost is ~$0.05-0.10 per
case — small relative to the ~$2-6 per-leaf evaluation step it protects.
If the LLM call fails (network, parse error, etc.) the filter is permissive
by default — every candidate gets kept, with a warning — so a transient
Bedrock issue can't silently drop work.

API:
    filtered, decisions = prefilter_candidates_with_llm(candidates, allegations)
        - filtered: list[DBCandidate] (only kept ones)
        - decisions: {listing_code: {"keep": bool, "matched_allegations": [...], "reason": str}}
"""
from __future__ import annotations

import json
import re
import time

from .evaluate import (
    _get_bedrock_client,
    BEDROCK_INFERENCE_PROFILE,
    _OUTER_RETRY_MAX,
    _OUTER_RETRY_BASE_S,
    _RETRYABLE_CODES,
)


_PREFILTER_SYSTEM = """You are pre-filtering SSA disability listings against a member's stated allegations.

For each candidate listing, decide whether a reasonable disability reviewer would consider evaluating it given the member's allegations.

"Plausibly relevant" means: at least one allegation could match this listing's body system or named condition. Be liberal — when in doubt, keep the listing. Only drop candidates that are clearly unrelated to every allegation (e.g. "Anxiety" allegation vs. a skin-conditions listing).

You MUST return ONLY valid JSON — no prose, no markdown, no code fences. Output a JSON array, one object per candidate, in the same order as the input list:

[
  {"code": "12.04", "keep": true, "matched_allegations": ["Depression"], "reason": "depressive disorder matches mood-disorder listing"},
  {"code": "8.09", "keep": false, "matched_allegations": [], "reason": "no allegation involves a skin or mucous-membrane condition"}
]
"""


def _build_user_prompt(allegations: list[dict], candidates: list) -> str:
    """Build the user message: list of allegations + list of candidates."""
    alleg_lines = []
    for i, a in enumerate(allegations, start=1):
        text = (a.get("text") or "").strip()
        src  = a.get("source") or ""
        alleg_lines.append(f"  {i}. {text}   (source: {src})")
    alleg_block = "\n".join(alleg_lines) or "  (none)"

    cand_lines = []
    for c in candidates:
        body = c.listing.body_system or "?"
        summ = (c.listing.summary or "").strip()
        if len(summ) > 300:
            summ = summ[:300] + "..."
        cand_lines.append(
            f"  - code: {c.listing.code}\n"
            f"    title: {c.listing.title}\n"
            f"    body_system: {body}\n"
            f"    summary: {summ}"
        )
    cand_block = "\n\n".join(cand_lines) or "  (none)"

    return (
        f"ALLEGATIONS (member's stated impairments):\n{alleg_block}\n\n"
        f"CANDIDATE LISTINGS (surfaced by embedding similarity, may contain false positives):\n\n"
        f"{cand_block}\n\n"
        f"For each candidate listing, decide keep/drop. Return only the JSON array."
    )


def _invoke_with_retry(client, body: dict) -> dict:
    """Call Bedrock with the same retry pattern as evaluate.evaluate_leaf."""
    last_err: Exception | None = None
    for attempt in range(_OUTER_RETRY_MAX):
        try:
            resp = client.invoke_model(
                modelId=BEDROCK_INFERENCE_PROFILE,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(resp["body"].read())
        except Exception as e:                       # noqa: BLE001
            code = getattr(getattr(e, "response", None), "get", lambda *_: {})("Error", {}).get("Code", "")
            if code not in _RETRYABLE_CODES and not any(s in str(e) for s in _RETRYABLE_CODES):
                raise
            last_err = e
            wait = _OUTER_RETRY_BASE_S * (2 ** attempt)
            print(f"      [prefilter] Bedrock {code or 'transient'}: "
                  f"retry {attempt+1}/{_OUTER_RETRY_MAX} in {wait:.0f}s")
            time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("prefilter: Bedrock retry exhausted without exception")


def _parse_llm_response(response_text: str) -> list[dict]:
    """Pull a JSON array out of Claude's response. Tolerates extra prose
    around it by extracting the largest [ ... ] block."""
    # Strip code fences if any
    txt = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()
    # Find the JSON array
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if m:
        txt = m.group(0)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"prefilter: failed to parse LLM response as JSON: {e}\n"
                         f"raw response: {response_text[:500]}")
    if not isinstance(data, list):
        raise ValueError(f"prefilter: expected JSON array, got {type(data).__name__}")
    return data


def prefilter_candidates_with_llm(
    candidates: list,
    allegations: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> tuple[list, dict[str, dict]]:
    """Filter candidates to only those the LLM judges plausibly relevant.

    Args:
        candidates:  list[DBCandidate] from find_candidates_sql
        allegations: list of allegation dicts (must have 'text' and 'source')

    Returns:
        (filtered_candidates, decisions_by_code)
        - filtered_candidates: subset of input with keep=True
        - decisions_by_code: {code: {"keep": bool, "matched_allegations": [...], "reason": str}}

    On any error (LLM unavailable, JSON parse failure, etc.) the filter
    is permissive — returns ALL candidates with a synthetic "keep=true,
    reason='prefilter unavailable'" decision so we don't silently drop
    work because of a Bedrock hiccup.
    """
    if not candidates:
        return [], {}
    if not allegations:
        # Nothing to filter against — let everything through
        return list(candidates), {
            c.listing.code: {"keep": True, "matched_allegations": [],
                              "reason": "no allegations to filter against"}
            for c in candidates
        }

    user_prompt = _build_user_prompt(allegations, candidates)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      _PREFILTER_SYSTEM,
        "messages":    [{"role": "user", "content": user_prompt}],
    }

    try:
        client = _get_bedrock_client()
        raw = _invoke_with_retry(client, body)
        # Anthropic response: content is [{type, text}, ...]
        text = "".join(blk.get("text", "") for blk in raw.get("content", [])
                       if blk.get("type") == "text")
        items = _parse_llm_response(text)
    except Exception as e:
        print(f"      [prefilter] WARNING — LLM call failed: {e}; "
              f"keeping all {len(candidates)} candidates")
        return list(candidates), {
            c.listing.code: {"keep": True, "matched_allegations": [],
                              "reason": f"prefilter unavailable: {e}"}
            for c in candidates
        }

    # Build decisions map. Default-keep for any candidate the LLM forgot
    # to mention (defensive — better to evaluate than silently drop).
    decisions: dict[str, dict] = {}
    for item in items:
        code = item.get("code")
        if not code:
            continue
        decisions[code] = {
            "keep":                bool(item.get("keep", True)),
            "matched_allegations": list(item.get("matched_allegations") or []),
            "reason":              str(item.get("reason") or ""),
        }

    filtered = []
    for c in candidates:
        d = decisions.get(c.listing.code)
        if d is None:
            decisions[c.listing.code] = {
                "keep": True,
                "matched_allegations": [],
                "reason": "no decision returned by LLM; default-kept",
            }
            filtered.append(c)
        elif d["keep"]:
            filtered.append(c)
    return filtered, decisions
