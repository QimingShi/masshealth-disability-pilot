"""Per-leaf evaluation: ask Claude to classify each leaf criterion as
met / not_met / insufficient, with verbatim chunk citations. Enforced server-side:
  - every cited chunk_id must be in the input set (else reject)
  - every quote must appear verbatim in the cited chunk's text (else reject)

Backend: Amazon Bedrock via boto3 + an application inference profile. The
profile binds a specific Claude model + region under the AWS account; rotate
models or regions by reconfiguring the profile in Bedrock, not by editing
this file. Authentication uses an SSO profile (env var AWS_PROFILE, default
"user") — the same path the embedding workers already use.
"""
from dataclasses import dataclass, asdict
import json
import os
import re

import boto3

from .chunks import Listing
from .retrieve import RetrievedChunk


# ---- Bedrock routing --------------------------------------------------------
# The inference profile ARN is what Bedrock actually routes to. The MODEL
# constant is a human-readable label persisted to leaf_assessments.model for
# analytics — change the underlying model by reconfiguring the profile in
# Bedrock, then update the label here to match.
BEDROCK_INFERENCE_PROFILE = os.environ.get(
    "LISTING_EVAL_BEDROCK_PROFILE",
    "arn:aws:bedrock:us-east-1:251862868170:application-inference-profile/i6q45gpa1bzt",
)
MODEL = os.environ.get("LISTING_EVAL_MODEL", "claude-sonnet-4-6 (bedrock)")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "user")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Lazy module-level Bedrock client. Session creation hits ~/.aws/config and is
# non-trivial; the client is thread-safe so cache it across all leaf calls.
_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        session = boto3.Session(profile_name=AWS_PROFILE)
        _bedrock_client = session.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


@dataclass
class EvidenceItem:
    chunk_id: str
    quote: str
    relevance: str


@dataclass
class LeafResult:
    leaf_path: str
    criterion: str
    verdict: str  # "met" | "not_met" | "insufficient"
    rationale: str
    evidence: list[EvidenceItem]
    retrieval_debug: list[str]  # which chunks were surfaced and why


_SYSTEM = """You are evaluating one SSA disability-listing criterion against a member's medical record.

Your output is consumed by a disability reviewer. They will verify your citations against the chart, so be conservative: when evidence is ambiguous, prefer "insufficient" over "met".

For each criterion, you must:
1. Decide a verdict: "met", "not_met", or "insufficient"
   - "met": the chart clearly documents the criterion
   - "not_met": the chart clearly documents the criterion is NOT present (e.g., "ruled out")
   - "insufficient": the chart neither confirms nor rules out the criterion
2. Cite evidence as a list of {chunk_id, quote, relevance}. Quote VERBATIM (you may abbreviate with ellipsis "...", but every non-elided fragment between ellipses must be a literal substring of the cited chunk's text). Only cite chunks from the input set.

Return JSON only, no markdown fence, conforming to:
{
  "verdict": "met" | "not_met" | "insufficient",
  "rationale": "<1-2 sentences>",
  "evidence": [
    {"chunk_id": "<exact id from input>", "quote": "<verbatim from that chunk>", "relevance": "<why this proves the verdict>"}
  ]
}
"""


def evaluate_leaf(
    leaf: dict,
    listing: Listing,
    retrieved: list[RetrievedChunk],
) -> LeafResult:
    """Call Claude on a single leaf with its retrieved chunks. Validates the
    response strictly; returns insufficient on any guardrail failure."""
    leaf_path = leaf.get("path", "?")
    criterion = leaf.get("criterion", "")

    if not retrieved:
        return LeafResult(
            leaf_path=leaf_path,
            criterion=criterion,
            verdict="insufficient",
            rationale="No chunks retrieved for this criterion.",
            evidence=[],
            retrieval_debug=[],
        )

    chunks_by_id = {r.chunk.chunk_id: r.chunk for r in retrieved}
    chunks_payload = "\n\n".join(
        f"[chunk_id: {r.chunk.chunk_id}]\n"
        f"(doc: {r.chunk.doc_type or r.chunk.doc_id}, section: {r.chunk.section}, "
        f"page: {r.chunk.page_start}, date: {r.chunk.encounter_date or 'unknown'}, "
        f"surfaced_by: {' + '.join(r.sources)})\n"
        f"{r.chunk.text}"
        for r in retrieved
    )

    user = (
        f"LISTING: {listing.code} — {listing.title}\n"
        f"LEAF PATH: {leaf_path}\n"
        f"CRITERION: {criterion}\n\n"
        f"=== RETRIEVED CHUNKS FROM MEMBER'S CHART ===\n{chunks_payload}\n"
        f"\nEvaluate."
    )

    if os.environ.get("MOCK_EVAL") == "1":
        from .mock_eval import mock_evaluate
        parsed = mock_evaluate(listing.code, leaf_path, chunks_by_id)
    else:
        response_text = _call_claude(_SYSTEM, user)
        parsed = _parse_response(response_text)
        # Retry once if the response didn't parse — guards against the
        # occasional malformed-JSON glitch. Without this, a single bad response
        # can flip a Meets verdict to Insufficient via the AND consolidation.
        if not parsed:
            response_text = _call_claude(
                _SYSTEM + "\n\nIMPORTANT: Return strictly valid JSON only, no preamble or markdown fence.",
                user,
            )
            parsed = _parse_response(response_text)

    # ---- guardrails ----
    debug = [
        f"{r.chunk.chunk_id} [{'+'.join(r.sources)}, score={r.score:.2f}]"
        for r in retrieved
    ]
    validated_evidence = []
    for e in parsed.get("evidence", []):
        cid = e.get("chunk_id", "")
        quote = e.get("quote", "").strip()
        if cid not in chunks_by_id:
            # Hallucinated chunk_id — drop this evidence item.
            continue
        if not _quote_is_verbatim(quote, chunks_by_id[cid].text):
            # Non-verbatim quote — drop.
            continue
        validated_evidence.append(EvidenceItem(
            chunk_id=cid,
            quote=quote,
            relevance=e.get("relevance", "").strip(),
        ))

    verdict = parsed.get("verdict", "insufficient")
    if verdict not in ("met", "not_met", "insufficient"):
        verdict = "insufficient"

    # If the model claimed "met" or "not_met" but no evidence survived
    # validation, downgrade to "insufficient" — we cannot trust the verdict
    # without verifiable citations.
    if verdict in ("met", "not_met") and not validated_evidence:
        verdict = "insufficient"
        rationale = (
            "Model claimed verdict but no citations survived verification; "
            "downgraded to insufficient. Original rationale: "
            + parsed.get("rationale", "")[:200]
        )
    else:
        rationale = parsed.get("rationale", "").strip()

    return LeafResult(
        leaf_path=leaf_path,
        criterion=criterion,
        verdict=verdict,
        rationale=rationale,
        evidence=validated_evidence,
        retrieval_debug=debug,
    )


def _call_claude(system: str, user: str) -> str:
    """Invoke Claude via Bedrock with the Messages API body shape.

    Why no prompt caching: _SYSTEM here is ~250 tokens, well below Sonnet/Opus
    4.x's 4096-token cacheable-prefix minimum, and the user message varies per
    leaf (different criterion + different retrieved chunks). There is no
    shared prefix large enough to benefit from cache_control. Revisit if we
    restructure to send the full case chunk set as cacheable context.
    """
    client = _get_bedrock_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        # max_tokens=4096: leaves with many retrieved chunks can produce
        # multi-evidence JSON > 1024 tokens; 1024 truncates mid-string and
        # the parse fails.
        "max_tokens": 4096,
        # NOTE: leaving temperature unset (model default). Setting temperature=0
        # was observed to occasionally produce overly-strict "insufficient"
        # verdicts on clear-cut leaves (e.g. 13.18 PRECONDITION on a chart with
        # pathology-confirmed CRC). Default temperature performs better.
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    response = client.invoke_model(
        modelId=BEDROCK_INFERENCE_PROFILE,
        body=json.dumps(body),
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    # Concatenate any text blocks. Bedrock returns content blocks as plain
    # dicts (not Pydantic objects), so use dict access.
    return "".join(
        b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"
    )


def _parse_response(text: str) -> dict:
    """Extract the first JSON object from the response text; tolerate fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    # Find the first balanced { ... } block.
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _quote_is_verbatim(quote: str, source: str) -> bool:
    """Quote may include ellipses. Each non-ellipsis fragment, normalized for
    whitespace, must be a substring of source (also whitespace-normalized)."""
    if not quote:
        return False
    src_norm = re.sub(r"\s+", " ", source).strip().lower()
    # Split on ellipsis variants.
    parts = re.split(r"\.{2,}|…", quote)
    for p in parts:
        p_norm = re.sub(r"\s+", " ", p).strip().lower()
        if not p_norm:
            continue
        if p_norm not in src_norm:
            return False
    return True
