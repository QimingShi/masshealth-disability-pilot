"""Mock evaluator for the synthetic CRC demo packet — runs end-to-end without
an Anthropic API key.

Stand-in for the LLM call only. Same response shape as the real call, so
downstream citation guardrails (chunk_id in input set, verbatim quote) still
run against this output and validate. Real LLM calls replace this verbatim.

Responses are hand-crafted for the synthetic data in data/chunks.json. For any
(listing_code, leaf_path) not explicitly handled, returns 'insufficient' so
consolidation correctly marks unrelated candidate listings as non-matching.

Activate with:  PowerShell: $env:MOCK_EVAL = '1'   bash: MOCK_EVAL=1
"""
from typing import Any


# (listing_code, leaf_path) -> mock JSON response
_MOCK_RESPONSES: dict[tuple[str, str], dict[str, Any]] = {
    # === 13.18 Large intestine — the matched listing ===
    ("13.18", "PRECONDITION"): {
        "verdict": "met",
        "rationale": "Surgical pathology of a sigmoid colon biopsy confirms invasive colorectal adenocarcinoma; visit diagnosis code C18.9 corroborates the colorectal site.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Biopsy-confirmed colorectal adenocarcinoma satisfies the pathology precondition.",
            },
            {
                "chunk_id": "synth-onc:p4:VisitDiagnoses:0",
                "quote": "Primary: Colon adenocarcinoma C18.9",
                "relevance": "Clinician-coded primary diagnosis matches the colon site.",
            },
        ],
    },
    ("13.18", "A"): {
        "verdict": "met",
        "rationale": "CT imaging documents two hepatic metastases in the right hepatic lobe (distant spread beyond regional colorectal lymph nodes), corroborated by the clinician-stated primary diagnosis of metastatic colorectal cancer and a markedly elevated CEA.",
        "evidence": [
            {
                "chunk_id": "synth-ct:p7:CTImpression:0",
                "quote": "Two hepatic metastases in the right hepatic lobe, measuring 1.8 cm and 0.9 cm respectively, consistent with metastatic disease beyond the regional lymph nodes",
                "relevance": "Hepatic metastases are distant (M1) disease, beyond regional colorectal lymph nodes.",
            },
            {
                "chunk_id": "synth-onc:p1:Header:0",
                "quote": "Diagnosis/Chief Complaint: metastatic colorectal cancer",
                "relevance": "Provider-stated diagnosis of metastatic disease.",
            },
            {
                "chunk_id": "synth-labs:p6:LabsCEA:0",
                "quote": "CEA 52.3 (H)",
                "relevance": "Elevated tumor marker consistent with metastatic disease burden.",
            },
        ],
    },
    ("13.18", "B"): {
        "verdict": "insufficient",
        "rationale": "Record indicates this is a newly diagnosed presentation receiving first-line chemotherapy; the past medical history explicitly states no prior abdominal surgery, so there has been no total mesorectal excision or low anterior resection from which recurrence could be assessed.",
        "evidence": [
            {
                "chunk_id": "synth-onc:p1:PMH:0",
                "quote": "No prior abdominal surgery.",
                "relevance": "Without a prior resection, the recurrence criterion is not applicable.",
            },
        ],
    },
    ("13.18", "C"): {
        "verdict": "not_met",
        "rationale": "Pathology specifies colorectal adenocarcinoma of sigmoid origin, not squamous cell carcinoma of the anus; the documented histology and primary site exclude this criterion.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Histology is adenocarcinoma, not squamous cell carcinoma; criterion C is for anal SCC.",
            },
        ],
    },
    ("13.18", "D"): {
        "verdict": "insufficient",
        "rationale": "Record describes systemic first-line chemotherapy but contains no explicit statement that the tumor is inoperable or unresectable. Surgical candidacy is not addressed in the available chart material.",
        "evidence": [],
    },

    # === 13.17 Small intestine — wrong primary site ===
    ("13.17", "PRECONDITION"): {
        "verdict": "not_met",
        "rationale": "Pathology localizes the primary cancer to the sigmoid colon; imaging confirms the tumor origin is colorectal. No small-intestine malignancy is documented.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Colorectal origin excludes the small-intestine precondition.",
            },
        ],
    },

    # === 13.10 Breast cancer — site mismatch ===
    ("13.10", "PRECONDITION"): {
        "verdict": "not_met",
        "rationale": "The documented malignancy is colorectal; no breast pathology, breast imaging, or breast cancer diagnosis appears anywhere in the record.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Primary malignancy is colorectal, not mammary.",
            },
        ],
    },

    # === 13.24 Prostate gland — site mismatch ===
    ("13.24", "PRECONDITION"): {
        "verdict": "not_met",
        "rationale": "No prostate malignancy is documented. The primary cancer in this case is colorectal.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Colorectal primary, not prostatic.",
            },
        ],
    },

    # === 13.02 Soft tissue cancers of head and neck — site mismatch ===
    ("13.02", "PRECONDITION"): {
        "verdict": "not_met",
        "rationale": "No head or neck malignancy is described. The primary cancer is colorectal adenocarcinoma.",
        "evidence": [
            {
                "chunk_id": "synth-path:p9:PathologyDx:0",
                "quote": "Invasive, moderately differentiated colorectal adenocarcinoma",
                "relevance": "Colorectal site, not head/neck.",
            },
        ],
    },
}


def mock_evaluate(listing_code: str, leaf_path: str, chunks_by_id: dict) -> dict:
    """Return a mock LLM response. For leaves not explicitly handled, returns
    a default 'insufficient' verdict (correct for unrelated cancer listings
    where the precondition has already returned 'not_met')."""
    key = (listing_code, leaf_path)
    if key in _MOCK_RESPONSES:
        resp = _MOCK_RESPONSES[key]
        # Filter out any cited chunks that aren't actually in the retrieved set
        # for this evaluation — mirrors what the real guardrail would do.
        filtered_evidence = [
            e for e in resp["evidence"]
            if e["chunk_id"] in chunks_by_id
        ]
        return {**resp, "evidence": filtered_evidence}
    return {
        "verdict": "insufficient",
        "rationale": "Mock evaluator has no canned response for this leaf; defaulting to insufficient.",
        "evidence": [],
    }
