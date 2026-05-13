# MassHealth Disability Reviewer Assistant — Pilot

An AI tool that matches a member's medical record packet against SSA disability
listings (Blue Book) and produces an evidence-cited summary form for the
disability reviewer. The reviewer remains the decision-maker; the AI is a
decision-support tool.

> ⚠️ **Pilot / research code.** Not production-ready. The data in `data/`
> is **synthetic** — entirely fabricated for demonstration. Do not use this
> code on real patient data without first deploying it inside a HIPAA-compliant
> environment with appropriate Business Associate Agreements.

## What it does

1. Reads a chunked medical record (`data/chunks.json`)
2. Identifies **candidate SSA listings** by combining three signals:
   - Allegations → listing summary similarity (semantic embedding)
   - ICD-10 codes from the chart → SSA body system filter
   - Keyword and synonym hits in chart text against each listing's leaf criteria
3. For each candidate, walks the listing's AND/OR criterion tree and asks
   Claude to classify each leaf as **met / not met / insufficient evidence**,
   citing verbatim chart quotes
4. Consolidates the tree with 3-valued logic and produces a populated
   **Matched_Listing form** per candidate listing, in markdown and HTML
   (HTML has clickable citations that open the source PDF at the cited page)

## Demo output

Running on the included synthetic data (a fabricated metastatic colorectal
cancer case) produces:

- `output/demo-synthetic-001/13.18.md` and `.html` — listing 13.18 marked
  **Meets** with citations to the CT impression (liver metastasis) and the
  pathology report
- Several rule-out forms for other cancer listings (13.17, 13.10, 13.24, 13.02)
  that the candidate stage surfaced but the deep evaluation correctly rejects

Open the HTML in Edge or Chrome to see clickable citations.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your Anthropic API key for real LLM evaluation
#    PowerShell:
$env:ANTHROPIC_API_KEY = "sk-ant-..."

#    Or use mock mode (no API key required) for testing the pipeline:
$env:MOCK_EVAL = "1"

# 3. Run on the included synthetic data
python run.py

# 4. Or run on a different chunks file
python run.py path/to/your/chunks.json
```

Outputs land in `output/<case_id>/`. The `case_id` is read from the chunks file.

## Project layout

```
.
├── data/
│   └── chunks.json              # SYNTHETIC demo chunks (tracked)
├── pipeline/
│   ├── chunks.py                # Loaders + ICD extraction + body-system map
│   ├── embed.py                 # sentence-transformers + cosine
│   ├── candidates.py            # UNION of allegation/ICD/keyword signals
│   ├── retrieve.py              # 3-variant per-leaf retrieval
│   ├── evaluate.py              # Claude per-leaf eval with citation guardrails
│   ├── mock_eval.py             # Canned responses for no-API testing
│   ├── consolidate.py           # AND/OR tree walk, 3-valued logic
│   ├── output.py                # Renders markdown + HTML Matched_Listing form
│   └── ingest_textract.py       # SKETCH: chunk_by_layout for Textract output
├── SSA JSON/                    # 120 SSA listing JSONs (public regulation data)
├── _pdf_survey.py               # Utility: survey PDF structure
├── _pdf_render.py               # Utility: render PDF pages to PNG
├── _dry_run.py                  # Exercise pipeline without LLM
├── run.py                       # End-to-end orchestrator
├── requirements.txt
└── README.md
```

## Architecture notes

**Chunks have rich metadata**: `chunk_id`, `doc_id`, `doc_type`, `section`,
`page_start/end`, `encounter_date`. Citations resolve to a human-readable form
(*"CT Abdomen/Pelvis — Impression, 2099-03-10, p.7"*) and a clickable PDF link
when the source PDF is available.

**Citation guardrails are enforced server-side**: every chunk_id Claude cites
must be in the input set; every quote must be a verbatim substring of that
chunk. Hallucinated citations cause the leaf to be downgraded to "insufficient
evidence" rather than silently included.

**Three-valued verdicts**: leaves and internal nodes carry one of `met`,
`not_met`, or `insufficient`. Insufficient is never collapsed into not_met —
the reviewer must see what couldn't be determined from the chart.

**OCR is deliberately mock**: the included data is hand-curated chunks. Real
OCR via AWS Textract is the next integration; a template
(`pipeline/ingest_textract.py`) is ready for the `chunk_by_layout()` step
once Textract output is available.

## Roadmap

- [ ] Real OCR pipeline via AWS Textract (with bounding boxes for tier-2
      citation highlighting)
- [ ] Annotated source-PDF generation (PyMuPDF highlight annotations at
      cited bboxes) so HTML citations open a pre-highlighted page
- [ ] Numeric criterion checker (FEV1 thresholds, lab values, etc.)
- [ ] Duration enforcement (`duration_months_required` against encounter dates)
- [ ] Candidate-scoring calibration against ≥5 gold-standard cases
- [ ] Side-by-side reviewer web UI (form + embedded PDF.js viewer)

## License

Pilot code for institutional research use. SSA listings under `SSA JSON/` are
sourced from public regulation data and remain in the public domain.
