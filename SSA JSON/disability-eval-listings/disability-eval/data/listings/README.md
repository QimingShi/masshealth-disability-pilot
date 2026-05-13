# SSA Listing Data Format

Each listing lives in `data/listings/<body_system>/<code>.json` and conforms to
the schema below. The seed loader (`scripts/seed_listings.py`) reads every file
under this directory and calls `materialize_criteria` per listing, which is
idempotent on `(listing_id, path)` — re-running picks up edits without
duplicating rows.

## Top-level schema

```json
{
  "code": "1.15",
  "title": "Disorders of the skeletal spine resulting in compromise of a nerve root(s)",
  "body_system": "musculoskeletal",
  "version": 1,
  "summary": "Short clinical-language summary, used for listing-signature embedding",
  "synonyms": {
    "<canonical term>": ["<synonym>", "<abbreviation>", "..."]
  },
  "rule_json": {
    "logic": "AND" | "OR",
    "path": "ROOT",
    "children": [ ... ]
  }
}
```

### Field rules

- `code` — must match the SSA blue book identifier exactly (e.g. `1.15`, `12.04`).
- `body_system` — one of: `musculoskeletal`, `special_senses`, `respiratory`,
  `cardiovascular`, `digestive`, `genitourinary`, `hematological`, `skin`,
  `endocrine`, `congenital_multisystem`, `neurological`, `mental`, `cancer`,
  `immune`.
- `version` — bump on every breaking edit to the criterion tree.
- `summary` — 1–2 sentences in clinical (not regulatory) language. This text
  is what `identify_candidate_listings` embeds and matches against the case's
  chunks to triage which listings are worth analyzing.
- `synonyms` — per-listing dictionary; see "Synonyms" below.
- `rule_json` — the criterion tree; see "Rule tree" below.

## Rule tree

Every node is one of:

**Internal node** — has children combined by a logic operator:
```json
{
  "logic": "AND" | "OR",
  "path": "ROOT" | "A" | "B.3_OR_B.4" | ...,
  "criterion": "Optional human-readable label for the group",
  "children": [ ... ]
}
```

**Leaf node** — what the LLM actually analyzes:
```json
{
  "path": "A.1",
  "criterion": "Clinical-language statement of what evidence we need",
  "keywords": ["clinical term", "abbreviation", "exam finding"],
  "duration_months_required": 12
}
```

### Path rules

Paths mirror the SSA's letter/number scheme: `ROOT > A > A.1 > A.1.a`. When the
regulation says "1, 2, and either 3 or 4," introduce a synthetic OR node with
a descriptive path like `B.3_OR_B.4`. When a leaf would normally have multiple
options expressed as "a or b," lift those into a child OR node — leaves never
contain logic.

### Criterion text rules

The `criterion` text on leaves is the single most important field for retrieval
quality. Rewrite SSA's prose into the language clinicians actually chart:

- Bad: *"Sign(s) of nerve root irritation, tension, or compression, consistent with compromise of the affected nerve root."*
- Good: *"Signs of nerve root irritation, tension, or compression on physical exam (e.g., positive straight leg raise, Spurling, Lasegue)."*

The bad version retrieves poorly because no clinician writes "consistent with
compromise"; the good version pulls in the chart-shorthand chunks where the
evidence actually lives.

### Keywords

The `keywords` array is appended to the *keywords* query variant (not a search
filter). Aim for 3–8 entries per leaf:

- The exam maneuvers, lab values, or imaging findings a clinician would chart.
- Standard medical abbreviations (MRI, EMG, NCS, MMSE, NYHA III).
- Lay-language variants clinicians sometimes write in PCP notes.

Don't pad — retrieval works better with a tight, high-precision keyword set
than with everything-and-the-kitchen-sink.

### Duration

Set `duration_months_required` on the leaf that owns the durational criterion.
Most listings in 1.00 and 4.00 require 12 months; 12.00 paragraph C requires
24; some surgical-management listings require ongoing/expected duration. The
field is read by the consolidator and surfaced to the reviewer; downstream
date-arithmetic logic can use it to evaluate against onset dates.

## Synonyms

The `synonyms` block is a dict keyed by *canonical term* (whatever the
clinician actually writes), with values that are alternative phrasings,
abbreviations, common misspellings, and lay variants. The retrieval layer's
synonym variant query expands using this map.

```json
"synonyms": {
  "depression":         ["major depressive disorder", "MDD", "depressive disorder", "low mood"],
  "12 months":          ["1 year", "one year", "year-long", "longstanding", "chronic", "for over a year"],
  "MRI":                ["magnetic resonance imaging", "MR imaging", "MR scan"],
  "straight leg raise": ["SLR", "Lasegue", "Lasegue's sign", "Lasegue maneuver"],
  "walker":             ["rollator", "rolling walker", "wheeled walker", "ambulatory aid"],
  "anhedonia":          ["loss of interest", "loss of pleasure", "no interest", "doesn't enjoy"]
}
```

### What goes in synonyms

- **Standard clinical synonyms** — e.g. *radiculopathy* ↔ *nerve root compression*
- **Abbreviations** — e.g. *NYHA III*, *FEV1*, *MMSE*, *EMG*, *NCS*
- **Chart shorthand** — e.g. *SOB* for shortness of breath, *DOE* for dyspnea on exertion
- **Lay-language variants** — what a patient says vs. what's charted (because
  some chart notes quote the patient: *"patient reports feeling down for months"*)
- **Branded vs. generic drug names** — when relevant to the listing

### What does NOT go in synonyms

- Misspellings (covered by embedding similarity in practice)
- OCR-confusion variants (better handled at OCR cleanup time)
- Loose conceptual cousins ("anxiety" is not a synonym for "depression" even
  though they're comorbid — keep them separate so the synonym query stays
  sharp)

### Per-listing not global

Synonyms travel with the listing rather than living in a global file because
the same word can mean different things in different listing contexts. For
example, *"compression"* in 1.15 means nerve root compression; in 4.02 it
might mean a compression stocking. Per-listing scoping prevents cross-domain
contamination.

## Body system choices

Use the SSA body-system numbering for `body_system`:

| number | name |
|---|---|
| 1.00 | musculoskeletal |
| 2.00 | special_senses |
| 3.00 | respiratory |
| 4.00 | cardiovascular |
| 5.00 | digestive |
| 6.00 | genitourinary |
| 7.00 | hematological |
| 8.00 | skin |
| 9.00 | endocrine |
| 10.00 | congenital_multisystem |
| 11.00 | neurological |
| 12.00 | mental |
| 13.00 | cancer |
| 14.00 | immune |

## Validation

Run `python scripts/validate_listings.py data/listings/` before committing —
it checks the JSON parses, that every internal node has children, that every
leaf has criterion + keywords, that paths are unique within a listing, and
that duration values are plausible.
