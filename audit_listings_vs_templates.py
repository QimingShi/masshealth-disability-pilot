"""Audit every per-listing JSON against its templates/listings/*.docx counterpart.

The .docx in templates/listings/ are the UMass DES review forms — verbatim
transcription of SSA Blue Book listings. They're our gold standard.

For each listing this script extracts:
  - Template: intro line + paragraph labels (A/B/C/D...) + sub-items (1/2/3/4)
             + duration language ("3 months", "12 months", "after the injury", etc.)
             + cross-references (11.00G3, 1.00B, etc.)
  - JSON:   per-listing rule_json structure under
            rule_json.children[].path == "ALTERNATIVES" (with children A/B/C...)
            Or sometimes a direct OR-of-paragraphs without an ALTERNATIVES wrapper.
            Plus precondition node (cross-ref text), summary, version.

Then compares and flags discrepancies:
  - Paragraph count mismatch (template has 4, JSON has 2)
  - Duration in different location (template says "in A, B, or C despite ...",
    JSON puts it on A only)
  - Duration entirely missing (JSON has none, template has one)
  - Sub-item count mismatch (template B.1-B.4, JSON B has one collapsed leaf)
  - "Post-injury persistence" vs "treatment adherence" mismatch

Outputs:
  - audit_report.md: per-listing findings, sorted by severity
  - audit_summary.csv: tabular summary (code, title, severity, issues)
  - exit code 0 if no issues, 1 if any High-severity findings

Usage:
  python audit_listings_vs_templates.py
  python audit_listings_vs_templates.py --code 11.14    # single listing
  python audit_listings_vs_templates.py --body-system neurological
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates" / "listings"
LISTINGS_DIR = (HERE / "SSA JSON" / "disability-eval-listings"
                / "disability-eval" / "data" / "listings")


# ---------------------------------------------------------------------------
# Template parsing
# ---------------------------------------------------------------------------

# Match a paragraph marker at line start, e.g. "____A.", "A.", "____B."
PARA_MARK_RE = re.compile(r"^[_]*([A-D])\.\s")
# Match a sub-item marker e.g. "____1.", "1.", "____2."
SUB_MARK_RE = re.compile(r"^\s*[_]*(\d)\.\s")
# Look for duration / treatment-adherence language. SSA phrases this
# many different ways; normalize the spaces (templates have non-breaking
# spaces + tab artifacts from docx round-trip) before matching.
DURATION_TREATMENT_RE = re.compile(
    r"despite\s+adherence\s+to\s+prescribed\s+treatment\s+for\s+at\s+least\s+(\d+)\s+months?",
    re.IGNORECASE)
DURATION_POSTINJURY_RE = re.compile(
    r"persisting\s+for\s+at\s+least\s+(\d+)\s+consecutive\s+months?\s+after\s+the\s+injury",
    re.IGNORECASE)
# Generic "lasting / expected to last 12 months" — common in 1.xx, 3.xx,
# 4.xx. Templates use varied wordings ("lasting", "has lasted", "is expected
# to last", "have lasted, or are expected to last, for a continuous period
# of at least 12 months"). Match the duration token regardless of phrasing
# noise — if any "lasted" or "expected to last" appears within ~40 chars
# before "(at least )? N months", count it.
DURATION_LASTING_RE = re.compile(
    r"(?:lasting|has\s+lasted|have\s+lasted|expected\s+to\s+last)"
    r"[^.]{0,120}?"   # allow intervening clauses (commas / "for a continuous period of" / etc.)
    r"(?:at\s+least\s+)?(\d+)\s+(?:consecutive\s+)?months?",
    re.IGNORECASE | re.DOTALL)
DURATION_GENERIC_RE = re.compile(
    r"(\d+)\s+(?:consecutive\s+)?months?",
    re.IGNORECASE)
CROSSREF_RE = re.compile(r"\(?see\s+(\d+\.\d+[A-Z]?\d*[a-z]?(?:\([ivx]+\))?)\)?",
                         re.IGNORECASE)
CHAR_BY_RE = re.compile(r"characterized\s+by\s+([A-Z](?:\s*,\s*[A-Z])*\s*(?:,?\s*or\s+[A-Z])?)",
                        re.IGNORECASE)


def extract_template(docx_path: Path) -> dict | None:
    """Read a .docx and return parsed structure or None on failure."""
    try:
        from docx import Document
    except ImportError:
        return None
    try:
        doc = Document(docx_path)
    except Exception as e:
        return {"error": str(e)}

    paragraphs: list[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        # Skip the standard form footers/scoring rubrics
        if (text.startswith("CIRCLE") or "Meets" in text and "Does not" in text
                or "Disability Reviewer #" in text or "OBJECTIVE MEDICAL" in text
                or set(text) == {"_"} or text.startswith("_____")):
            continue
        paragraphs.append(text)

    # Find "characterized by ..." line
    char_paras = "(unknown)"
    for t in paragraphs:
        m = CHAR_BY_RE.search(t)
        if m:
            char_paras = m.group(1).strip()
            break

    # Walk paragraphs collecting top-level A/B/C/D and their sub-items
    para_letters: list[str] = []
    sub_items: dict[str, list[str]] = defaultdict(list)
    current_para: str | None = None
    for t in paragraphs:
        m = PARA_MARK_RE.match(t)
        if m:
            current_para = m.group(1)
            if current_para not in para_letters:
                para_letters.append(current_para)
            continue
        m = SUB_MARK_RE.match(t)
        if m and current_para is not None:
            sub_items[current_para].append(m.group(1))

    # Duration language — locate. Normalize whitespace because docx
    # round-trips inject tabs and non-breaking spaces inside phrases.
    full_text = re.sub(r"\s+", " ", "\n".join(paragraphs))
    treatment_match  = DURATION_TREATMENT_RE.search(full_text)
    postinjury_match = DURATION_POSTINJURY_RE.search(full_text)
    lasting_match    = DURATION_LASTING_RE.search(full_text)
    duration_info = None
    if treatment_match:
        duration_info = {
            "type": "treatment_adherence",
            "months": int(treatment_match.group(1)),
            "raw":    treatment_match.group(0),
        }
    elif postinjury_match:
        duration_info = {
            "type": "post_injury_persistence",
            "months": int(postinjury_match.group(1)),
            "raw":    postinjury_match.group(0),
        }
    elif lasting_match:
        duration_info = {
            "type": "expected_duration",
            "months": int(lasting_match.group(1)),
            "raw":    lasting_match.group(0),
        }

    # Cross-references — list distinct ones
    crossrefs = sorted(set(m.group(1) for m in CROSSREF_RE.finditer(full_text)))

    return {
        "characterized_by": char_paras,
        "paragraph_letters": para_letters,
        "sub_items": dict(sub_items),
        "duration_info": duration_info,
        "crossrefs": crossrefs,
        "raw_paragraphs": paragraphs,
    }


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def walk_paths(node: dict, depth: int = 0):
    """Yield (path, criterion, depth) for every node in the rule_json tree."""
    if not isinstance(node, dict):
        return
    yield node.get("path", "?"), node.get("criterion", ""), depth
    for child in node.get("children") or []:
        yield from walk_paths(child, depth + 1)


def extract_json(json_path: Path) -> dict | None:
    """Read per-listing JSON and return parsed structure or None on failure."""
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    rule = d.get("rule_json") or {}
    # Find top-level "ALTERNATIVES" node OR direct OR-of-paragraphs
    top_paragraphs: list[str] = []
    sub_items: dict[str, list[str]] = defaultdict(list)
    durations: list[dict] = []
    has_precondition = False

    for path, criterion, depth in walk_paths(rule):
        if path == "PRECONDITION":
            has_precondition = True
        # Top-level paragraph: single letter A/B/C/D at ANY depth (some JSONs
        # have ROOT > A/B/C/D flat, others have ROOT > ALTERNATIVES > A/B/C/D).
        # Exclude PRECONDITION which is a separate concept.
        if re.fullmatch(r"[A-D]", path or "") and path != "PRECONDITION":
            if path not in top_paragraphs:
                top_paragraphs.append(path)
        # Match B.1 / C.2 / B.2.a etc.
        m = re.fullmatch(r"([A-D])\.(\d)", path or "")
        if m:
            sub_items[m.group(1)].append(m.group(2))

    # Walk the raw tree once more for duration fields
    def _find_durations(node):
        if not isinstance(node, dict):
            return
        if "duration_months_required" in node:
            durations.append({
                "path": node.get("path"),
                "months": node["duration_months_required"],
                "basis": node.get("duration_basis", "treatment_adherence"),
            })
        for c in node.get("children") or []:
            _find_durations(c)

    _find_durations(rule)

    # Look for hints in the raw JSON text — fallback for un-fielded durations
    raw_text = json.dumps(d, ensure_ascii=False)
    has_3mo_text = "3 months" in raw_text or "≥3 months" in raw_text or ">=3 months" in raw_text

    return {
        "code": d.get("code"),
        "title": d.get("title"),
        "version": d.get("version"),
        "summary": d.get("summary", ""),
        "top_paragraphs": top_paragraphs,
        "sub_items": dict(sub_items),
        "durations": durations,
        "has_precondition": has_precondition,
        "has_3mo_text_anywhere": has_3mo_text,
    }


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}


def diff_listing(code: str, template: dict, jdata: dict) -> list[dict]:
    """Return a list of findings (each a dict with severity, kind, detail)."""
    findings: list[dict] = []
    if "error" in template:
        findings.append({"severity": "HIGH", "kind": "TEMPLATE_READ_FAIL",
                         "detail": template["error"]})
        return findings
    if "error" in jdata:
        findings.append({"severity": "HIGH", "kind": "JSON_READ_FAIL",
                         "detail": jdata["error"]})
        return findings

    t_paras = set(template["paragraph_letters"])
    j_paras = set(jdata["top_paragraphs"])

    # Paragraph count / labels mismatch (highest severity)
    if t_paras != j_paras:
        findings.append({
            "severity": "HIGH",
            "kind": "PARAGRAPH_SET_MISMATCH",
            "detail": (f"template has {sorted(t_paras) or 'none'}, "
                       f"JSON has {sorted(j_paras) or 'none'}"),
        })

    # Sub-item count mismatch per shared paragraph
    for p in (t_paras & j_paras):
        t_subs = set(template["sub_items"].get(p, []))
        j_subs_raw = jdata["sub_items"].get(p, [])
        j_subs = set(j_subs_raw)
        # Allow JSON's collapsed enumeration if its B.2 has 4 children (a-d pattern)
        if not t_subs and not j_subs:
            continue
        if t_subs and not j_subs:
            findings.append({
                "severity": "HIGH",
                "kind": f"PARAGRAPH_{p}_SUBITEMS_MISSING",
                "detail": (f"template has {sorted(t_subs)} sub-items, "
                           f"JSON paragraph {p} has none enumerated"),
            })
        elif t_subs != j_subs:
            findings.append({
                "severity": "MEDIUM",
                "kind": f"PARAGRAPH_{p}_SUBITEM_DIFF",
                "detail": (f"template has {sorted(t_subs)}, "
                           f"JSON has {sorted(j_subs)}"),
            })

    # Duration analysis
    t_dur = template["duration_info"]
    j_durs = jdata["durations"]
    j_has_3mo = jdata["has_3mo_text_anywhere"]

    if t_dur and not j_durs:
        findings.append({
            "severity": "HIGH",
            "kind": "DURATION_MISSING_IN_JSON",
            "detail": (f"template requires {t_dur['type']} of {t_dur['months']}mo; "
                       f"JSON has no duration field anywhere"),
        })
    elif not t_dur and j_durs:
        # JSON has duration but template doesn't — possible spurious clause
        for jd in j_durs:
            # Skip if it's the "post_injury_persistence" basis on a TBI-like
            # listing — but that should only appear when the template also
            # has the post-injury language, which we already checked.
            findings.append({
                "severity": "HIGH",
                "kind": "DURATION_SPURIOUS",
                "detail": (f"JSON has duration_months_required={jd['months']} "
                           f"on path={jd['path']} (basis={jd['basis']}), but "
                           f"template has no duration language"),
            })
    elif t_dur and j_durs:
        t_basis = t_dur["type"]
        j_bases = {d["basis"] for d in j_durs}
        if t_basis not in j_bases:
            findings.append({
                "severity": "MEDIUM",
                "kind": "DURATION_BASIS_MISMATCH",
                "detail": (f"template basis={t_basis}, JSON basis(es)={sorted(j_bases)}"),
            })
        # Check the duration is applied at the right scope (precondition vs paragraph)
        # If treatment adherence — usually a precondition (applies to all paragraphs)
        # If post-injury persistence — usually per-paragraph
        if t_basis == "treatment_adherence":
            on_precond = any(d["path"] == "PRECONDITION" for d in j_durs)
            if not on_precond:
                findings.append({
                    "severity": "MEDIUM",
                    "kind": "DURATION_SHOULD_BE_PRECONDITION",
                    "detail": (f"template applies '{t_basis}' uniformly via "
                               f"'characterized by ... despite adherence', but "
                               f"JSON has duration on paragraph-level path(s) "
                               f"{[d['path'] for d in j_durs]} only"),
                })

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filename_to_code(name: str) -> str | None:
    """Extract listing code from a docx filename like '11.14 Peripheral Neuropathy.docx'."""
    m = re.match(r"^(\d+\.\d+)\s", name)
    return m.group(1) if m else None


def find_json_for_code(code: str) -> Path | None:
    matches = list(LISTINGS_DIR.rglob(f"{code}.json"))
    return matches[0] if matches else None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--code", help="Audit only this listing code (e.g. 11.14)")
    p.add_argument("--body-system", help="Audit only this body system subfolder")
    p.add_argument("--report", default="audit_report.md",
                   help="Output markdown report path")
    p.add_argument("--csv", default="audit_summary.csv",
                   help="Output CSV summary path")
    args = p.parse_args(argv[1:])

    if not TEMPLATES_DIR.exists():
        print(f"ERROR: {TEMPLATES_DIR} does not exist", file=sys.stderr)
        return 2

    # Gather templates
    templates = sorted(TEMPLATES_DIR.glob("*.docx"))
    if args.code:
        templates = [t for t in templates if filename_to_code(t.name) == args.code]

    print(f"Auditing {len(templates)} listing template(s)...")
    rows = []          # CSV summary
    sections = {}      # markdown sections: code -> (title, list of findings)

    for t in templates:
        code = filename_to_code(t.name)
        if code is None:
            continue
        json_path = find_json_for_code(code)
        if json_path is None:
            sections[code] = ("(no JSON found)",
                              [{"severity": "HIGH", "kind": "JSON_MISSING",
                                "detail": f"no {code}.json in listings/ subtree"}])
            rows.append((code, "?", "HIGH", "JSON_MISSING", "no JSON file"))
            continue
        if args.body_system and args.body_system not in str(json_path):
            continue

        template = extract_template(t) or {"error": "extract returned None"}
        jdata = extract_json(json_path) or {"error": "extract returned None"}
        findings = diff_listing(code, template, jdata)
        title = jdata.get("title", "?") if "error" not in jdata else "(json error)"
        sections[code] = (title, findings)

        if not findings:
            rows.append((code, title, "OK", "—", "matches template"))
        else:
            # severity for CSV: max severity across findings
            severities = sorted({f["severity"] for f in findings},
                                key=SEVERITY_ORDER.get)
            worst = severities[0]
            kinds = ", ".join(sorted({f["kind"] for f in findings}))
            details_summary = "; ".join(f["detail"] for f in findings)
            rows.append((code, title, worst, kinds, details_summary))

    # Write CSV
    csv_path = HERE / args.csv
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("code,title,severity,kinds,details\n")
        for code, title, sev, kinds, detail in rows:
            # naive CSV escape
            def esc(s):
                s = str(s).replace('"', "''")
                return f'"{s}"' if "," in s or '"' in s else s
            f.write(",".join(esc(x) for x in
                             (code, title, sev, kinds, detail)) + "\n")
    print(f"  CSV summary -> {csv_path.relative_to(HERE)}")

    # Write markdown report — grouped by severity then by code
    md_path = HERE / args.report
    by_sev = defaultdict(list)
    for code, (title, findings) in sections.items():
        if not findings:
            by_sev["OK"].append((code, title, findings))
        else:
            worst = sorted({f["severity"] for f in findings},
                           key=SEVERITY_ORDER.get)[0]
            by_sev[worst].append((code, title, findings))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SSA listing JSON-vs-template audit\n\n")
        f.write(f"- Templates checked: {len(templates)}\n")
        f.write(f"- Listings with findings: {sum(1 for _, fs in sections.values() if fs)}\n")
        f.write(f"- HIGH: {len(by_sev['HIGH'])}  "
                f"MEDIUM: {len(by_sev['MEDIUM'])}  "
                f"LOW: {len(by_sev['LOW'])}  "
                f"OK: {len(by_sev['OK'])}\n\n")
        for sev in ("HIGH", "MEDIUM", "LOW"):
            items = by_sev.get(sev, [])
            if not items:
                continue
            f.write(f"## {sev} ({len(items)} listings)\n\n")
            for code, title, findings in sorted(items):
                f.write(f"### {code} — {title}\n\n")
                for finding in findings:
                    f.write(f"- **{finding['kind']}**: {finding['detail']}\n")
                f.write("\n")
        if by_sev.get("OK"):
            f.write(f"## OK ({len(by_sev['OK'])} listings)\n\n")
            f.write("Listings whose JSON encoding matches the template:\n\n")
            for code, title, _ in sorted(by_sev["OK"]):
                f.write(f"- {code} — {title}\n")
    print(f"  markdown report -> {md_path.relative_to(HERE)}")

    # Print short summary to stdout
    print(f"\nSummary: HIGH {len(by_sev['HIGH'])}  "
          f"MEDIUM {len(by_sev['MEDIUM'])}  "
          f"LOW {len(by_sev['LOW'])}  "
          f"OK {len(by_sev['OK'])}")

    return 0 if not by_sev.get("HIGH") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
