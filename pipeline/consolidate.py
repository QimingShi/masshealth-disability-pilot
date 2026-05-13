"""Walk the rule_json tree bottom-up applying 3-valued AND/OR logic.

Three-valued because "insufficient" is its own state — we MUST NOT collapse it
into "not met" silently, since the reviewer needs to know which criteria the
chart couldn't speak to.

  AND: all met -> met
       any not_met -> not_met
       otherwise -> insufficient
  OR:  any met -> met
       all not_met -> not_met
       otherwise -> insufficient
"""
from dataclasses import dataclass

from .evaluate import LeafResult


@dataclass
class NodeVerdict:
    path: str
    logic: str | None      # "AND" / "OR" / None (leaf)
    verdict: str           # "met" | "not_met" | "insufficient"
    criterion: str | None
    children: list["NodeVerdict"]


def consolidate(rule_json: dict, leaf_results: dict[str, LeafResult]) -> NodeVerdict:
    """Walk rule_json, returning the root NodeVerdict with full subtree state."""
    return _walk(rule_json, leaf_results)


def _walk(node: dict, leaf_results: dict[str, LeafResult]) -> NodeVerdict:
    if "children" not in node:
        lr = leaf_results.get(node.get("path", ""))
        return NodeVerdict(
            path=node.get("path", "?"),
            logic=None,
            verdict=lr.verdict if lr else "insufficient",
            criterion=node.get("criterion"),
            children=[],
        )

    child_verdicts = [_walk(c, leaf_results) for c in node["children"]]
    logic = node.get("logic", "AND").upper()
    states = [cv.verdict for cv in child_verdicts]

    if logic == "AND":
        if all(s == "met" for s in states):
            verdict = "met"
        elif any(s == "not_met" for s in states):
            verdict = "not_met"
        else:
            verdict = "insufficient"
    elif logic == "OR":
        if any(s == "met" for s in states):
            verdict = "met"
        elif all(s == "not_met" for s in states):
            verdict = "not_met"
        else:
            verdict = "insufficient"
    else:
        verdict = "insufficient"

    return NodeVerdict(
        path=node.get("path", "?"),
        logic=logic,
        verdict=verdict,
        criterion=node.get("criterion"),
        children=child_verdicts,
    )


# Map our 3-valued verdict to the SSA form's "Meets / Equals / Does not meet/equal" trichotomy.
def form_verdict(root: NodeVerdict) -> str:
    if root.verdict == "met":
        return "Meets"
    if root.verdict == "not_met":
        return "Does not meet/equal"
    return "Insufficient evidence (review chart)"
