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


# ---------------------------------------------------------------------------
#  Load-bearing leaf identification
# ---------------------------------------------------------------------------
#
# A leaf is "load-bearing" if its verdict, were it to flip, could change the
# root verdict. Walking the consolidated tree top-down with awareness of each
# node's verdict + logic, we keep only the children whose state drove the
# parent's conclusion:
#
#   Node verdict = met:
#     AND  - all children must be met for AND to be met -> all are load-bearing
#     OR   - any met child satisfies OR -> only met children are load-bearing
#            (the not_met / insufficient siblings are irrelevant - some
#             alternative path already succeeded)
#
#   Node verdict = not_met:
#     AND  - any not_met child kills AND -> only not_met children are
#            load-bearing (met / insufficient siblings didn't drive the
#            decision; the not_met child(ren) did)
#     OR   - all children must be not_met -> all are load-bearing
#
#   Node verdict = insufficient:
#     AND  - no not_met children, >=1 insufficient -> insufficient children
#            are load-bearing (if any flipped to met OR not_met, the AND
#            would resolve)
#     OR   - no met children, >=1 insufficient -> insufficient children
#            are load-bearing (if any flipped to met, the OR would be met)
#
# This is used by pipeline/groundedness.py to compute avg_top_similarity
# over only the leaves that actually drove the verdict, avoiding the
# "OR-pathway drag-down" where alternative branches with low retrieval
# similarity dilute the score even though they were irrelevant to the
# listing's decision.

def load_bearing_leaf_paths(rule_json: dict, root: NodeVerdict) -> set[str]:
    """Return the set of leaf paths whose verdict drove the root decision.

    Args:
        rule_json: the listing's criterion tree (same shape passed to
            consolidate()).
        root: the consolidated NodeVerdict returned by consolidate(). Walks
            it in parallel with rule_json — they have matching shape.

    Returns: set of leaf path strings. May be empty if the tree has no
        leaves at all (shouldn't happen on a real listing).
    """
    return _load_bearing(rule_json, root)


def _load_bearing(node: dict, nv: NodeVerdict) -> set[str]:
    # Leaf: itself is the load-bearing path
    if "children" not in node:
        return {node.get("path", "?")}

    logic = (node.get("logic", "AND") or "AND").upper()
    verdict = nv.verdict
    # Pair each child dict with its consolidated NodeVerdict (parallel shape)
    pairs = list(zip(node["children"], nv.children))

    if verdict == "met":
        if logic == "AND":
            relevant = pairs                              # all required
        else:  # OR
            relevant = [p for p in pairs if p[1].verdict == "met"]
    elif verdict == "not_met":
        if logic == "AND":
            relevant = [p for p in pairs if p[1].verdict == "not_met"]
        else:  # OR
            relevant = pairs                              # all must have failed
    else:  # insufficient
        # The blocking insufficient children — they could have resolved
        # the node either way had they been decided.
        relevant = [p for p in pairs if p[1].verdict == "insufficient"]

    out: set[str] = set()
    for child_node, child_nv in relevant:
        out |= _load_bearing(child_node, child_nv)
    return out
