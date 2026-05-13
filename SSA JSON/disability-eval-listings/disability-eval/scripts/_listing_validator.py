"""Pure-Python validator for SSA listing JSON files.

This module has zero DB dependencies — it imports only stdlib. Both
seed_listings.py (DB loader) and validate_listings.py (CI check) import the
same validator from here, so a JSON document that passes one passes the other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

VALID_BODY_SYSTEMS = {
    "musculoskeletal", "special_senses", "respiratory", "cardiovascular",
    "digestive", "genitourinary", "hematological", "skin", "endocrine",
    "congenital_multisystem", "neurological", "mental", "cancer", "immune",
}


class ListingValidationError(ValueError):
    """Raised when a listing JSON file violates the schema contract."""


def _validate_node(node: Any, path_prefix: str, seen_paths: set[str]) -> None:
    """Recursively validate a rule_json node.

    Rules:
      - every node has a `path`
      - paths are unique within a listing
      - internal nodes have `logic` (AND/OR) and `children` (non-empty list)
      - leaves have `criterion` and at least one item in `keywords`
      - `duration_months_required`, when present, is a positive int
    """
    if not isinstance(node, dict):
        raise ListingValidationError(
            f"node at {path_prefix} is not an object: got {type(node).__name__}"
        )

    if "path" not in node:
        raise ListingValidationError(f"node at {path_prefix} is missing 'path'")
    path = node["path"]
    if path in seen_paths:
        raise ListingValidationError(f"duplicate path '{path}'")
    seen_paths.add(path)

    has_children = "children" in node

    if has_children:
        if "logic" not in node:
            raise ListingValidationError(
                f"internal node {path} has 'children' but no 'logic'"
            )
        if node["logic"] not in {"AND", "OR"}:
            raise ListingValidationError(
                f"node {path} has invalid logic '{node['logic']}' "
                f"(expected AND or OR)"
            )
        if not isinstance(node["children"], list) or not node["children"]:
            raise ListingValidationError(
                f"internal node {path} has empty or non-list 'children'"
            )
        for child in node["children"]:
            _validate_node(child, f"{path_prefix}>{path}", seen_paths)
    else:
        # Leaf node
        if "criterion" not in node:
            raise ListingValidationError(
                f"leaf node {path} is missing 'criterion'"
            )
        if "keywords" not in node:
            raise ListingValidationError(
                f"leaf node {path} is missing 'keywords'"
            )
        if not isinstance(node["keywords"], list) or not node["keywords"]:
            raise ListingValidationError(
                f"leaf node {path} has empty or non-list 'keywords'"
            )

    if "duration_months_required" in node:
        dur = node["duration_months_required"]
        if not isinstance(dur, int) or dur <= 0:
            raise ListingValidationError(
                f"node {path} has invalid duration_months_required: {dur}"
            )


def validate_listing_doc(doc: Any, source_path: Path) -> None:
    """Validate a parsed listing JSON document against the schema contract.

    Raises:
        ListingValidationError on any violation. Does not return on error.
    """
    if not isinstance(doc, dict):
        raise ListingValidationError(
            f"{source_path.name}: top-level value is not an object"
        )

    required = ("code", "title", "body_system", "summary", "rule_json")
    for field in required:
        if field not in doc:
            raise ListingValidationError(
                f"{source_path.name}: missing required field '{field}'"
            )

    if doc["body_system"] not in VALID_BODY_SYSTEMS:
        raise ListingValidationError(
            f"{source_path.name}: invalid body_system '{doc['body_system']}' "
            f"(expected one of {sorted(VALID_BODY_SYSTEMS)})"
        )

    if "synonyms" in doc:
        if not isinstance(doc["synonyms"], dict):
            raise ListingValidationError(
                f"{source_path.name}: 'synonyms' must be an object, "
                f"got {type(doc['synonyms']).__name__}"
            )
        for canonical, variants in doc["synonyms"].items():
            if not isinstance(variants, list):
                raise ListingValidationError(
                    f"{source_path.name}: synonyms['{canonical}'] is not a list"
                )

    seen_paths: set[str] = set()
    _validate_node(doc["rule_json"], doc["code"], seen_paths)
