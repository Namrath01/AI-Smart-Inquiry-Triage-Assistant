"""Canonical taxonomy loading, validation, and classifier-context formatting.

data/taxonomy.json is the single source of truth for the category enum used
throughout the pipeline (classification, routing, fallback scoring). Nothing
in this module hard-codes category names, descriptions, or keywords.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"

REQUIRED_CATEGORY_FIELDS = ("name", "description", "keywords")

# Small stopword set for the description-overlap fallback signal. Kept short
# and generic on purpose -- this is a simple heuristic, not an NLP pipeline.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "such", "that", "the",
    "their", "this", "to", "was", "were", "will", "with", "not", "than",
    "also", "may", "can", "if", "so", "but",
}


class TaxonomyError(ValueError):
    """Raised when data/taxonomy.json is missing, malformed, or invalid."""


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _validate_taxonomy(raw: dict) -> list[dict]:
    if not isinstance(raw, dict) or "categories" not in raw:
        raise TaxonomyError("taxonomy.json must contain a top-level 'categories' list")

    categories = raw["categories"]
    if not isinstance(categories, list) or len(categories) == 0:
        raise TaxonomyError("taxonomy.json 'categories' must be a non-empty list")

    seen_names: set[str] = set()
    for i, category in enumerate(categories):
        if not isinstance(category, dict):
            raise TaxonomyError(f"taxonomy category at index {i} is not an object")

        missing = [f for f in REQUIRED_CATEGORY_FIELDS if f not in category]
        if missing:
            raise TaxonomyError(
                f"taxonomy category at index {i} is missing field(s): {missing}"
            )

        name = category["name"]
        if not isinstance(name, str) or not name.strip():
            raise TaxonomyError(f"taxonomy category at index {i} has an invalid 'name'")
        if name in seen_names:
            raise TaxonomyError(f"duplicate taxonomy category name: {name!r}")
        seen_names.add(name)

        description = category["description"]
        if not isinstance(description, str) or not description.strip():
            raise TaxonomyError(f"category {name!r} has an invalid 'description'")

        keywords = category["keywords"]
        if not isinstance(keywords, list) or len(keywords) == 0:
            raise TaxonomyError(f"category {name!r} has no usable 'keywords'")
        if not all(isinstance(k, str) and k.strip() for k in keywords):
            raise TaxonomyError(f"category {name!r} has empty/invalid keyword entries")

    return categories


@lru_cache(maxsize=1)
def load_taxonomy() -> tuple[dict, ...]:
    """Load and validate data/taxonomy.json. Cached after first successful load."""
    if not TAXONOMY_PATH.exists():
        raise TaxonomyError(f"taxonomy file not found at {TAXONOMY_PATH}")

    with TAXONOMY_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)

    categories = _validate_taxonomy(raw)
    return tuple(categories)


def get_category_names() -> tuple[str, ...]:
    """Canonical category enum, in taxonomy.json order. Reusable by the classifier."""
    return tuple(c["name"] for c in load_taxonomy())


def get_category(name: str) -> dict:
    for c in load_taxonomy():
        if c["name"] == name:
            return c
    raise TaxonomyError(f"unknown category: {name!r}")


def format_taxonomy_for_prompt() -> str:
    """Render the full taxonomy as classifier context (name, description, keywords)."""
    lines = []
    for c in load_taxonomy():
        keywords = ", ".join(c["keywords"])
        lines.append(
            f"- {c['name']}: {c['description']}\n  keywords: {keywords}"
        )
    return "\n".join(lines)


def compute_fallback_evidence(query: str) -> dict[str, dict]:
    """Per-category deterministic evidence used by the classification fallback.

    Returns, for every canonical category, the raw signals the fallback is
    built from -- kept separate so each signal is independently observable
    and testable:

    - keyword_matches: which of the category's keyword phrases appear as a
      substring of the query (case-insensitive).
    - keyword_score: len(keyword_matches) / total keywords for that category,
      in [0, 1].
    - name_match: whether the category name itself appears in the query.
    - description_overlap_score: fraction of the category description's
      (stopword-filtered) tokens that also appear in the query, in [0, 1].

    No combination formula is applied here. Whether/how these signals are
    combined into a single fallback decision is determined by the caller in
    src/main.py, which uses keyword/name hits. This function only makes the
    raw evidence observable and testable.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    query_lower = query.lower()
    query_tokens = _tokenize(query)

    evidence: dict[str, dict] = {}
    for category in load_taxonomy():
        name = category["name"]
        keywords = category["keywords"]

        keyword_matches = [kw for kw in keywords if kw.lower() in query_lower]
        keyword_score = len(keyword_matches) / len(keywords)

        name_match = name.lower() in query_lower

        description_tokens = _tokenize(category["description"])
        if description_tokens:
            overlap = query_tokens & description_tokens
            description_overlap_score = len(overlap) / len(description_tokens)
        else:
            description_overlap_score = 0.0

        evidence[name] = {
            "keyword_matches": keyword_matches,
            "keyword_score": keyword_score,
            "name_match": name_match,
            "description_overlap_score": description_overlap_score,
        }

    return evidence
