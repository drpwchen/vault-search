"""
Vault Search Scoring — Post-query reranking with path, recency, and relation weighting.

Used by both mcp_server.py and api_server.py to adjust vector search similarity scores.

Tunable constants are at the top of this file.
"""

import math
import re
import time

# --- Path Weighting ---
# Folder name → multiplier (configurable via VAULT_SEARCH_PATH_WEIGHTS).
# Unlisted folders default to 1.00. Empty by default = every folder equal.
from config import PATH_WEIGHTS, ARCHIVE_FOLDER

DEFAULT_PATH_WEIGHT = 1.00

# --- Recency Weighting ---
# Logarithmic decay: weight = 1 / (1 + log1p(age_days / half_life) * scale)
# Floor prevents ancient notes from being completely buried.
RECENCY_HALF_LIFE_DAYS = 365.0  # Reference point for decay speed
RECENCY_SCALE = 0.3             # Controls steepness (higher = steeper)
RECENCY_FLOOR = 0.70            # Minimum recency weight (never below this)


def get_path_weight(folder: str, include_archived: bool = False) -> float:
    """Return path weight multiplier for a folder.

    Args:
        folder: Top-level vault folder name.
        include_archived: If True, treat the configured archive folder as neutral (1.0).
    """
    if include_archived and ARCHIVE_FOLDER and folder == ARCHIVE_FOLDER:
        return DEFAULT_PATH_WEIGHT
    return PATH_WEIGHTS.get(folder, DEFAULT_PATH_WEIGHT)


def get_recency_weight(mtime: float | None, now: float | None = None) -> float:
    """Return recency weight based on file modification time.

    Args:
        mtime: File modification time as epoch float. None → neutral (1.0).
        now: Current time as epoch float. None → time.time().
    """
    if mtime is None:
        return 1.0
    if now is None:
        now = time.time()
    age_days = max((now - mtime) / 86400.0, 0)
    weight = 1.0 / (1.0 + math.log1p(age_days / RECENCY_HALF_LIFE_DAYS) * RECENCY_SCALE)
    return max(weight, RECENCY_FLOOR)


# --- Relation Weighting ---
# Query keyword → relation type mapping for boosting
QUERY_RELATION_KEYWORDS = {
    # Chinese
    "治療": "treats", "處理": "treats", "處置": "treats",
    "原因": "causes", "���成": "causes", "導致": "causes",
    "診斷": "diagnoses", "檢查": "diagnoses", "評估": "diagnoses",
    "併發症": "complication_of", "合併症": "complication_of",
    "症狀": "symptom_of", "表現": "symptom_of",
    "禁忌": "contraindicated_for", "適應症": "indicated_for",
    # English
    "treat": "treats", "treatment": "treats", "manage": "treats",
    "cause": "causes", "etiology": "causes",
    "diagnos": "diagnoses", "assess": "diagnoses",
    "complication": "complication_of",
    "symptom": "symptom_of", "presentation": "symptom_of",
    "contraindic": "contraindicated_for", "indication": "indicated_for",
}
RELATION_BOOST = 1.15  # Multiplier for results matching query's relation intent


def get_relation_weight(
    note_name: str, query: str, graph_relations: list[dict]
) -> float:
    """Return relation weight if the note's entities match query intent."""
    if not graph_relations or not query:
        return 1.0

    # Detect query intent
    query_lower = query.lower()
    target_types = set()
    for keyword, rel_type in QUERY_RELATION_KEYWORDS.items():
        if keyword in query_lower:
            target_types.add(rel_type)

    if not target_types:
        return 1.0

    # Check if note has matching relation types
    for rel in graph_relations:
        if rel.get("source_note") == note_name and rel.get("type") in target_types:
            return RELATION_BOOST

    return 1.0


def rerank(
    results: list[dict],
    boost_recent: bool = True,
    include_archived: bool = False,
    query: str = "",
    graph_relations: list[dict] | None = None,
) -> list[dict]:
    """Rerank search results by applying path and recency weights.

    Modifies results in-place: overwrites 'similarity' with adjusted score,
    adds 'raw_similarity' with original value.

    Args:
        results: List of result dicts with 'similarity', 'folder', and optionally 'mtime'.
        boost_recent: Whether to apply recency weighting.
        include_archived: Whether to treat 89Archived as neutral.

    Returns:
        Results sorted by adjusted similarity descending.
    """
    now = time.time()
    for r in results:
        raw_sim = r.get("similarity", 0) or 0
        r["raw_similarity"] = raw_sim

        pw = get_path_weight(r.get("folder", ""), include_archived)
        rw = get_recency_weight(r.get("mtime"), now) if boost_recent else 1.0
        rlw = get_relation_weight(r.get("note", ""), query, graph_relations or [])

        r["similarity"] = round(raw_sim * pw * rw * rlw, 4)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
