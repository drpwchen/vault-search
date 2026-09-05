"""Tests for rerank(rank_key=...) — the fix for the RRF-as-similarity bug (#5).

/api/similar and vault_similar rank by Reciprocal Rank Fusion, whose score is
bounded by 2/RRF_K, about 0.03. Writing that number into 'similarity' made every
client threshold calibrated for 0-1 cosine reject the whole list. These tests pin
the three properties the fix has to hold: 'similarity' reports the real cosine
under rerank's weighting, the fused ordering survives, and the ranking field
never leaks into the result.

Plain stdlib — run directly (`python server/tests/test_similarity_reporting.py`)
or under pytest. No lancedb/ollama import; a throwaway VAULT_PATH is injected
because scoring imports config.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

from scoring import rerank  # noqa: E402

RRF_K = 60


def fused(candidates: list[tuple[str, float]]) -> list[dict]:
    """Build what /api/similar hands to rerank: the real cosine in 'similarity',
    the RRF score in 'fused'. Rank order = list order."""
    return [
        {"note": note, "folder": "", "mtime": None,
         "similarity": cosine,
         "fused": 1.0 / (RRF_K + rank)}
        for rank, (note, cosine) in enumerate(candidates)
    ]


def test_similarity_is_the_cosine_not_the_rrf_score():
    """The bug: a top-ranked pair scored 0.0167, so nothing ever cleared 0.5."""
    results = fused([("A", 0.82), ("B", 0.61)])
    assert max(r["fused"] for r in results) < 0.034  # RRF ceiling

    rerank(results, boost_recent=False, rank_key="fused")

    assert [r["similarity"] for r in results] == [0.82, 0.61]
    assert all(r["similarity"] >= 0.5 for r in results)


def test_fused_ordering_survives():
    """Graph fusion is the point of the endpoint: a note the vector search ranked
    lower can still lead, and reporting the cosine must not resort the list."""
    results = rerank(fused([("graph-led", 0.55), ("closer", 0.91)]),
                     boost_recent=False, rank_key="fused")

    assert [r["note"] for r in results] == ["graph-led", "closer"]
    assert [r["similarity"] for r in results] == [0.55, 0.91]


def test_ranking_field_is_consumed():
    """Neither the fused score nor the sort scratch field may reach the client."""
    results = rerank(fused([("A", 0.8)]), boost_recent=False, rank_key="fused")

    assert "fused" not in results[0]
    assert "_rank_score" not in results[0]


def test_weights_apply_to_the_cosine_and_to_the_ordering():
    """Path weight has to move both numbers: the reported cosine, as it does on a
    plain vector search, and the fused score that decides the order."""
    import scoring
    saved = dict(scoring.PATH_WEIGHTS)
    scoring.PATH_WEIGHTS.update({"52Medicine": 1.2, "89Archived": 0.5})
    try:
        results = fused([("buried", 0.80), ("boosted", 0.80)])
        results[0]["folder"] = "89Archived"
        results[1]["folder"] = "52Medicine"
        rerank(results, boost_recent=False, rank_key="fused")
    finally:
        scoring.PATH_WEIGHTS.clear()
        scoring.PATH_WEIGHTS.update(saved)

    by_note = {r["note"]: r["similarity"] for r in results}
    assert by_note["boosted"] == 0.96   # 0.80 * 1.2
    assert by_note["buried"] == 0.40    # 0.80 * 0.5
    # 'buried' led the fused ranking; the 0.5 path weight has to overturn that.
    assert [r["note"] for r in results] == ["boosted", "buried"]


def test_plain_rerank_is_unchanged():
    """Without rank_key, /api/search keeps ordering by the weighted similarity."""
    results = rerank([
        {"note": "low", "folder": "", "mtime": None, "similarity": 0.30},
        {"note": "high", "folder": "", "mtime": None, "similarity": 0.90},
    ], boost_recent=False)

    assert [r["note"] for r in results] == ["high", "low"]
    assert [r["similarity"] for r in results] == [0.90, 0.30]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
