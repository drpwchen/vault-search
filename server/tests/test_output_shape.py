"""Output-shape guards for the vault tools.

Both behaviours here were real divergences found while checking this code
against a running installation on the same index:

  1. `vault_similar` quoted 500 characters of a note while `vault_search`
     quoted 800, because one path built its rows through format_results() and
     the other inline. Same index, same note, two different answers to "how
     much of this note do I get".
  2. The entities block was built by iterating a SET of note names, so its
     order — and, with the 15-entry cap, its CONTENT — changed with
     PYTHONHASHSEED. Two processes on one index answered the same query
     differently.

Run directly (`python server/tests/test_output_shape.py`) or under pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

try:
    import pandas as pd

    import mcp_server
except ImportError:
    mcp_server = None


if mcp_server is not None:

    def test_excerpt_budget_is_one_number_everywhere():
        """format_results must quote exactly EXCERPT_CHARS, not its own limit."""
        long_text = "x" * (mcp_server.EXCERPT_CHARS * 2)
        df = pd.DataFrame([{
            "file": "n.md", "note": "n", "section": "", "folder": "", "tags": "",
            "text": long_text, "mtime": None, "_distance": 0.1,
        }])
        rows = mcp_server.format_results(df, include_excerpt=True)
        assert len(rows[0]["excerpt"]) == mcp_server.EXCERPT_CHARS

    def test_entities_follow_result_order_not_hash_order():
        graph_entities = {
            "A": [{"name": "a", "type": "condition"}],
            "B": [{"name": "b", "type": "condition"}],
            "C": [{"name": "c", "type": "condition"}],
        }
        for order in (["C", "A", "B"], ["B", "C", "A"]):
            out = mcp_server.find_entities_for_notes(graph_entities, order)
            assert [e["note"] for e in out] == order

    def test_entity_cap_keeps_the_top_ranked_notes():
        """The 15-entry cap must cut the tail, not an arbitrary hash slice."""
        graph_entities = {f"N{i}": [{"name": "x", "type": "condition"}] for i in range(20)}
        order = [f"N{i}" for i in range(20)]
        out = mcp_server.find_entities_for_notes(graph_entities, order)
        assert [e["note"] for e in out] == order[:15]

else:
    def test_skipped_deps_missing():
        """Placeholder so the file reports as skipped-by-design, not empty."""
        pass


if __name__ == "__main__":
    if mcp_server is None:
        print("SKIP: dependencies not installed")
    else:
        test_excerpt_budget_is_one_number_everywhere()
        test_entities_follow_result_order_not_hash_order()
        test_entity_cap_keeps_the_top_ranked_notes()
        print("OK: 3 tests passed")
