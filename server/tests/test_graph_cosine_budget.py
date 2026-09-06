"""The graph-only similarity backfill must not starve the notes it exists for.

/api/similar reports a real cosine for candidates that arrived through the wiki
link graph instead of the vector search. That number comes from one prefiltered
vector query, and the query ranks CHUNKS: a note split into hundreds of them can
fill the row budget on its own, and whichever notes sit furthest away fall off
the end. They then report 0.0 — under every client threshold, so the Obsidian
panel (which hides anything below 0.5) drops rows the endpoint did return.

Measured on a real vault: 4 of 40 graph candidates came back with no cosine,
each of them one of the four least similar. Chunks per note averaged 20, which
was exactly the old per-note budget.

Run directly (`python server/tests/test_graph_cosine_budget.py`) or under pytest.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

try:
    import pandas as pd

    import api_server
except ImportError:  # fastapi / lancedb / pandas not installed
    api_server = None


class FakeTable:
    """Enough of the LanceDB builder chain for _cosine_for_notes.

    `chunks` is [(note, distance)]; a query returns the nearest `limit` of the
    rows whose note is named in the WHERE clause, exactly as a prefiltered
    vector search does.
    """

    def __init__(self, chunks):
        self.chunks = sorted(chunks, key=lambda c: c[1])
        self.queries = []
        self._names = None
        self._limit = None

    def search(self, _vector):
        return self

    def metric(self, _name):
        return self

    def where(self, sql, prefilter=False):
        self._names = set(re.findall(r"'([^']*)'", sql))
        return self

    def limit(self, n):
        self._limit = n
        return self

    def to_pandas(self):
        rows = [{"note": n, "_distance": d}
                for n, d in self.chunks if n in self._names][:self._limit]
        self.queries.append((sorted(self._names), self._limit, len(rows)))
        return pd.DataFrame(rows, columns=["note", "_distance"])


if api_server is not None:

    def test_the_farthest_note_still_gets_its_cosine():
        """One chunk-heavy note used to consume the whole budget by itself."""
        table = FakeTable([("heavy", 0.1)] * 500 + [("far", 0.55)])

        got = api_server._cosine_for_notes(table, [0.0], ["heavy", "far"])

        assert set(got) == {"heavy", "far"}, got
        assert got["far"] == 0.45
        assert len(table.queries) == 2  # first pass cut it, second asked again

    def test_one_pass_when_nothing_is_missing():
        table = FakeTable([("a", 0.2), ("b", 0.4)])

        got = api_server._cosine_for_notes(table, [0.0], ["a", "b"])

        assert got == {"a": 0.8, "b": 0.6}
        assert len(table.queries) == 1

    def test_a_note_absent_from_the_table_does_not_loop():
        """Not every graph node is indexed; retrying those is pure latency."""
        table = FakeTable([("a", 0.2)])

        got = api_server._cosine_for_notes(table, [0.0], ["a", "ghost"])

        assert got == {"a": 0.8}
        assert len(table.queries) == 2  # one retry, then the empty round stops

    def test_no_notes_asks_nothing():
        table = FakeTable([("a", 0.2)])

        assert api_server._cosine_for_notes(table, [0.0], []) == {}
        assert table.queries == []


if __name__ == "__main__":
    if api_server is None:
        print("SKIP test_graph_cosine_budget (api_server dependencies missing)")
    else:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"ok  {name}")
        print("all passed")
