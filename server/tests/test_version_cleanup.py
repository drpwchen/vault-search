"""Tests for LanceDB old-version cleanup (indexer._cleanup_versions).

Needs lancedb + pyarrow; the whole module is skipped when they are not
installed so the hermetic CI suite stays green. Importing `indexer` requires
VAULT_PATH, so a throwaway one is injected before the import.

Run directly (`python server/tests/test_version_cleanup.py`) or under pytest.
"""

import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    lancedb = None

if lancedb is not None:
    _tmp_vault = tempfile.mkdtemp(prefix="vsearch_test_vault_")
    os.environ.setdefault("VAULT_PATH", _tmp_vault)
    from indexer import _cleanup_versions, _delete_files, VERSION_RETENTION  # noqa: E402

    SCHEMA = pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("file", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
    ])

    def _rows(file_name: str, n: int = 3):
        return [
            {"id": f"{file_name}::{i}", "file": file_name, "vector": [0.1] * 4}
            for i in range(n)
        ]

    def _table_with_history(d):
        db = lancedb.connect(d)
        table = db.create_table("t", data=_rows("a.md"), schema=SCHEMA)
        # Simulate incremental runs: each delete/add is one version snapshot
        for name in ("b.md", "c.md", "d.md", "e.md"):
            table.delete(f"file = '{name}'")
            table.add(_rows(name))
        assert len(table.list_versions()) >= 9
        return table

    def test_cleanup_versions_prunes_old_snapshots():
        with tempfile.TemporaryDirectory() as d:
            table = _table_with_history(d)

            _cleanup_versions(table, older_than=timedelta(0))

            # Only the latest state (+ at most the optimize step itself) remains
            assert len(table.list_versions()) < 9
            assert table.count_rows() == 15  # data intact: 5 files x 3 chunks

    def test_cleanup_versions_keeps_versions_inside_retention_window():
        """The window is the point: a search re-opens the table per query, so a
        reader that started moments ago still holds the version this run just
        superseded. Versions written seconds ago must survive the default call."""
        with tempfile.TemporaryDirectory() as d:
            table = _table_with_history(d)
            before = len(table.list_versions())

            _cleanup_versions(table)  # default window (indexer.VERSION_RETENTION)

            assert VERSION_RETENTION > timedelta(0)
            assert len(table.list_versions()) >= before
            assert table.count_rows() == 15

    def test_delete_files_batches_into_single_version():
        with tempfile.TemporaryDirectory() as d:
            db = lancedb.connect(d)
            files = [f"f{i}.md" for i in range(10)]
            data = [r for f in files for r in _rows(f)]
            table = db.create_table("t", data=data, schema=SCHEMA)
            before = len(table.list_versions())

            _delete_files(table, files[:9])

            # 9 files removed in ONE delete call → exactly one new version
            assert len(table.list_versions()) == before + 1
            assert table.count_rows() == 3

    def test_delete_files_escapes_quotes():
        with tempfile.TemporaryDirectory() as d:
            db = lancedb.connect(d)
            table = db.create_table(
                "t", data=_rows("it's.md") + _rows("keep.md"), schema=SCHEMA
            )
            _delete_files(table, ["it's.md"])
            assert table.count_rows() == 3
else:
    def test_skipped_lancedb_not_installed():
        """Placeholder so the file reports as skipped-by-design, not empty."""
        pass


if __name__ == "__main__":
    if lancedb is None:
        print("SKIP: lancedb not installed")
    else:
        test_cleanup_versions_prunes_old_snapshots()
        test_cleanup_versions_keeps_versions_inside_retention_window()
        test_delete_files_batches_into_single_version()
        test_delete_files_escapes_quotes()
        print("OK: 4 tests passed")
