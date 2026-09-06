"""Tests for the v1/v2 dual-track vault retrieval in mcp_server.

Two behaviours matter here and neither is visible in normal use:

  1. The flag FILE decides which index serves searches, so switching does not
     need a restart. If the flag stops being read, a cutover silently does
     nothing.
  2. When v2 is enabled but its index is missing or broken, the search must
     fall back to v1 and RECORD that it did. A silent downgrade to the older
     index looks exactly like "search got worse for no reason".

Run directly (`python server/tests/test_vault_v2_dual_track.py`) or under pytest.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_tmp = tempfile.mkdtemp(prefix="vsearch_test_")
os.environ.setdefault("VAULT_PATH", _tmp)
# Keep the flag file and logs out of the real data dir. This only works when
# nothing has imported config yet, so it is a hint, not the guarantee — the
# guarantee is _isolate_real_state() below.
os.environ.setdefault("VAULT_SEARCH_DATA_DIR", _tmp)

try:
    import lancedb  # noqa: F401  (mcp_server imports it at module level)
    import mcp_server
except ImportError:
    mcp_server = None


def _isolate_real_state() -> None:
    """Repoint the flag file and the vault log at this test's temp dir.

    These tests DELETE the flag file, and the flag file is what switches a real
    installation between the v1 and v2 index. config.py resolves it at import
    time, so if any earlier module imported config without a temp data dir, the
    env var set above arrives too late and `mcp_server.VAULT_V2_FLAG` still
    points at the user's `~/.vault-search`. That happened: a suite run deleted a
    live flag and silently moved daily searches back to the retired v1 index.

    Overriding the module attributes outright removes the import-order
    dependency — there is no ordering in which these tests can reach real state.
    """
    import logging
    import logging.handlers

    mcp_server.VAULT_V2_FLAG = Path(_tmp) / "vault_v2.enabled"

    log_path = Path(_tmp) / "vault_search_log.jsonl"
    mcp_server._vault_log_path = log_path
    for h in list(mcp_server._vault_logger.handlers):
        mcp_server._vault_logger.removeHandler(h)
        h.close()
    handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=1024 * 1024, backupCount=0, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    mcp_server._vault_logger.addHandler(handler)


if mcp_server is not None:
    _isolate_real_state()


def _rows(*notes):
    return [
        {"file": f"{n}.md", "note": n, "section": "", "folder": "", "tags": "",
         "similarity": 0.9 - i * 0.1, "mtime": None}
        for i, n in enumerate(notes)
    ]


if mcp_server is not None:

    def _run_search(monkeypatched: dict):
        """Call _vault_search with the retrieval + graph layers stubbed out."""
        saved = {k: getattr(mcp_server, k) for k in monkeypatched}
        saved["get_graph"] = mcp_server.get_graph
        mcp_server.get_graph = lambda: {}
        for k, v in monkeypatched.items():
            setattr(mcp_server, k, v)
        try:
            return mcp_server._vault_search({"query": "anything", "n_results": 5})
        finally:
            for k, v in saved.items():
                setattr(mcp_server, k, v)

    def _last_log_entry() -> dict:
        for h in mcp_server._vault_logger.handlers:
            h.flush()
        path = mcp_server._vault_log_path
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return json.loads(lines[-1])

    def test_flag_file_controls_which_index_is_active():
        flag = mcp_server.VAULT_V2_FLAG
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.unlink(missing_ok=True)
        assert mcp_server.vault_v2_active() is False

        flag.write_text("", encoding="utf-8")
        try:
            assert mcp_server.vault_v2_active() is True
        finally:
            flag.unlink(missing_ok=True)

    def test_v2_failure_falls_back_to_v1_and_says_so():
        """A broken v2 index must not fail the call — but must not hide either."""
        flag = mcp_server.VAULT_V2_FLAG
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("", encoding="utf-8")

        def boom(*a, **kw):
            raise RuntimeError("table 'vault_chunks_v2' is missing")

        try:
            out = _run_search({
                "_vault_retrieve_v2": boom,
                "_vault_retrieve_v1": lambda *a, **kw: _rows("Fallback note"),
            })
        finally:
            flag.unlink(missing_ok=True)

        assert "Fallback note" in out
        entry = _last_log_entry()
        assert entry["index_version"] == "v1"
        assert "vault_chunks_v2" in (entry["v2_fallback_reason"] or "")

    def test_v2_success_is_recorded_as_v2():
        flag = mcp_server.VAULT_V2_FLAG
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("", encoding="utf-8")

        def _never(*a, **kw):
            raise AssertionError("v1 must not be consulted when v2 works")

        try:
            out = _run_search({
                "_vault_retrieve_v2": lambda *a, **kw: _rows("V2 note"),
                "_vault_retrieve_v1": _never,
            })
        finally:
            flag.unlink(missing_ok=True)

        assert "V2 note" in out
        entry = _last_log_entry()
        assert entry["index_version"] == "v2"
        assert entry["v2_fallback_reason"] is None
        assert entry["embedding_model"] == mcp_server.VAULT_V2_MODEL

    def test_v1_path_is_used_when_the_flag_is_absent():
        flag = mcp_server.VAULT_V2_FLAG
        flag.unlink(missing_ok=True)

        def _never(*a, **kw):
            raise AssertionError("v2 must not be consulted without the flag")

        out = _run_search({
            "_vault_retrieve_v2": _never,
            "_vault_retrieve_v1": lambda *a, **kw: _rows("V1 note"),
        })

        assert "V1 note" in out
        assert _last_log_entry()["index_version"] == "v1"

else:
    def test_skipped_lancedb_not_installed():
        """Placeholder so the file reports as skipped-by-design, not empty."""
        pass


if __name__ == "__main__":
    if mcp_server is None:
        print("SKIP: lancedb not installed")
    else:
        test_flag_file_controls_which_index_is_active()
        test_v2_failure_falls_back_to_v1_and_says_so()
        test_v2_success_is_recorded_as_v2()
        test_v1_path_is_used_when_the_flag_is_absent()
        print("OK: 4 tests passed")
