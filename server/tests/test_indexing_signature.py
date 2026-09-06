"""Tests for the indexing signature and the query-template identifier.

The signature decides whether an existing index is still valid. Two properties
have to hold, and they pull in opposite directions:

  1. Editing the QUERY prefix must NOT invalidate the corpus. Documents are
     embedded without the prefix, so a "rebuild" would re-emit byte-identical
     vectors. Before 2.9.0 the prefix was part of the payload, which turned a
     harmless edit into `sys.exit(2)` on the next incremental run — the index
     silently stopped updating.
  2. Editing something the stored vectors DO depend on (chunk sizes, the model)
     must still stop the run, because mixing two chunkings in one table is
     corruption nothing ever reports.

Plus the migration path: an index built before 2.9.0 carries the old signature
and must be recognized as compatible rather than demanding a pointless rebuild.

Run directly (`python server/tests/test_indexing_signature.py`) or under pytest.
"""

import hashlib
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("VAULT_PATH", tempfile.mkdtemp(prefix="vsearch_test_vault_"))

try:
    import lancedb  # noqa: F401  (textbook_indexer imports it at module level)
    import textbook_indexer
except ImportError:
    textbook_indexer = None

from config import template_version  # noqa: E402


def _reload_textbook(**env):
    """Re-import textbook_indexer with the given env vars applied."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        return importlib.reload(textbook_indexer)
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


# --- template_version: the identifier written to the search log --------------

def test_template_version_is_stable_for_the_same_inputs():
    a = template_version("qwen3-v1", "Instruct: find things.\nQuery: ")
    b = template_version("qwen3-v1", "Instruct: find things.\nQuery: ")
    assert a == b
    assert a.startswith("qwen3-v1-")


def test_template_version_changes_when_only_the_prefix_changes():
    """The whole point: the name alone can lie about which text was used."""
    same_name_a = template_version("qwen3-v1", "Instruct: find A.\nQuery: ")
    same_name_b = template_version("qwen3-v1", "Instruct: find B.\nQuery: ")
    assert same_name_a != same_name_b


# --- The signature -----------------------------------------------------------

if textbook_indexer is not None:

    def test_query_prefix_is_not_part_of_the_signature():
        """A prefix edit must not invalidate a corpus it cannot have affected.

        NOTE: importlib.reload mutates the module in place and hands back the
        same object, so every value has to be read out before the next reload.
        """
        m = _reload_textbook(
            VAULT_SEARCH_TEXTBOOK_QUERY_PREFIX="Instruct: generic.\nQuery: ")
        sig_generic, tmpl_generic = m.compute_indexing_signature(), m.QUERY_TEMPLATE_VERSION

        m = _reload_textbook(
            VAULT_SEARCH_TEXTBOOK_QUERY_PREFIX="Instruct: something specific.\nQuery: ")
        sig_domain, tmpl_domain = m.compute_indexing_signature(), m.QUERY_TEMPLATE_VERSION

        assert sig_generic == sig_domain
        # ...while the logged identifier still distinguishes the two
        assert tmpl_generic != tmpl_domain

    def test_template_name_is_not_part_of_the_signature():
        m = _reload_textbook(VAULT_SEARCH_TEXTBOOK_TEMPLATE_VERSION="name-a")
        sig_a = m.compute_indexing_signature()
        m = _reload_textbook(VAULT_SEARCH_TEXTBOOK_TEMPLATE_VERSION="name-b")
        sig_b = m.compute_indexing_signature()
        assert sig_a == sig_b

    def test_chunking_change_does_invalidate_the_signature():
        """The half that must keep working: real changes still stop the run."""
        m = _reload_textbook()
        before = m.compute_indexing_signature()
        original = m.CHILD_TARGET_TOKENS
        try:
            m.CHILD_TARGET_TOKENS = original + 64
            assert m.compute_indexing_signature() != before
        finally:
            m.CHILD_TARGET_TOKENS = original
        assert m.compute_indexing_signature() == before

    def test_model_change_does_invalidate_the_signature():
        m = _reload_textbook()
        before = m.compute_indexing_signature()
        original = m.TEXTBOOK_EMBEDDING_MODEL
        try:
            m.TEXTBOOK_EMBEDDING_MODEL = original + "-other"
            assert m.compute_indexing_signature() != before
        finally:
            m.TEXTBOOK_EMBEDDING_MODEL = original

    def test_legacy_signature_reproduces_the_pre_2_9_0_payload():
        """An index built by an older version must be recognized, not rebuilt.

        Recomputes the old formula here rather than trusting the helper: the
        payload order is the thing under test, and getting it wrong would make
        every upgraded install do a pointless multi-hour rebuild.
        """
        m = _reload_textbook()
        payload = "|".join([
            m.ALGORITHM_VERSION,
            m.CHUNKING_VERSION,
            m.TEXTBOOK_EMBEDDING_MODEL,
            m.QUERY_TEMPLATE_NAME,          # the bare name, as it used to be
            f"CHILD_TARGET={m.CHILD_TARGET_TOKENS}",
            f"CHILD_OVERLAP={m.CHILD_OVERLAP_TOKENS}",
            f"PARENT_MAX={m.PARENT_MAX_TOKENS}",
            f"PARENT_MIN={m.PARENT_MIN_TOKENS}",
            f"TABLE_MAX={m.TABLE_HARD_MAX_TOKENS}",
            f"MIN_EMBED={m.MIN_EMBED_TOKENS}",
        ])
        expected = hashlib.sha256(payload.encode()).hexdigest()[:16]

        assert m.legacy_indexing_signature() == expected
        assert expected in m.compatible_signatures()
        assert m.compute_indexing_signature() in m.compatible_signatures()

    def test_legacy_and_current_signatures_differ():
        """If these collided the migration branch would never be exercised."""
        m = _reload_textbook()
        assert m.legacy_indexing_signature() != m.compute_indexing_signature()

else:
    def test_skipped_lancedb_not_installed():
        """Placeholder so the file reports as skipped-by-design, not empty."""
        pass


if __name__ == "__main__":
    test_template_version_is_stable_for_the_same_inputs()
    test_template_version_changes_when_only_the_prefix_changes()
    if textbook_indexer is None:
        print("OK: 2 tests passed (lancedb missing — signature tests skipped)")
    else:
        test_query_prefix_is_not_part_of_the_signature()
        test_template_name_is_not_part_of_the_signature()
        test_chunking_change_does_invalidate_the_signature()
        test_model_change_does_invalidate_the_signature()
        test_legacy_signature_reproduces_the_pre_2_9_0_payload()
        test_legacy_and_current_signatures_differ()
        print("OK: 8 tests passed")
