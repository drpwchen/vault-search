"""Test-session guard: no test may touch the real data directory.

`config.py` resolves DATA_DIR (flag file, search logs, hash caches) at IMPORT
time, and the first test module that imports it fixes that value for the whole
session. A module that sets the env var in its own header is therefore too late
whenever an earlier module imported config first — which is exactly how the
v2 flag file in a real `~/.vault-search` got deleted by a test run.

conftest.py is imported before any test module, so setting it here is the one
place the ordering cannot go wrong.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="vsearch_tests_")
os.environ.setdefault("VAULT_PATH", _TMP)
# Force, not setdefault: an inherited value from the developer's shell would
# point the whole suite back at real state.
os.environ["VAULT_SEARCH_DATA_DIR"] = _TMP
