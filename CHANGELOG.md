# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 2.6.0 were not tagged. `plugin/manifest.json` carries the Obsidian
plugin's own version (2.5.1) and moves only when something inside `plugin/` changes,
so it deliberately does not track these tags.

## [2.7.0] — 2026-08-19 — Low-memory MCP client, OpenBLAS cap, tests in CI

Nothing here changes what a search returns. It is all about what the tooling costs
to keep running, plus the first CI that actually executes the test suite.

### Added

- **`server/mcp_thin.py`, a thin MCP client, and the two API endpoints it talks to
  (`GET /api/mcp/tools`, `POST /api/mcp/call`).** Registering `mcp_server.py`
  directly gives every MCP client session its own copy of the server, and each copy
  opens lancedb plus, on first use, the textbook tokenizer and parent maps. Measured
  on the author's vault, one such session holds 105 MB at startup, 367 MB after a
  single `vault_search`, and 2.2 GB once a `textbook_search` has run — so several
  concurrent agent sessions paid gigabytes for one index. `mcp_thin.py` uses only the
  standard library, holds no index, and forwards each call to a running
  `api_server.py`; the same three steps leave it at 14 MB.

  (The textbook corpus is an optional add-on. Without it the per-session cost tops
  out around 367 MB, which is still 26× the thin client.)

  Behaviour is identical by construction: `/api/mcp/call` dispatches to the same
  `handle_tool_call()` the standalone server uses, and `/api/mcp/tools` returns the
  same `TOOLS` list. The trade-off is a dependency on the API server being up, so
  the client degrades rather than failing: `tools/list` falls back to the last
  cached schema (`<data dir>/mcp_tools_cache.json`) so the session still starts, and
  a tool call returns a readable error naming the URL it could not reach.

  This is opt-in. `mcp_server.py` still works exactly as before, and remains the
  right choice for a single session with no API server running.

- **A `Tests` workflow.** `.github/workflows/tests.yml` byte-compiles every module
  under `server/` and runs `server/tests/test_file_selection.py` on Python 3.10,
  3.11 and 3.12. The repo has had those 17 tests since 2.6.0 but nothing ran them.
  They are plain stdlib and never touch lancedb or ollama, so CI needs no index and
  installs no dependencies.

### Changed

- **`api_server.py` and `mcp_server.py` cap OpenBLAS at one thread on import.**
  numpy bundles its own OpenBLAS, which pre-commits one scratch buffer per CPU core
  the moment numpy is imported: 373 MB of private bytes against 19 MB capped,
  measured on a 12-thread machine with numpy 2.5.1. Neither process does linear
  algebra of its own — Ollama computes embeddings, LanceDB does the vector math — so
  the buffers were pure overhead. `os.environ.setdefault` is used, so exporting your
  own `OPENBLAS_NUM_THREADS` still wins.

- **`_textbook_search_v2()` split into a thin Markdown wrapper over a new
  `_textbook_search_v2_raw()` that returns the structured result list.** Since the
  MCP output moved to compact Markdown, the only way to consume results
  programmatically was to parse that Markdown back — which is why the author's own
  smoke benchmark had been broken since the change. The missing-index case now
  raises `TextbookIndexMissing` and the wrapper converts it to the same error JSON
  as before, so the tool's output is byte-for-byte unchanged.

## [2.6.0] — 2026-08-16 — Clean-install fixes, textbook-only installs, whole-book deduplication

Three community reports, all reproducible, all fixed. Thanks to
[@drivysu](https://github.com/drivysu) (PRs #1, #2, #3) and
[@tony860616-sudo](https://github.com/tony860616-sudo) (issue #4).

### Fixed

- **`pandas` was missing from `requirements.txt`, so search was broken on every
  clean install** (#1, #4). Every LanceDB read goes through `.to_pandas()` — 8 call
  sites in `api_server.py`, 7 in `mcp_server.py`, 3 in `textbook_indexer.py` — but
  `pyarrow` does not pull pandas in, it only shims it if already present. Indexing
  appeared to succeed and then every query died with
  `ModuleNotFoundError: No module named 'pandas'`. The failure surfaced late and
  read like an index problem: the imports sit inside call paths, so
  `claude mcp list` still reported the server **✔ Connected** and the error only
  appeared on the first tool call — in one report, after a multi-hour index had
  already been built. Reported independently on macOS 15 and Windows 11.
- **Textbook-only installs could not call any tool** (#3). `handle_tool_call()`
  opened the `vault` notes table before dispatching, so an install that indexes
  only a textbook corpus — `VAULT_SEARCH_TEXTBOOK_PATH` set, the vault itself never
  indexed — failed every `textbook_search` with
  `ValueError: Table 'vault' was not found`, even though that tool never reads the
  notes table. The notes table is now opened only by the three tools that use it
  (`vault_search`, `vault_related`, `vault_similar`), and `vault_stats` degrades to
  reporting the textbook half instead of failing the whole call.

### Added

- **`VAULT_SEARCH_TEXTBOOK_SKIP_FILES`** (#2) — comma-separated basenames never
  indexed from the textbook corpus, default `INDEX.md`. Previously hardcoded.
- **`VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES`** — whole-book files that duplicate
  per-chapter siblings, default `full_text.md`. See *Changed* below. Set it to an
  empty value to restore the old behaviour.
- **`VAULT_SEARCH_TEXTBOOK_CHAPTER_PATTERN`** — what a chapter filename looks like,
  and therefore the evidence that the whole-book file beside it is redundant. The
  default matches `ch01_x.md`, `01_x.md` and `pages_0001-0030.md`. A corpus naming
  chapters some other way matches nothing and keeps both files — duplicates, never
  data loss.
- **`server/tests/test_file_selection.py`** — the repo's first tests. 17 cases,
  stdlib only, no lancedb/ollama import, runnable as
  `python server/tests/test_file_selection.py` or under pytest.

### Changed

- **Whole-book files are no longer indexed alongside their chapter files.**
  Converters such as [textbook-to-note](https://github.com/drpwchen/textbook-to-note)
  emit both `full_text.md` and `chNN_*.md` for any PDF with bookmarks. Indexing both
  embedded every page twice: index time doubled, and each query returned the same
  passage twice — once attributed to the chapter, once to the whole book — burning
  the `limit` budget. On the 399-book corpus this was developed against, 340 books
  were affected.

  The rule is **per book, not global**: `full_text.md` is dropped only where a
  sibling file whose name looks like a chapter exists. A book whose whole-book file
  is its only copy (no PDF bookmarks to split on) keeps it and stays searchable —
  59 books on that same corpus. This is deliberately not what
  `VAULT_SEARCH_TEXTBOOK_SKIP_FILES` does; see the footgun below.

  "Looks like a chapter" is a name test, not "any other markdown in the folder".
  The looser rule would let a stray `README.md` stand in for a chapter split, drop
  the only real copy of the book, and have `orphan_cleanup()` delete its rows.

- **Basename matching is now case-insensitive.** This runs on Windows and macOS,
  where `Index.md` and `INDEX.md` are the same file. Comparing case-sensitively
  there let a differently-cased index page pass as book content — with the looser
  chapter rule above, that was enough to displace the book itself.

- **`vault_related` no longer opens the notes table.** It answers from the
  wiki-link graph alone and never reads that table, so a textbook-only install can
  use it too. This completes the boundary PR #3 drew.

  **Upgrade note:** the first indexer run after upgrading drops the duplicate rows
  via `orphan_cleanup()` — any run that is not `--book <name>`, incremental or full
  rebuild alike. This deletes rows, it does not re-embed; the surviving chapter rows
  are untouched. Set `VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES=` to keep the old
  behaviour.

- **`VAULT_SEARCH_TEXTBOOK_SKIP_FILES` can no longer silently erase a book.** The
  skip list is global, but "is the whole-book file redundant?" is a per-book
  question — the footgun @drivysu flagged on PR #2 after deploying it. On a mixed
  corpus, `VAULT_SEARCH_TEXTBOOK_SKIP_FILES=INDEX.md,full_text.md` dropped every
  unsplit book, and the next `--incremental` run's `orphan_cleanup()` then deleted
  its already-indexed rows: no error, the book just stopped being retrievable. If
  the configured skip list would leave a directory with nothing to index, its files
  are now kept and a `[WARN]` names the directory. `INDEX.md` is exempt from the
  guard — it is never a book's own content.

- File-selection rules moved to `server/file_selection.py` so they can be tested
  without importing lancedb or ollama. No behaviour change from the move itself.
