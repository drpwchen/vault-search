# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 2.6.0 were not tagged. `plugin/manifest.json` carries the Obsidian
plugin's own version (2.5.1) and is unchanged by this release — nothing in
`plugin/` was touched.

## [2.6.0] — 2026-08-16 — Clean-install fixes, textbook-only installs, whole-book deduplication

Three community reports, all reproducible, all fixed. Thanks to
[@drivysu](https://github.com/drivysu) (PRs #1, #2, #3) and
[@tony860616-sudo](https://github.com/tony860616-sudo) (issue #4).

### Fixed

- **`pandas` was missing from `requirements.txt`, so search was broken on every
  clean install** (#1, #4). Every LanceDB read goes through `.to_pandas()` — 8 call
  sites in `api_server.py`, 8 in `mcp_server.py`, 3 in `textbook_indexer.py` — but
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
- **`server/tests/test_file_selection.py`** — the repo's first tests. Stdlib only,
  no lancedb/ollama import, runnable as `python server/tests/test_file_selection.py`
  or under pytest.

### Changed

- **Whole-book files are no longer indexed alongside their chapter files.**
  Converters such as [textbook-to-note](https://github.com/drpwchen/textbook-to-note)
  emit both `full_text.md` and `chNN_*.md` for any PDF with bookmarks. Indexing both
  embedded every page twice: index time doubled, and each query returned the same
  passage twice — once attributed to the chapter, once to the whole book — burning
  the `limit` budget. On the 399-book corpus this was developed against, 340 books
  were affected.

  The rule is **per book, not global**: `full_text.md` is dropped only where a
  sibling chapter file exists. A book whose whole-book file is its only copy (no
  PDF bookmarks to split on) keeps it and stays searchable — 59 books on that same
  corpus. This is deliberately not what `VAULT_SEARCH_TEXTBOOK_SKIP_FILES` does;
  see the footgun below.

  **Upgrade note:** the first `--incremental` run after upgrading will drop the
  duplicate rows via `orphan_cleanup()`. This deletes rows, it does not re-embed —
  the surviving chapter rows are untouched. Set
  `VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES=` to keep the old behaviour.

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
