# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 2.6.0 were not tagged. `plugin/manifest.json` carries the Obsidian
plugin's own version (2.5.1) and moves only when something inside `plugin/` changes,
so it deliberately does not track these tags.

## [2.9.0] — 2026-09-06 — Parent-child vault index, one GPU lease, no hardcoded folders

Vault search gets the retrieval architecture the textbook corpus has had for
months: every note is split into small child chunks for a precise vector hit,
plus subsection-sized parents that are what you actually get back. It ships as a
**second pair of tables**, so the existing index keeps serving until you decide
to switch — create `~/.vault-search/vault_v2.enabled` to read v2, delete it to go
back, no restart either way.

This release also removes the last folder names baked into the code. Ranking
weights, the archive folder, derivative-note exclusion, entity-extraction
priority and both query templates now come from the environment, so the defaults
suit any vault rather than the author's.

### Added

- **`server/vault_indexer_v2.py`** — parent-child indexing for the vault, with
  incremental hashing, a generation marker and error accounting. Its embedding
  model is its own knob (`VAULT_SEARCH_V2_MODEL`): a vault-only install never has
  to configure a textbook corpus to use v2.
- **Dual-track retrieval.** `vault_search`, `/api/search` and the chat context
  picker read v2 when the flag file is present and fall back to v1 when the v2
  index is missing or broken — and the fallback is RECORDED
  (`index_version`, `v2_fallback_reason`) in `vault_search_log.jsonl`, because a
  silent downgrade to the older index looks exactly like "search got worse for no
  reason".
- **`server/gpu_lease_client.py`** — one `GpuLease` class shared by both
  indexers: acquire before any GPU work, heartbeat while a long run makes
  progress, and yield the card between files when another job is queued.
  Unset `VAULT_SEARCH_GPU_LEASE` (the default) makes every call a no-op — no
  subprocess is ever spawned on the open-source path.

### Changed

- **Ranking is configuration, not code.** `VAULT_SEARCH_PATH_WEIGHTS`,
  `VAULT_SEARCH_ARCHIVE_FOLDER`, `VAULT_SEARCH_EXCLUDE_PATTERNS` and
  `VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS` replace folder names that were written
  into `scoring.py`, `mcp_server.py` and `graph_builder.py`. All default to
  empty: every folder weighted equally, nothing excluded, no folder favoured.
- **Query templates are configurable and self-identifying.** The logged
  `query_template_version` is now the template's name plus a short hash of the
  prefix itself, so a log line can never name one template while a different
  prefix is in use. The template is deliberately NOT part of the indexing
  signature — it only shapes queries, never the stored vectors, so editing it
  no longer forces a full re-index.
- **One excerpt budget.** `vault_similar` quoted 500 characters of a note where
  `vault_search` quoted 800, because one path built its rows through
  `format_results()` and the other inline. Both now use `EXCERPT_CHARS = 800`.

### Fixed

- **`_search_vault_context` embedded its query with the wrong model.** The chat
  context picker built a bge-m3 vector and searched the qwen3 textbook table with
  it. Both are 1024-dimensional, so nothing raised — the results were just
  quietly worse.
- **The entities block was ordered by `PYTHONHASHSEED`.** It was built by
  iterating a `set` of note names, and with its 15-entry cap that meant two
  processes on the same index answered one query with a different SUBSET of
  entities. It now follows result rank.
- **`/api/similar` reported `similarity: 0.0` for some graph-only neighbours.**
  Their cosine is fetched with one prefiltered vector query, and that query ranks
  chunks, not notes: with parent-child chunking the row budget ran out before the
  least similar notes were reached — exactly the ones a wiki-link candidate tends
  to be. They reported 0.0 and fell under every client threshold, so the Obsidian
  panel hid rows the endpoint had returned. Whatever the budget cuts is now asked
  for again. On a 50,700-chunk vault this took a 40-candidate lookup from 36
  answers to 40.

### Notes

- Tests never touch a real data directory any more. `config.py` resolves
  `DATA_DIR` at import time, so a module setting the env var in its own header
  loses to whichever test file imported config first — and a test that deletes
  the v2 flag file then deletes the one in `~/.vault-search`. The new
  `server/tests/conftest.py` sets it before any test module is imported, and the
  flag-file tests override the module attribute outright.
- 57 tests, no network and no lancedb required for most of them.

## [2.8.1] — 2026-09-05 — Related Notes showed nothing: /api/similar reported a fusion rank, not a similarity

If the Obsidian plugin's Related Notes panel (🔗) has been answering "No results
with similarity ≥ 0.50" for every note in your vault, this release is the fix.
Nothing was wrong with your index; the number the panel filtered on was never a
similarity in the first place. Reported by @tony860616-sudo in #5, with the root
cause already located — thank you.

### Fixed

- **`/api/similar` and the `vault_similar` MCP tool now report a real cosine
  similarity.** Both rank their candidates by Reciprocal Rank Fusion over the
  semantic and wiki-link-graph rankings, then wrote that fusion score into the
  `similarity` field. An RRF score is bounded by `2/RRF_K` — about 0.033, with a
  single-ranking top hit at 0.0167 — so it could never approach the 0.5 and 0.7
  thresholds the plugin's panel, and its chat context picker, are calibrated for.
  Every result was filtered out no matter how close the content actually was.
  The real cosine the LanceDB search already computed was discarded before the
  overwrite. `similarity` now means on these two paths exactly what it means on
  `/api/search`: cosine under the same path and recency weighting.
- **The fusion ranking is unchanged.** `rerank()` takes a new optional
  `rank_key` argument: it orders by that field under the same weights and
  consumes it, leaving `similarity` to report the similarity. Graph-led results
  still outrank closer-but-unlinked notes exactly as before — only the number
  shown next to them changed.
- **Graph-only candidates carry a real number too.** A note that reaches the
  result set through the wiki-link graph never went through the vector search,
  so it had no similarity at all. One prefiltered vector query now fetches
  theirs. Notes with no indexed content (an attachment, an empty stub) honestly
  report 0.00 and fall below the panel's threshold, where they belong.

### Notes

- `server/tests/test_similarity_reporting.py` pins all of it: the reported value
  is the cosine, the fused ordering survives, the weights reach both numbers,
  and the ranking field never leaks into a result. Plain stdlib, no lancedb.
- Restart the API server to pick this up — the plugin talks to the running
  process, not to the files on disk.

## [2.8.0] — 2026-08-21 — Unbounded disk growth fixed: LanceDB old versions are now purged

If your `lance_db/` directory keeps growing even though your vault does not, this
release is the fix — and the growth it stops is not small. LanceDB snapshots the
whole table state on **every** write and never deletes those snapshots on its own.
Neither indexer ever asked it to, so every incremental run left one more full
version on disk. On the author's machine the textbook index reached 51 versions
and 13 GB on disk for what compacts to under 4 GB, and eventually filled the drive
to the point that indexing itself crashed with `LanceError(IO): os error 112`
(compaction needs scratch space, so the failure shows up exactly when you can least
afford it). A friend running this project reported the same mystery shrinkage.

Two compounding causes, both fixed:

### Fixed

- **Both indexers now purge old table versions at the end of every run**
  (`table.optimize(cleanup_older_than=0)`, with a `compact_files()` +
  `cleanup_old_versions()` fallback for older lancedb). Cleanup never touches the
  current version, so live queries are unaffected. Note that `compact_files()`
  alone — what `textbook_indexer.py` did before — actually made things *worse*:
  it wrote the merged fragments as yet another version while all previous
  versions stayed on disk.
- **`indexer.py` incremental mode batches its deletes.** It used to issue one
  `table.delete()` per modified file, and each of those calls is a full version
  snapshot; re-indexing after touching 200 notes meant 200 snapshots before this
  release. Deletes now go through one `file IN (...)` predicate per 400 files
  (matching what `textbook_indexer.py` already did), so the same run writes a
  handful of versions instead.

### Added

- `server/tests/test_version_cleanup.py`: three tests covering version purge,
  delete batching, and quote escaping in the batched predicate. They need
  `lancedb` installed and skip themselves cleanly on the hermetic CI runners.
  All three were mutation-verified (cleanup made a no-op, batching reverted to
  per-file, escaping removed — each mutation is caught).

### For existing installs

The fix prevents future growth but does not shrink what is already on disk —
old versions are purged by the *next* index run. Run either indexer once (even a
no-op incremental run cleans up), or reclaim immediately without re-indexing:

```python
import lancedb
from datetime import timedelta
db = lancedb.connect("path/to/lance_db")
for name in db.table_names():
    db.open_table(name).optimize(cleanup_older_than=timedelta(0))
```

## [2.7.0] — 2026-08-19 — Low-memory server and MCP client, OpenBLAS cap, tests in CI

Nothing here changes what a search returns. It is all about what the tooling costs
to keep running, plus the first CI that actually executes the test suite.

One thread runs through it, and it is worth stating up front because it nearly sent
this release out with the wrong conclusion. The resident `api_server` appeared to
hold the whole index in 116 MB while a standalone `mcp_server.py` needed 2.2 GB for
the same `textbook_search`. It did not. `uvicorn(reload=True)` forks a supervisor
that owns the port and a child that owns the app; 116 MB was the supervisor, and the
child beside it held 2,211 MB. Both paths ran the same code and paid the same price.
The real cost was two resident structures neither path needed to keep.

### Added

- **`server/mcp_thin.py`, a thin MCP client, and the two API endpoints it talks to
  (`GET /api/mcp/tools`, `POST /api/mcp/call`).** Registering `mcp_server.py`
  directly gives every MCP client session its own copy of the server, and each copy
  opens lancedb plus, on first use, the textbook tokenizer and the graph cache.
  Measured on the author's vault after the memory fixes below, one such session holds
  103 MB at import, 293 MB after a single `vault_search`, and 338 MB once a
  `textbook_search` has run — before those fixes that last step reached 2.2 GB.
  `mcp_thin.py` uses only the standard library, holds no index, and forwards each
  call to a running `api_server.py`; the same three steps leave it at 14 MB.

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

  Its first run earned its keep: `test_whole_book_match_is_case_insensitive` used a
  `FULL_TEXT.MD` fixture, which `rglob("*.md")` never sees on Linux, so the test had
  been passing on Windows for a reason that does not hold elsewhere. The fixture is
  now `FULL_TEXT.md` — uppercase basename, lowercase extension — which is the case
  rule `file_selection.py` actually owns. Worth knowing separately: on Linux a
  genuinely `.MD`-suffixed file is not indexed at all, because the indexer's glob is
  case-sensitive there. That behaviour is unchanged here.

### Changed

- **Textbook parents are read per search instead of being held in a dict.**
  `_load_parent_map()` pulled every parent row into memory at first use: 2,120 MB
  of private bytes for a 162,864-parent corpus. A search only ever needs the ~20
  parents it is about to return, so `fetch_textbook_parents()` asks LanceDB for
  exactly those with one `parent_id IN (...)` query — 13–17 ms warm, against the
  3–4 s an end-to-end search already spends embedding the query. The vault-side
  parent map went the same way in the author's copy.

  This also retires the mtime + `READY.*` marker reload dance and the
  `POST /api/admin/reload-parents` endpoint it existed for: every search now reads
  the current index, so a reindex has no stale window. The route still answers, so
  old callers do not 404, but it reports that it did nothing.

- **The Qwen3 tokenizer loads through `tokenizers`, not `transformers`.**
  `AutoTokenizer` imports torch, which cost 371 MB of private bytes for what is
  pure Rust tokenization; the `tokenizers` backend costs ~58 MB and never touches
  torch. `get_tokenizer()` returns a thin adapter with the same `encode`/`decode`
  shape, and prefers the local Hugging Face cache so an offline machine still
  starts. Verified identical before the swap on 300 real parent texts plus empty,
  whitespace, CJK, emoji and 5000-character inputs: same ids, same decode output,
  zero mismatches. `requirements.txt` swaps `transformers` for `tokenizers` +
  `huggingface-hub`.

- **`api_server.py` no longer starts uvicorn's reloader by default.** Pass
  `--reload` while editing the server; `--no-reload` is still accepted. Besides the
  supervisor's own ~116 MB, the reloader's child process has no readable command
  line and answers no HTTP, which is exactly the shape a stray-process sweep flags
  as suspect.

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

### Measured

One MCP session, on the author's vault (private bytes, 162,864-parent textbook
corpus):

| step | 2.6.0 | 2.7.0 |
|---|---|---|
| import | 105 MB | 103 MB |
| after one `vault_search` | 367 MB | 293 MB |
| after one `textbook_search` | 2,225 MB | 338 MB |
| the same three steps via `mcp_thin.py` | — | 14 MB |

The 20-query textbook smoke suite passes 20/20 before and after, and the 17 stdlib
tests pass on 3.10–3.12.

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
