"""Which .md files of the textbook corpus actually get indexed.

Split out of textbook_indexer.py so the rules can be exercised without importing
lancedb/ollama — see tests/test_file_selection.py.
"""

import os
from collections import defaultdict
from pathlib import Path

# Skip patterns. SKIP_FILES is env-driven because a converter that emits both a
# whole-book file and per-chapter files (textbook-to-note does, for any PDF with
# bookmarks) would otherwise have every page embedded twice — doubling index time
# and returning the same passage twice per query.
SKIP_FILES = {
    f.strip()
    for f in os.environ.get("VAULT_SEARCH_TEXTBOOK_SKIP_FILES", "INDEX.md").split(",")
    if f.strip()
}
SKIP_DIR_FRAGMENTS = {"__pycache__"}

# Basenames that are never a book's own content, so removing one can never cost
# coverage. Everything else in SKIP_FILES is guarded (see select_md_files).
NEVER_BOOK_CONTENT = {"INDEX.md"}

# Whole-book files that duplicate per-chapter siblings. Skipped ONLY when the same
# directory also holds chapter files — a book whose whole-book file is its only copy
# (no PDF bookmarks to split on) keeps it and stays searchable.
#
# A plain entry in SKIP_FILES cannot express this: that list is global, but
# "is the whole-book file redundant?" is a per-book question. Setting
# VAULT_SEARCH_TEXTBOOK_SKIP_FILES=full_text.md on a mixed corpus drops the
# unsplit books entirely, and the next --incremental run's orphan_cleanup() then
# deletes their already-indexed rows. No error, the book just stops being
# retrievable. Reported by @drivysu on PR #2.
#
# Set to an empty value to index whole-book files even where chapters exist.
WHOLE_BOOK_FILES = {
    f.strip()
    for f in os.environ.get(
        "VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES", "full_text.md"
    ).split(",")
    if f.strip()
}


def select_md_files(
    all_md: list[Path],
    skip_files: set[str] | None = None,
    whole_book_files: set[str] | None = None,
) -> tuple[list[Path], dict]:
    """Pick the files to index, resolving whole-book/per-chapter duplication.

    Two rules, applied per directory so one book's shape never decides another's:

    1. `skip_files` — dropped by basename. Entries outside `NEVER_BOOK_CONTENT`
       are guarded: if dropping them would leave a directory with nothing to
       index, they are kept and the caller warns. Without that guard a global
       skip list silently erases every book that has no chapter split, and the
       next orphan_cleanup() deletes its already-indexed rows.
    2. `whole_book_files` — dropped only where a sibling chapter file survives
       rule 1. That is the per-book question a global list cannot express.

    Returns (files_to_index, notes) where notes carries what the caller should
    report: how many whole-book files were deduplicated, and which directories
    the guard rescued.
    """
    skip = SKIP_FILES if skip_files is None else skip_files
    whole_book = WHOLE_BOOK_FILES if whole_book_files is None else whole_book_files

    hard_skip = skip & NEVER_BOOK_CONTENT      # never a book's own content
    guarded_skip = skip - NEVER_BOOK_CONTENT   # may not empty a directory

    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in sorted(all_md):
        if any(part in SKIP_DIR_FRAGMENTS for part in f.parts):
            continue
        if not f.is_file():
            continue
        if f.name in hard_skip:
            continue
        by_dir[f.parent].append(f)

    kept: list[Path] = []
    notes: dict = {"whole_book_skipped": 0, "guard_kept": []}

    for parent, files in by_dir.items():
        survivors = [f for f in files if f.name not in guarded_skip]
        if not survivors:
            # Dropping the configured skip list would erase this book entirely.
            survivors = files
            notes["guard_kept"].append(parent)

        chapters = [f for f in survivors if f.name not in whole_book]
        if chapters:
            notes["whole_book_skipped"] += len(survivors) - len(chapters)
            survivors = chapters
        # else: the whole-book file is this book's only copy — keep it.

        kept.extend(survivors)

    return sorted(kept), notes
