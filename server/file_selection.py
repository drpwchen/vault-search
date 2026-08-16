"""Which .md files of the textbook corpus actually get indexed.

Split out of textbook_indexer.py so the rules can be exercised without importing
lancedb/ollama — see tests/test_file_selection.py.

The selected set is authoritative: index_textbooks() derives `current_files_rel`
from it and hands that to orphan_cleanup(), which DELETES rows for anything not in
it. Every rule here therefore defaults toward keeping a file. Dropping one too many
costs index coverage silently; keeping one too many only costs duplicate rows,
which is visible and reversible.
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def _names(env_var: str, default: str) -> set[str]:
    """Parse a comma-separated basename list, casefolded for matching."""
    return {
        f.strip().casefold()
        for f in os.environ.get(env_var, default).split(",")
        if f.strip()
    }


# Skip patterns. SKIP_FILES is env-driven because a converter that emits both a
# whole-book file and per-chapter files (textbook-to-note does, for any PDF with
# bookmarks) would otherwise have every page embedded twice — doubling index time
# and returning the same passage twice per query.
#
# Matching is casefolded throughout: this runs on Windows and macOS, where
# `Index.md` and `INDEX.md` are the same file, and a case-sensitive comparison
# there let a differently-cased index page pass as book content.
SKIP_FILES = _names("VAULT_SEARCH_TEXTBOOK_SKIP_FILES", "INDEX.md")
SKIP_DIR_FRAGMENTS = {"__pycache__"}

# Basenames that are never a book's own content, so removing one can never cost
# coverage. Everything else in SKIP_FILES is guarded (see select_md_files).
NEVER_BOOK_CONTENT = {"index.md"}

# Whole-book files that duplicate per-chapter siblings. Skipped ONLY when the same
# directory also holds files that actually look like chapters — a book whose
# whole-book file is its only copy (no PDF bookmarks to split on) keeps it and
# stays searchable.
#
# A plain entry in SKIP_FILES cannot express this: that list is global, but
# "is the whole-book file redundant?" is a per-book question. Setting
# VAULT_SEARCH_TEXTBOOK_SKIP_FILES=full_text.md on a mixed corpus drops the
# unsplit books entirely, and the next --incremental run's orphan_cleanup() then
# deletes their already-indexed rows. No error, the book just stops being
# retrievable. Reported by @drivysu on PR #2.
#
# Set to an empty value to index whole-book files even where chapters exist.
WHOLE_BOOK_FILES = _names("VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES", "full_text.md")

# What counts as a chapter file, and therefore as evidence that the whole-book file
# beside it is redundant. "Any other markdown in the folder" is NOT good enough: a
# stray README.md or conversion-notes file would then displace the only real copy of
# the book, and orphan_cleanup() would delete its rows.
#
# The default covers the conventions seen in practice — `ch01_title.md`,
# `01_title.md`, `pages_0001-0030.md` — i.e. an optional structure word, then a
# number, then a separator. A corpus that names chapters some other way (plain
# `Introduction.md`) matches nothing, so both files are kept: duplicates, not data
# loss. Point this at your own convention to deduplicate such a corpus.
CHAPTER_PATTERN = os.environ.get(
    "VAULT_SEARCH_TEXTBOOK_CHAPTER_PATTERN",
    r"^(?:(?:ch(?:apter)?|pp?|pages?|part|sect?(?:ion)?|vol(?:ume)?)[ _.\-]?)?\d+[ _.\-]",
)
CHAPTER_RE = re.compile(CHAPTER_PATTERN, re.IGNORECASE)


def select_md_files(
    all_md: list[Path],
    skip_files: set[str] | None = None,
    whole_book_files: set[str] | None = None,
    chapter_re: "re.Pattern[str] | None" = None,
) -> tuple[list[Path], dict]:
    """Pick the files to index, resolving whole-book/per-chapter duplication.

    Two rules, applied per directory so one book's shape never decides another's:

    1. `skip_files` — dropped by basename. Entries outside `NEVER_BOOK_CONTENT`
       are guarded: if dropping them would leave a directory with nothing to
       index, they are kept and the caller warns. Without that guard a global
       skip list silently erases every book that has no chapter split, and the
       next orphan_cleanup() deletes its already-indexed rows.
    2. `whole_book_files` — dropped only where a sibling that matches
       `chapter_re` survives rule 1. That is the per-book question a global list
       cannot express, and requiring a chapter-shaped name (rather than "any
       other markdown") keeps a stray README from displacing the book itself.

    Basename matching is casefolded; `skip_files` and `whole_book_files` passed in
    by a caller are casefolded here, so callers may pass them in any case.

    Returns (files_to_index, notes) where notes carries what the caller should
    report: how many whole-book files were deduplicated, and which directories
    the guard rescued.
    """
    skip = SKIP_FILES if skip_files is None else {s.casefold() for s in skip_files}
    whole_book = (
        WHOLE_BOOK_FILES
        if whole_book_files is None
        else {s.casefold() for s in whole_book_files}
    )
    chapter = CHAPTER_RE if chapter_re is None else chapter_re

    hard_skip = skip & NEVER_BOOK_CONTENT      # never a book's own content
    guarded_skip = skip - NEVER_BOOK_CONTENT   # may not empty a directory

    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in sorted(all_md):
        if any(part in SKIP_DIR_FRAGMENTS for part in f.parts):
            continue
        if not f.is_file():
            continue
        if f.name.casefold() in hard_skip:
            continue
        by_dir[f.parent].append(f)

    kept: list[Path] = []
    notes: dict = {"whole_book_skipped": 0, "guard_kept": []}

    for parent, files in by_dir.items():
        survivors = [f for f in files if f.name.casefold() not in guarded_skip]
        if not survivors:
            # Dropping the configured skip list would erase this book entirely.
            survivors = files
            notes["guard_kept"].append(parent)

        whole = [f for f in survivors if f.name.casefold() in whole_book]
        if whole:
            chapters = [
                f for f in survivors
                if f.name.casefold() not in whole_book and chapter.search(f.name)
            ]
            if chapters:
                # Chapter files cover the same pages — drop the whole-book copy.
                notes["whole_book_skipped"] += len(whole)
                survivors = [f for f in survivors if f.name.casefold() not in whole_book]
            # else: no chapter-shaped sibling, so this whole-book file may be the
            # book's only copy. Keep it.

        kept.extend(survivors)

    return sorted(kept), notes
