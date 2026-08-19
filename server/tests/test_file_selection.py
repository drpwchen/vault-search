"""Tests for textbook corpus file selection.

Plain stdlib — run directly (`python server/tests/test_file_selection.py`) or under
pytest. No lancedb/ollama import, so it works on a machine that has never indexed
anything.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_selection import select_md_files  # noqa: E402

SKIP = {"INDEX.md"}
WHOLE = {"full_text.md"}


def build(root: Path, layout: dict[str, list[str]]) -> list[Path]:
    """Create {book_dir: [filenames]} under root and return every .md path."""
    for book, names in layout.items():
        d = root / book
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_text("x", encoding="utf-8")
    return list(root.rglob("*.md"))


def names(paths: list[Path]) -> set[str]:
    return {f"{p.parent.name}/{p.name}" for p in paths}


def test_split_book_drops_whole_book_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "ch01_a.md", "ch02_b.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/ch01_a.md", "Split/ch02_b.md"}
        assert notes["whole_book_skipped"] == 1


def test_unsplit_book_keeps_whole_book_file():
    """The regression @drivysu reported: a book with no chapter split must survive."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Unsplit": ["full_text.md", "INDEX.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Unsplit/full_text.md"}
        assert notes["whole_book_skipped"] == 0


def test_mixed_corpus_decides_per_book():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {
            "Split": ["full_text.md", "ch01_a.md"],
            "Unsplit": ["full_text.md"],
        })
        kept, _ = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/ch01_a.md", "Unsplit/full_text.md"}


def test_index_md_always_dropped_and_never_guarded():
    """INDEX.md is never book content, so a folder holding only it indexes nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"OnlyIndex": ["INDEX.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert kept == []
        assert notes["guard_kept"] == []


def test_guard_keeps_book_a_skip_list_would_erase():
    """SKIP_FILES=full_text.md must not silently delete an unsplit book."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Unsplit": ["full_text.md"]})
        kept, notes = select_md_files(all_md, {"INDEX.md", "full_text.md"}, WHOLE)
        assert names(kept) == {"Unsplit/full_text.md"}
        assert len(notes["guard_kept"]) == 1


def test_skip_list_still_applies_where_something_survives():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "ch01_a.md"]})
        kept, notes = select_md_files(all_md, {"INDEX.md", "full_text.md"}, WHOLE)
        assert names(kept) == {"Split/ch01_a.md"}
        assert notes["guard_kept"] == []


def test_empty_whole_book_config_indexes_everything():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "ch01_a.md"]})
        kept, notes = select_md_files(all_md, SKIP, set())
        assert names(kept) == {"Split/full_text.md", "Split/ch01_a.md"}
        assert notes["whole_book_skipped"] == 0


def test_pycache_is_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {
            "Book": ["ch01_a.md"],
            "Book/__pycache__": ["junk.md"],
        })
        kept, _ = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Book/ch01_a.md"}


def test_readme_sibling_does_not_displace_the_book():
    """A stray non-chapter markdown file is not evidence of a chapter split.

    Treating it as one drops full_text.md from the selected set, and
    orphan_cleanup() then deletes the only real copy of the book.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Unsplit": ["full_text.md", "README.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Unsplit/full_text.md", "Unsplit/README.md"}
        assert notes["whole_book_skipped"] == 0


def test_case_variant_index_does_not_displace_the_book():
    """`Index.md` is the same file as `INDEX.md` on Windows and macOS."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Unsplit": ["full_text.md", "Index.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Unsplit/full_text.md"}
        assert notes["whole_book_skipped"] == 0


def test_whole_book_match_is_case_insensitive():
    """Basename matching ignores case: FULL_TEXT.md still counts as full_text.md.

    The extension stays lowercase on purpose. Which files reach select_md_files()
    is decided by the indexer's `rglob("*.md")`, which is case-sensitive on Linux
    and case-insensitive on Windows and macOS — so a `.MD` fixture would test the
    platform's glob, not this module.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["FULL_TEXT.md", "ch01_a.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/ch01_a.md"}
        assert notes["whole_book_skipped"] == 1


def test_non_chapter_siblings_survive_alongside_chapters():
    """Deduplication removes the whole-book file, not every other file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "ch01_a.md", "README.md"]})
        kept, _ = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/ch01_a.md", "Split/README.md"}


def test_numeric_prefix_chapters_count_as_chapters():
    """The other convention seen in practice: 05_Imaging_Techniques.md"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "05_Imaging.md", "06_Gait.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/05_Imaging.md", "Split/06_Gait.md"}
        assert notes["whole_book_skipped"] == 1


def test_page_range_splits_count_as_chapters():
    """Page-range splits cover the same pages as the whole-book file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {
            "Split": ["full_text.md", "pages_0001-0030.md", "pages_0031-0060.md"],
        })
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Split/pages_0001-0030.md", "Split/pages_0031-0060.md"}
        assert notes["whole_book_skipped"] == 1


def test_unrecognised_chapter_naming_keeps_both():
    """Unknown convention → duplicates, which are visible. Never data loss."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "Introduction.md", "Gait.md"]})
        kept, notes = select_md_files(all_md, SKIP, WHOLE)
        assert "Split/full_text.md" in names(kept)
        assert notes["whole_book_skipped"] == 0


def test_custom_chapter_pattern():
    import re
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {"Split": ["full_text.md", "part-a.md", "part-b.md"]})
        kept, notes = select_md_files(
            all_md, SKIP, WHOLE, chapter_re=re.compile(r"^part-")
        )
        assert names(kept) == {"Split/part-a.md", "Split/part-b.md"}
        assert notes["whole_book_skipped"] == 1


def test_nested_volumes_are_independent():
    """Multi-volume books live in sibling subdirs; one volume's shape must not
    decide the other's."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_md = build(root, {
            "Brukner/Vol1": ["full_text.md", "ch01_a.md"],
            "Brukner/Vol2": ["full_text.md"],
        })
        kept, _ = select_md_files(all_md, SKIP, WHOLE)
        assert names(kept) == {"Vol1/ch01_a.md", "Vol2/full_text.md"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
