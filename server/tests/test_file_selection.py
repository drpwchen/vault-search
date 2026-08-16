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
