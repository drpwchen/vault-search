"""
Textbook Semantic Indexer v2 — Parent-Child Chunking with Qwen3-Embedding-0.6B

Reads all .md files under textbook-md/, builds a 2-tier index:
  - textbook_chunks: ~512-tok children, what we search by default
  - textbook_parents: subsection-sized (~1024 tok), returned for context
Both have L2-normalized vectors; cosine distance.

Usage:
    python textbook_indexer.py                          # full rebuild (drops & rebuilds)
    python textbook_indexer.py --incremental            # only changed files (mtime + hash)
    python textbook_indexer.py --book Braddom_7e        # one book, preserves others
    python textbook_indexer.py --retry-failed           # retry status=pending error log entries
    python textbook_indexer.py --retry-permanent        # retry status=permanent (manual triage)
    python textbook_indexer.py --reset-error-status ID  # reset one entry to pending
"""

import os
# CRITICAL: must be set BEFORE any tokenizers/transformers import to prevent
# Fast tokenizer + Python multi-thread deadlock during long indexing runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import lancedb
import ollama
import pyarrow as pa


# =============================================================================
# Constants — pinned in code; bump these to invalidate corpus via INDEX_SIGNATURE
# =============================================================================

# Path / connection (resolved from environment / .env via config.py)
from config import (
    TEXTBOOK_PATH, DB_PATH, OLLAMA_HOST,
    TEXTBOOK_EMBEDDING_MODEL,
    TEXTBOOK_HASH_CACHE as HASH_CACHE_PATH,
    TEXTBOOK_ERROR_LOG as ERROR_LOG_PATH,
    TEXTBOOK_ERROR_ARCHIVE as ERROR_LOG_ARCHIVE_PATH,
    GPU_LEASE_PATH,
)

# File-selection rules (which .md files of the corpus get indexed at all) live in
# their own module so they can be tested without importing lancedb/ollama.
from file_selection import (
    WHOLE_BOOK_FILES,
    select_md_files,
)

READY_MARKER_DIR = DB_PATH  # READY.{generation_id} marker lives beside the index

# Model
EMBEDDING_DIM = 1024
# Instruction prefix prepended to every query before embedding (Qwen3 instruct format).
# Customize for your domain (the default is generic). Bump QUERY_TEMPLATE_VERSION when changed.
TEXTBOOK_QUERY_PREFIX = os.environ.get(
    "VAULT_SEARCH_TEXTBOOK_QUERY_PREFIX",
    "Instruct: Given a query, retrieve the most relevant reference passages.\nQuery: ",
)
QUERY_TEMPLATE_VERSION = os.environ.get("VAULT_SEARCH_TEXTBOOK_TEMPLATE_VERSION", "qwen3-generic-v1")
OLLAMA_NUM_CTX = 4096  # explicit; default 2048 silently truncates parents > 2K

# Versions
CHUNKING_VERSION = "v2.1"
ALGORITHM_VERSION = "v2.0.0"  # bump on chunking-logic code changes (regex/abbrev/etc)

# Chunking thresholds (Qwen3 tokens)
PARENT_MAX_TOKENS = 1800  # 200-tok safety margin below Ollama internal handling
PARENT_MIN_TOKENS = 100
CHILD_TARGET_TOKENS = 512
CHILD_OVERLAP_TOKENS = 75
CHILD_MIN_CHARS = 200
TABLE_HARD_MAX_TOKENS = 3000
MIN_EMBED_TOKENS = 20  # below this, skip (OCR garbage, ToC fragments)

# Indexing
BATCH_SIZE = 64  # reduced 300→64 (2026-06-05): on the 8GB GPU under sustained
                 # load, 300-input batches (esp. books with large table chunks)
                 # crash the Ollama model runner → connection-refused → whole
                 # batch fails (~30% loss, deterministic on big-chunk books even
                 # with retry). 64 = 0 failures. Bump back up only if you verify
                 # row count == reported Chunks afterwards.
PARENT_BATCH_SIZE = 64  # same rationale as BATCH_SIZE
FLUSH_EVERY_FILES = 50
HASH_SAVE_EVERY_FILES = 100

# Latency watchdog (Gemini r6 #4)
# Tuning history:
#   v1: 10×/3  — too aggressive (Kirshblum false-aborted)
#   v2: 20×/5  — better, but full reindex aborted at 24% on long-parent batches
#   v3: 30×/7  — current. Long-parent batches measured 5-6× normal under load.
#                Need to tolerate >6× while still catching sustained CPU fallback.
WATCHDOG_BASELINE_BATCHES = 10  # bigger sample → baseline less skewed by short-text early batches
WATCHDOG_MULTIPLIER = 30
WATCHDOG_TRIGGER_CONSECUTIVE = 7


class WatchdogAbort(BaseException):
    """Special exception type — bypasses normal `except Exception` so abort is hard."""
    pass

# English abbreviation set for sentence-end heuristic (best-effort, not exhaustive —
# overlap is the real safety net)
ENGLISH_ABBREVS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Jr.", "Sr.",
    "e.g.", "i.e.", "et al.", "etc.", "vs.", "approx.",
    "Fig.", "Eq.", "No.", "Vol.", "Ch.", "p.", "pp.",
    "cm.", "mm.", "kg.", "mg.", "mL.", "mmHg.",
    "7e.", "6e.", "5e.", "4e.",
}


# =============================================================================
# Schemas — symmetric provenance fields on chunks AND parents
# =============================================================================

CHUNKS_SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("parent_id", pa.utf8()),
    pa.field("file", pa.utf8()),
    pa.field("book", pa.utf8()),
    pa.field("chapter", pa.utf8()),
    pa.field("section_path", pa.utf8()),
    pa.field("section_path_raw", pa.utf8()),
    pa.field("page_start", pa.int32()),
    pa.field("page_end", pa.int32()),
    pa.field("chunk_kind", pa.utf8()),
    pa.field("chunk_idx", pa.int32()),
    pa.field("n_siblings", pa.int32()),
    pa.field("token_count", pa.int32()),
    pa.field("heading_origin", pa.utf8()),
    pa.field("figure_ids", pa.list_(pa.utf8())),
    pa.field("has_figure_ref", pa.bool_()),
    pa.field("chunking_version", pa.utf8()),
    pa.field("embedding_model", pa.utf8()),
    pa.field("query_template_version", pa.utf8()),
    pa.field("indexing_signature", pa.utf8()),
    pa.field("mtime", pa.float64()),
])

PARENTS_SCHEMA = pa.schema([
    pa.field("parent_id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("file", pa.utf8()),
    pa.field("book", pa.utf8()),
    pa.field("chapter", pa.utf8()),
    pa.field("section_path", pa.utf8()),
    pa.field("section_path_raw", pa.utf8()),
    pa.field("page_start", pa.int32()),
    pa.field("page_end", pa.int32()),
    pa.field("token_count", pa.int32()),
    pa.field("heading_origin", pa.utf8()),
    pa.field("figure_ids", pa.list_(pa.utf8())),
    pa.field("has_figure_ref", pa.bool_()),
    pa.field("chunking_version", pa.utf8()),
    pa.field("embedding_model", pa.utf8()),
    pa.field("query_template_version", pa.utf8()),
    pa.field("indexing_signature", pa.utf8()),
    pa.field("mtime", pa.float64()),
])

CHUNKS_TABLE = "textbook_chunks"
PARENTS_TABLE = "textbook_parents"


# =============================================================================
# Tokenizer (lazy global; fast Rust backend)
# =============================================================================

_tokenizer = None
TOKENIZER_REPO = "Qwen/Qwen3-Embedding-0.6B"


class _FastTokenizer:
    """Adapter over `tokenizers.Tokenizer` with the AutoTokenizer call shape the
    rest of this codebase uses (`encode` -> list[int], `decode` -> str).

    transformers drags torch in on import: +371 MB of private bytes for what is
    pure Rust tokenization. The `tokenizers` backend costs ~58 MB and never
    imports torch. Equivalence verified before the swap on 300 real parent texts
    plus empty / whitespace / CJK / emoji / 5000-char inputs: identical ids and
    identical decode output, zero mismatches. Do not "simplify" this back to
    AutoTokenizer.
    """

    def __init__(self, backend):
        self._backend = backend

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._backend.decode(list(ids), skip_special_tokens=skip_special_tokens)


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        cached = None
        try:
            from huggingface_hub import try_to_load_from_cache
            hit = try_to_load_from_cache(TOKENIZER_REPO, "tokenizer.json")
            cached = hit if isinstance(hit, str) else None
        except Exception:
            cached = None
        # Prefer the local HF cache so an offline box still starts.
        backend = (
            Tokenizer.from_file(cached) if cached
            else Tokenizer.from_pretrained(TOKENIZER_REPO)
        )
        _tokenizer = _FastTokenizer(backend)
    return _tokenizer


def tok_len(text: str) -> int:
    """Qwen3 token count for canonical thresholds (NOT char-based)."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# =============================================================================
# Helpers
# =============================================================================

def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def normalize_section_path(s: str) -> str:
    """NFKC + collapse whitespace + strip. Preserves case."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_indexing_signature() -> str:
    payload = "|".join([
        ALGORITHM_VERSION,
        CHUNKING_VERSION,
        TEXTBOOK_EMBEDDING_MODEL,
        QUERY_TEMPLATE_VERSION,
        f"CHILD_TARGET={CHILD_TARGET_TOKENS}",
        f"CHILD_OVERLAP={CHILD_OVERLAP_TOKENS}",
        f"PARENT_MAX={PARENT_MAX_TOKENS}",
        f"PARENT_MIN={PARENT_MIN_TOKENS}",
        f"TABLE_MAX={TABLE_HARD_MAX_TOKENS}",
        f"MIN_EMBED={MIN_EMBED_TOKENS}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Figure ID regexes
_FIG_REF_PATTERNS = [
    re.compile(r"<!--\s*REF:\s*(?:Fig(?:ure)?\.?\s*\d+[-\.]\d+\w?)", re.IGNORECASE),
    re.compile(r"\b(?:Fig(?:ure)?\.?\s*\d+[-\.]\d+\w?)", re.IGNORECASE),
    re.compile(r"圖\s*\d+[-\.]\d+\w?"),
]
_FIG_ID_NORMALIZE = re.compile(r"\bfig(?:ure)?\.?\s*", re.IGNORECASE)


def extract_figure_ids(text: str) -> list[str]:
    """Extract distinct figure IDs from chunk text (markers + inline mentions)."""
    found: set[str] = set()
    for pat in _FIG_REF_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            # strip "<!-- REF: " prefix if present
            raw = re.sub(r"<!--\s*REF:\s*", "", raw)
            # normalize "Figure 32-4" / "fig. 32.4" / "Fig 32-4" → "Fig 32-4"
            norm = _FIG_ID_NORMALIZE.sub("Fig ", raw).strip().rstrip(".,;:")
            # also normalize separator . → -
            norm = re.sub(r"(\d)\.(\d)", r"\1-\2", norm)
            found.add(norm)
    return sorted(found)


# =============================================================================
# Markdown parser — extract pages + heading hierarchy
# =============================================================================

PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_md_pages_and_headings(content: str) -> dict:
    """Parse markdown, return:
        {
          "lines": [(line_str, page_int_or_0), ...],
          "headings": [{"level": 1..3, "text": str, "line_idx": int, "page": int}, ...],
          "first_h1": str | None
        }
    Pages numbered from <!-- page N --> markers; lines before any marker get page=0.
    """
    lines = content.split("\n")
    out_lines: list[tuple[str, int]] = []
    headings: list[dict] = []
    current_page = 0
    first_h1: str | None = None

    for i, line in enumerate(lines):
        # Detect page marker: it occupies its own line typically
        m = PAGE_MARKER_RE.search(line)
        if m:
            current_page = int(m.group(1))
            # Don't emit the marker line itself as content
            out_lines.append(("", current_page))
            continue

        out_lines.append((line, current_page))

        h = HEADING_RE.match(line)
        if h and not (line.lstrip().startswith("```")):
            level = len(h.group(1))
            if level <= 3:
                heading_text = h.group(2).strip()
                if level == 1 and first_h1 is None:
                    first_h1 = heading_text
                headings.append({
                    "level": level,
                    "text": heading_text,
                    "line_idx": i,
                    "page": current_page,
                })

    return {"lines": out_lines, "headings": headings, "first_h1": first_h1}


# =============================================================================
# Parent / child builder — deterministic 3-tier rule
# =============================================================================

def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|") and "|" in line.lstrip()[1:]


def _detect_table_block(lines_pages: list[tuple[str, int]], start: int) -> tuple[int, int] | None:
    """If lines_pages[start] starts a markdown table, return (start, end_exclusive). Else None.
    A 'table' is >= 2 consecutive lines starting with `|`."""
    n = len(lines_pages)
    if start >= n:
        return None
    if not _is_table_line(lines_pages[start][0]):
        return None
    end = start + 1
    while end < n and _is_table_line(lines_pages[end][0]):
        end += 1
    if end - start >= 2:
        return (start, end)
    return None


def _slice_text(lines_pages: list[tuple[str, int]], start: int, end_exclusive: int) -> tuple[str, int, int]:
    """Return (text, page_start, page_end) for a slice."""
    if end_exclusive <= start:
        return ("", 0, 0)
    seg = lines_pages[start:end_exclusive]
    page_seen = [p for _, p in seg if p > 0]
    page_start = page_seen[0] if page_seen else 0
    page_end = page_seen[-1] if page_seen else 0
    text = "\n".join(s for s, _ in seg).strip("\n")
    return (text, page_start, page_end)


def _split_paragraphs_to_target(
    text: str, page_start: int, page_end: int, target_tokens: int
) -> list[dict]:
    """Pack \\n\\n-separated paragraphs into sub-blocks of ~target_tokens each.
    Returns [{"text", "page_start", "page_end"}, ...]"""
    # We don't have page resolution per paragraph here, so we approximate by reusing
    # the slice's page span. This is acceptable for parents that span pages — page_start/end
    # comes from the parent slice anyway.
    paragraphs = re.split(r"\n\n+", text)
    out: list[dict] = []
    cur = ""
    cur_tok = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        p_tok = tok_len(p)
        if cur and cur_tok + p_tok > target_tokens:
            out.append({"text": cur, "page_start": page_start, "page_end": page_end})
            cur = p
            cur_tok = p_tok
        else:
            cur = (cur + "\n\n" + p) if cur else p
            cur_tok += p_tok
    if cur.strip():
        out.append({"text": cur, "page_start": page_start, "page_end": page_end})
    return out


def build_parents(
    lines_pages: list[tuple[str, int]],
    headings: list[dict],
    book: str,
    file_rel: str,
    chapter: str,
    file_mtime: float,
) -> list[dict]:
    """Build parent records. Returns list of dicts with keys:
       text, section_path, section_path_raw, page_start, page_end, heading_origin, ordinal.
    parent_id assigned later (after we know heading_origin).
    """
    n_lines = len(lines_pages)

    # Determine subsection boundaries — H2 spans by default, fall back to file-level if no H2.
    h2_indices = [h for h in headings if h["level"] == 2]
    h3_indices = [h for h in headings if h["level"] == 3]

    sections: list[dict] = []  # {start, end, path_stack, heading_origin}

    if not h2_indices:
        # Rule (e): no H2s at all — fall back to page-block parents
        # Make a parent every ~1024 tokens of content.
        content_text, ps, pe = _slice_text(lines_pages, 0, n_lines)
        if content_text.strip():
            blocks = _split_paragraphs_to_target(content_text, ps, pe, 1024)
            for i, b in enumerate(blocks):
                sections.append({
                    "text": b["text"],
                    "page_start": b["page_start"],
                    "page_end": b["page_end"],
                    "h1_text": chapter or "",
                    "h2_text": "",
                    "h3_text": "",
                    "section_path_raw_parts": [],
                    "heading_origin": "page_fallback",
                })
        return _finalize_parents(sections, book, file_rel, chapter, file_mtime)

    # Build H2 segments
    raw_segments: list[dict] = []
    for i, h2 in enumerate(h2_indices):
        seg_start = h2["line_idx"]
        seg_end = h2_indices[i + 1]["line_idx"] if i + 1 < len(h2_indices) else n_lines
        text, ps, pe = _slice_text(lines_pages, seg_start, seg_end)
        if not text.strip():
            continue
        raw_segments.append({
            "h2": h2,
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "page_start": ps,
            "page_end": pe,
        })

    # Pre-segment area (above first H2) — anything substantive belongs to a "preface-like" parent
    if h2_indices and h2_indices[0]["line_idx"] > 0:
        text, ps, pe = _slice_text(lines_pages, 0, h2_indices[0]["line_idx"])
        if tok_len(text) >= PARENT_MIN_TOKENS:
            sections.append({
                "text": text,
                "page_start": ps,
                "page_end": pe,
                "h1_text": chapter or "",
                "h2_text": "",
                "h3_text": "",
                "section_path_raw_parts": [],
                "heading_origin": "H2",  # treat as a degenerate H2-equivalent
            })

    # For each H2 segment, decide split rule
    for seg in raw_segments:
        seg_text = seg["text"]
        seg_tokens = tok_len(seg_text)
        seg_h2 = seg["h2"]

        # Check if this segment contains an oversized table that should keep parent atomic
        contained_h3s = [h for h in h3_indices if seg["start"] <= h["line_idx"] < seg["end"]]

        if seg_tokens <= PARENT_MAX_TOKENS:
            # Rule (a): default — H2 = 1 parent
            sections.append({
                "text": seg_text,
                "page_start": seg["page_start"],
                "page_end": seg["page_end"],
                "h1_text": chapter or "",
                "h2_text": seg_h2["text"],
                "h3_text": "",
                "section_path_raw_parts": [seg_h2["text"]],
                "heading_origin": "H2",
            })
            continue

        # Rule (b): check if oversize is solely a single big table
        # Walk the segment — if it contains a table block whose token-count is itself
        # > PARENT_MAX_TOKENS but ≤ TABLE_HARD_MAX_TOKENS, accept the oversized parent.
        # (Body-text oversize → fall to H3 split.)
        big_table = _find_oversized_atomic_table(lines_pages, seg["start"], seg["end"])
        if big_table is not None and tok_len(big_table) <= TABLE_HARD_MAX_TOKENS:
            sections.append({
                "text": seg_text,
                "page_start": seg["page_start"],
                "page_end": seg["page_end"],
                "h1_text": chapter or "",
                "h2_text": seg_h2["text"],
                "h3_text": "",
                "section_path_raw_parts": [seg_h2["text"]],
                "heading_origin": "H2",  # kept atomic, still H2-origin
            })
            continue

        # Rule (b'): split by H3
        if contained_h3s:
            # Pre-H3 area (between H2 line and first H3)
            pre_start = seg_h2["line_idx"] + 1  # skip the H2 line itself
            first_h3_line = contained_h3s[0]["line_idx"]
            if first_h3_line > pre_start:
                pre_text, pps, ppe = _slice_text(lines_pages, pre_start, first_h3_line)
                if tok_len(pre_text) >= PARENT_MIN_TOKENS:
                    sections.append({
                        "text": pre_text,
                        "page_start": pps,
                        "page_end": ppe,
                        "h1_text": chapter or "",
                        "h2_text": seg_h2["text"],
                        "h3_text": "",
                        "section_path_raw_parts": [seg_h2["text"]],
                        "heading_origin": "H3",  # came from an H2-with-H3-split context
                    })

            for j, h3 in enumerate(contained_h3s):
                h3_start = h3["line_idx"]
                h3_end = contained_h3s[j + 1]["line_idx"] if j + 1 < len(contained_h3s) else seg["end"]
                h3_text, hps, hpe = _slice_text(lines_pages, h3_start, h3_end)
                h3_tokens = tok_len(h3_text)

                if h3_tokens <= PARENT_MAX_TOKENS:
                    sections.append({
                        "text": h3_text,
                        "page_start": hps,
                        "page_end": hpe,
                        "h1_text": chapter or "",
                        "h2_text": seg_h2["text"],
                        "h3_text": h3["text"],
                        "section_path_raw_parts": [seg_h2["text"], h3["text"]],
                        "heading_origin": "H3",
                    })
                else:
                    # Rule (c): paragraph split
                    blocks = _split_paragraphs_to_target(h3_text, hps, hpe, 1024)
                    for k, b in enumerate(blocks):
                        sections.append({
                            "text": b["text"],
                            "page_start": b["page_start"],
                            "page_end": b["page_end"],
                            "h1_text": chapter or "",
                            "h2_text": seg_h2["text"],
                            "h3_text": h3["text"],
                            "section_path_raw_parts": [seg_h2["text"], h3["text"]],
                            "heading_origin": "paragraph_split",
                        })
        else:
            # No H3s → recursive paragraph split
            blocks = _split_paragraphs_to_target(seg_text, seg["page_start"], seg["page_end"], 1024)
            for k, b in enumerate(blocks):
                sections.append({
                    "text": b["text"],
                    "page_start": b["page_start"],
                    "page_end": b["page_end"],
                    "h1_text": chapter or "",
                    "h2_text": seg_h2["text"],
                    "h3_text": "",
                    "section_path_raw_parts": [seg_h2["text"]],
                    "heading_origin": "paragraph_split",
                })

    # Rule (d): merge adjacent micro-parents
    sections = _merge_small_adjacent_parents(sections)

    return _finalize_parents(sections, book, file_rel, chapter, file_mtime)


def _find_oversized_atomic_table(
    lines_pages: list[tuple[str, int]], start: int, end: int
) -> str | None:
    """Scan slice for a single table block that itself exceeds PARENT_MAX_TOKENS.
    Return its text if found, else None."""
    i = start
    while i < end:
        block = _detect_table_block(lines_pages, i)
        if block is None:
            i += 1
            continue
        bs, be = block
        if be > end:
            be = end
        text, _, _ = _slice_text(lines_pages, bs, be)
        if tok_len(text) > PARENT_MAX_TOKENS:
            return text
        i = be
    return None


def _merge_small_adjacent_parents(sections: list[dict]) -> list[dict]:
    """Merge pairs of adjacent parents both below PARENT_MIN_TOKENS into one."""
    if len(sections) < 2:
        return sections
    out: list[dict] = []
    i = 0
    while i < len(sections):
        cur = sections[i]
        # If current is small AND next is small AND they share an h2 ancestor, merge
        if (
            i + 1 < len(sections)
            and tok_len(cur["text"]) < PARENT_MIN_TOKENS
            and tok_len(sections[i + 1]["text"]) < PARENT_MIN_TOKENS
            and cur["h2_text"] == sections[i + 1]["h2_text"]
        ):
            nxt = sections[i + 1]
            merged = {
                **cur,
                "text": cur["text"] + "\n\n" + nxt["text"],
                "page_start": cur["page_start"] or nxt["page_start"],
                "page_end": nxt["page_end"] or cur["page_end"],
                "h3_text": "" if cur["h3_text"] != nxt["h3_text"] else cur["h3_text"],
                "section_path_raw_parts": cur["section_path_raw_parts"]
                if cur["h3_text"] == nxt["h3_text"]
                else cur["section_path_raw_parts"][:1],  # drop H3 if differing
                "heading_origin": "merged_small_parent",
            }
            out.append(merged)
            i += 2
        else:
            out.append(cur)
            i += 1
    return out


def _finalize_parents(
    sections: list[dict], book: str, file_rel: str, chapter: str, file_mtime: float
) -> list[dict]:
    """Compute parent_id, section_path strings, token_count, figure_ids, etc."""
    out = []
    for ordinal, sec in enumerate(sections):
        section_path_raw = " > ".join(sec.get("section_path_raw_parts") or [])
        section_path = normalize_section_path(section_path_raw)
        h_origin = sec["heading_origin"]
        text = sec["text"].strip()
        if not text or tok_len(text) < MIN_EMBED_TOKENS:
            continue
        pid_raw = "|".join([
            book, file_rel,
            sec["h1_text"], sec["h2_text"], sec["h3_text"],
            str(sec["page_start"]), str(ordinal),
            h_origin,
        ])
        parent_id = hashlib.md5(pid_raw.encode("utf-8", errors="replace")).hexdigest()
        fig_ids = extract_figure_ids(text)
        out.append({
            "parent_id": parent_id,
            "text": text,
            "file": file_rel,
            "book": book,
            "chapter": chapter,
            "section_path": section_path,
            "section_path_raw": section_path_raw,
            "page_start": int(sec["page_start"] or 0),
            "page_end": int(sec["page_end"] or 0),
            "token_count": tok_len(text),
            "heading_origin": h_origin,
            "figure_ids": fig_ids,
            "has_figure_ref": bool(fig_ids),
            "mtime": file_mtime,
        })
    return out


# =============================================================================
# Children — recursive splitter with atomicity precedence
# =============================================================================

# Sentence-end splitting (best-effort)
_ZH_SENT_END = re.compile(r"(?<=[。！？；])")
_EN_SENT_END_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z一-鿿])")


def _split_at_sentences(text: str) -> list[str]:
    """Best-effort sentence split; abbrev list handled by post-filter."""
    # First split by Chinese punctuation
    parts: list[str] = []
    for chunk in _ZH_SENT_END.split(text):
        if chunk:
            # Then split by English sentence boundaries
            for sub in _EN_SENT_END_RE.split(chunk):
                if sub.strip():
                    parts.append(sub)
    # Heal abbreviations: if a part ends with an abbrev, merge with next
    merged: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i].strip()
        last_word = cur.split()[-1] if cur.split() else ""
        if last_word in ENGLISH_ABBREVS and i + 1 < len(parts):
            cur = cur + " " + parts[i + 1].strip()
            i += 2
        else:
            i += 1
        merged.append(cur)
    return [p for p in merged if p.strip()]


def _build_children_from_text(
    text: str,
    parent_page_start: int,
    parent_page_end: int,
    target_tokens: int = CHILD_TARGET_TOKENS,
    overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> list[dict]:
    """Recursive splitter (atomicity-first):
      1. TABLE block atomic (handled separately by table_fragment caller)
      2. \\n\\n paragraph
      3. \\n line
      4. sentence-end
      5. word
      6. char
    Returns list of {"text", "token_count", "page_start", "page_end"}."""
    if not text.strip():
        return []
    if tok_len(text) <= target_tokens:
        return [{
            "text": text,
            "token_count": tok_len(text),
            "page_start": parent_page_start,
            "page_end": parent_page_end,
        }]

    # Try paragraph split
    paragraphs = re.split(r"\n\n+", text)
    units: list[str]
    if len(paragraphs) > 1:
        units = paragraphs
    else:
        # Try line split
        lines = text.split("\n")
        if len(lines) > 1:
            units = lines
        else:
            # Try sentence split
            units = _split_at_sentences(text)
            if len(units) <= 1:
                # Fall back to word split
                words = text.split()
                if len(words) > 1:
                    units = words
                else:
                    # Last resort: char chunks
                    units = list(text)

    chunks: list[dict] = []
    cur = ""
    cur_tok = 0
    for u in units:
        u = u if isinstance(u, str) else str(u)
        u_tok = tok_len(u)
        if cur and cur_tok + u_tok > target_tokens:
            chunks.append({
                "text": cur,
                "token_count": cur_tok,
                "page_start": parent_page_start,
                "page_end": parent_page_end,
            })
            # overlap: prepend last `overlap_tokens` worth of cur to next
            if overlap_tokens > 0:
                cur_words = cur.split()
                # approximate overlap by taking trailing N tokens worth of words
                # we don't know exact word→token mapping cheaply; use 4×words as proxy
                prefix_words = cur_words[-max(1, overlap_tokens // 2):]
                cur = " ".join(prefix_words) + ("\n" if "\n" in u else " ") + u
                cur_tok = tok_len(cur)
            else:
                cur = u
                cur_tok = u_tok
        else:
            sep = "\n\n" if u in paragraphs else ("\n" if u in (text.split("\n")) else " ")
            cur = (cur + sep + u) if cur else u
            cur_tok += u_tok

    if cur.strip():
        chunks.append({
            "text": cur,
            "token_count": cur_tok,
            "page_start": parent_page_start,
            "page_end": parent_page_end,
        })

    return chunks


def _classify_chunk_kind(text: str) -> str:
    """Lenient detection: check first 3 lines."""
    lines = text.split("\n")[:3]
    table_lines = sum(1 for l in lines if l.lstrip().startswith("|"))
    if table_lines >= 2:
        # Could be table_fragment if part of a larger table; the indexer marks accordingly
        return "table"
    if any("<!-- REF:" in l for l in lines):
        return "figure_caption"
    return "body"


def _build_table_fragments(
    table_text: str, page_start: int, page_end: int
) -> list[dict]:
    """State machine: extract first 2 lines as header_lines (header + |---| separator),
    iterate body lines, accumulate by token count; on hitting CHILD_TARGET_TOKENS,
    emit fragment as `header + accumulated_body`. Returns list of {"text", "token_count",
    "page_start", "page_end", "is_fragment": bool}."""
    lines = table_text.split("\n")
    if len(lines) < 3:
        # tiny table — return as single non-fragment
        return [{
            "text": table_text,
            "token_count": tok_len(table_text),
            "page_start": page_start,
            "page_end": page_end,
            "is_fragment": False,
        }]
    header_lines = lines[:2]  # row + separator
    header_str = "\n".join(header_lines)
    header_tok = tok_len(header_str)
    body_lines = lines[2:]

    fragments: list[dict] = []
    cur_body: list[str] = []
    cur_tok = 0
    for bl in body_lines:
        bl_tok = tok_len(bl)
        if cur_body and cur_tok + bl_tok + header_tok > CHILD_TARGET_TOKENS:
            frag_text = header_str + "\n" + "\n".join(cur_body)
            fragments.append({
                "text": frag_text,
                "token_count": tok_len(frag_text),
                "page_start": page_start,
                "page_end": page_end,
                "is_fragment": True,
            })
            cur_body = [bl]
            cur_tok = bl_tok
        else:
            cur_body.append(bl)
            cur_tok += bl_tok

    if cur_body:
        frag_text = header_str + "\n" + "\n".join(cur_body)
        fragments.append({
            "text": frag_text,
            "token_count": tok_len(frag_text),
            "page_start": page_start,
            "page_end": page_end,
            "is_fragment": len(fragments) > 0,  # only mark as fragment if not the only one
        })

    return fragments


def build_children_for_parent(parent: dict) -> list[dict]:
    """Build child records for a single parent. Returns list of dicts ready for embedding
    (vector and metadata applied later)."""
    text = parent["text"]
    if tok_len(text) <= CHILD_TARGET_TOKENS:
        # Whole parent fits as a single child
        kind = _classify_chunk_kind(text)
        return [_make_child_dict(text, parent, idx=0, kind=kind, n_siblings=1)]

    # Detect oversized table block — split into table_fragments
    if _classify_chunk_kind(text) == "table" and tok_len(text) > CHILD_TARGET_TOKENS:
        frags = _build_table_fragments(text, parent["page_start"], parent["page_end"])
        out = []
        for i, frag in enumerate(frags):
            kind = "table_fragment" if frag.get("is_fragment") else "table"
            out.append(_make_child_dict(frag["text"], parent, idx=i, kind=kind, n_siblings=len(frags)))
        return out

    # Otherwise, recursive split
    chunks = _build_children_from_text(text, parent["page_start"], parent["page_end"])
    # Re-merge anything below CHILD_MIN_CHARS into previous chunk
    merged: list[dict] = []
    for c in chunks:
        if merged and len(c["text"]) < CHILD_MIN_CHARS:
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + c["text"]
            merged[-1]["token_count"] = tok_len(merged[-1]["text"])
        else:
            merged.append(c)

    out = []
    for i, c in enumerate(merged):
        kind = _classify_chunk_kind(c["text"])
        out.append(_make_child_dict(c["text"], parent, idx=i, kind=kind, n_siblings=len(merged), token_count_override=c["token_count"]))
    return out


def _make_child_dict(
    chunk_text: str,
    parent: dict,
    idx: int,
    kind: str,
    n_siblings: int,
    token_count_override: int | None = None,
) -> dict:
    """Construct a child record. heading_path prefix is added to embed/store text."""
    heading_prefix = f"[{parent['book']} — {parent['chapter']}"
    if parent["section_path"]:
        heading_prefix += f" > {parent['section_path']}"
    heading_prefix += "]\n"
    full_text = heading_prefix + chunk_text
    tcount = token_count_override or tok_len(full_text)
    fig_ids = extract_figure_ids(chunk_text)
    return {
        "text": full_text,
        "parent_id": parent["parent_id"],
        "file": parent["file"],
        "book": parent["book"],
        "chapter": parent["chapter"],
        "section_path": parent["section_path"],
        "section_path_raw": parent["section_path_raw"],
        "page_start": parent["page_start"],
        "page_end": parent["page_end"],
        "chunk_kind": kind,
        "chunk_idx": idx,
        "n_siblings": n_siblings,
        "token_count": tcount,
        "heading_origin": parent["heading_origin"],
        "figure_ids": fig_ids,
        "has_figure_ref": bool(fig_ids),
        "mtime": parent["mtime"],
    }


# =============================================================================
# Embedding (NaN handling + L2 normalize + latency watchdog)
# =============================================================================

def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(v * v for v in vec))
    if s == 0 or math.isnan(s):
        return vec
    return [v / s for v in vec]


class LatencyWatchdog:
    def __init__(self, baseline_n: int, multiplier: int, trigger_consecutive: int):
        self.baseline_n = baseline_n
        self.multiplier = multiplier
        self.trigger = trigger_consecutive
        self.samples: list[float] = []
        self.consecutive_slow = 0
        self.baseline: float | None = None

    def observe(self, dt: float):
        if self.baseline is None:
            self.samples.append(dt)
            if len(self.samples) >= self.baseline_n:
                self.baseline = sum(self.samples) / len(self.samples)
                print(f"[watchdog] baseline established: {self.baseline:.2f}s/batch", flush=True)
            return
        if dt > self.baseline * self.multiplier:
            self.consecutive_slow += 1
            print(
                f"[watchdog] slow batch {self.consecutive_slow}/{self.trigger}: {dt:.1f}s "
                f"({dt / self.baseline:.0f}× baseline)",
                flush=True,
            )
            if self.consecutive_slow >= self.trigger:
                raise WatchdogAbort(
                    f"[FATAL] Batch latency degraded {self.multiplier}× for "
                    f"{self.trigger} consecutive batches — Ollama may have OOMed and "
                    f"silently fell back to CPU. Restart Ollama, verify nvidia-smi shows GPU usage, "
                    f"and re-run with --incremental to resume."
                )
        else:
            self.consecutive_slow = 0


def embed_texts(
    texts: list[str],
    client: ollama.Client,
    error_log: dict,
    chunk_meta: list[dict],
    watchdog: LatencyWatchdog,
) -> list[list[float] | None]:
    """Embed a batch. Returns list of L2-normalized vectors (or None for failures).
    Failed entries logged via upsert_error_log; caller skips them."""
    t0 = time.time()
    last_err = None
    embs = None
    # Retry with backoff: under sustained load the Ollama model runner can
    # intermittently crash/restart, giving a brief window of connection-refused
    # that fails the whole batch. Riding it out converges to 0 failures.
    for attempt in range(4):
        try:
            r = client.embed(
                model=TEXTBOOK_EMBEDDING_MODEL,
                input=texts,
                options={"num_ctx": OLLAMA_NUM_CTX},
            )
            embs = r["embeddings"]
            dt = time.time() - t0
            watchdog.observe(dt)
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s — let runner reload
    if embs is None:
        # All retries exhausted — log each, return Nones
        for meta in chunk_meta:
            upsert_error_log(error_log, meta, "batch_failure", str(last_err))
        return [None] * len(texts)

    out: list[list[float] | None] = []
    for emb, meta in zip(embs, chunk_meta):
        if any(math.isnan(v) for v in emb):
            upsert_error_log(error_log, meta, "NaN", "embedding contained NaN values")
            out.append(None)
        else:
            out.append(_l2_normalize(emb))
    return out


# =============================================================================
# Error log lifecycle
# =============================================================================

def load_error_log() -> dict:
    if ERROR_LOG_PATH.exists():
        try:
            return json.loads(ERROR_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_error_log(log: dict):
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_error_log(log: dict, meta: dict, error_type: str, error_msg: str):
    """meta should have file, parent_id, chunk_idx, text (or text_preview)."""
    eid = f"{meta.get('file','?')}:{meta.get('parent_id','?')}:{meta.get('chunk_idx','?')}"
    now = now_iso()
    if eid in log:
        log[eid]["last_seen"] = now
        log[eid]["retry_count"] = log[eid].get("retry_count", 0) + 1
        if log[eid]["retry_count"] >= 3:
            log[eid]["status"] = "permanent"
        log[eid]["error_type"] = error_type
        log[eid]["error_msg"] = error_msg
    else:
        text_preview = (meta.get("text") or meta.get("text_preview") or "")[:200]
        log[eid] = {
            "id": eid,
            "status": "pending",
            "first_seen": now,
            "last_seen": now,
            "retry_count": 0,
            "file": meta.get("file"),
            "parent_id": meta.get("parent_id"),
            "chunk_idx": meta.get("chunk_idx"),
            "text_preview": text_preview,
            "error_type": error_type,
            "error_msg": error_msg,
        }


def mark_error_fixed(log: dict, meta: dict):
    eid = f"{meta.get('file','?')}:{meta.get('parent_id','?')}:{meta.get('chunk_idx','?')}"
    if eid in log:
        log[eid]["status"] = "fixed"
        log[eid]["last_seen"] = now_iso()


def archive_old_fixed_entries(log: dict, days: int = 30):
    """Move status=fixed entries older than `days` to the archive file."""
    if not log:
        return log
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    archive: dict = {}
    if ERROR_LOG_ARCHIVE_PATH.exists():
        try:
            archive = json.loads(ERROR_LOG_ARCHIVE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    keep: dict = {}
    moved = 0
    for eid, e in log.items():
        if e.get("status") == "fixed":
            try:
                ts = datetime.fromisoformat(e.get("last_seen", "").replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = 0
            if ts < cutoff:
                archive[eid] = e
                moved += 1
                continue
        keep[eid] = e
    if moved:
        ERROR_LOG_ARCHIVE_PATH.write_text(
            json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[errors] archived {moved} fixed entries", flush=True)
    return keep


# =============================================================================
# Hash cache (file-level)
# =============================================================================

def load_hash_cache() -> dict:
    if HASH_CACHE_PATH.exists():
        try:
            return json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_hash_cache(cache: dict):
    HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main indexer
# =============================================================================

def get_or_create_table(db, name: str, schema: pa.Schema):
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema, mode="overwrite")


def write_ready_marker(generation_id: str):
    """Atomic READY marker for parent_map cache invalidation."""
    READY_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up old READY markers
    for old in READY_MARKER_DIR.glob("READY.*"):
        try:
            old.unlink()
        except Exception:
            pass
    marker = READY_MARKER_DIR / f"READY.{generation_id}"
    marker.write_text(now_iso(), encoding="utf-8")
    print(f"[ok] READY marker written: {marker.name}", flush=True)


def orphan_cleanup(chunks_table, parents_table, current_files: set[str]):
    """Remove rows whose `file` no longer exists on disk. Batch + compact."""
    deleted_total = 0
    for table, label in [(chunks_table, "chunks"), (parents_table, "parents")]:
        try:
            df = table.to_pandas(columns=["file"]) if hasattr(table, "to_pandas") else None
        except Exception:
            df = None
        # Get distinct files via search().select() for robustness
        try:
            df = table.search().select(["file"]).limit(10**9).to_pandas()
        except Exception as e:
            print(f"[orphan] could not enumerate {label}: {e}", flush=True)
            continue
        if df is None or df.empty:
            continue
        existing_files = set(df["file"].unique())
        gone = existing_files - current_files
        if gone:
            quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in gone)
            try:
                table.delete(f"file IN ({quoted})")
                deleted_total += len(gone)
                print(f"[orphan] deleted {len(gone)} orphan files from {label}", flush=True)
            except Exception as e:
                print(f"[orphan] delete failed in {label}: {e}", flush=True)
    if deleted_total > 0:
        for table in (chunks_table, parents_table):
            try:
                table.compact_files()
            except Exception:
                pass


def index_textbooks(
    incremental: bool = False,
    book_filter: str | None = None,
    force: bool = False,
):
    """Main entry point. Builds parents + children, embeds, writes to LanceDB."""
    print(f"[start] textbook_indexer v2 — {now_iso()}", flush=True)
    print(f"  incremental={incremental} book_filter={book_filter} force={force}", flush=True)

    db = lancedb.connect(str(DB_PATH))
    chunks_table = get_or_create_table(db, CHUNKS_TABLE, CHUNKS_SCHEMA)
    parents_table = get_or_create_table(db, PARENTS_TABLE, PARENTS_SCHEMA)

    sig = compute_indexing_signature()
    print(f"  indexing_signature={sig}", flush=True)

    # Check existing corpus signature
    if incremental and not force:
        try:
            sample = chunks_table.search().select(["indexing_signature"]).limit(1).to_pandas()
            if not sample.empty:
                old_sig = sample["indexing_signature"].iloc[0]
                if old_sig != sig:
                    print(
                        f"[WARN] indexing_signature mismatch (old={old_sig}, new={sig}). "
                        f"Re-run without --incremental for full rebuild, "
                        f"or pass --force to overlay (NOT recommended).",
                        flush=True,
                    )
                    sys.exit(2)
        except Exception:
            pass

    # Find files
    if book_filter:
        book_dir = TEXTBOOK_PATH / book_filter
        if not book_dir.exists():
            print(f"[ERR] Book dir not found: {book_dir}", flush=True)
            sys.exit(1)
        all_md = list(book_dir.rglob("*.md"))
    else:
        all_md = list(TEXTBOOK_PATH.rglob("*.md"))
    md_files, selection_notes = select_md_files(all_md)
    print(f"  found {len(md_files)} md files", flush=True)
    if selection_notes["whole_book_skipped"]:
        print(
            f"  [skip] {selection_notes['whole_book_skipped']} whole-book file(s) "
            f"({', '.join(sorted(WHOLE_BOOK_FILES))}) skipped — chapter siblings "
            f"cover the same pages. Set VAULT_SEARCH_TEXTBOOK_WHOLE_BOOK_FILES= "
            f"to index them anyway.",
            flush=True,
        )
    for parent in selection_notes["guard_kept"]:
        print(
            f"  [WARN] VAULT_SEARCH_TEXTBOOK_SKIP_FILES would leave "
            f"{parent} with nothing to index — keeping its files rather than "
            f"dropping the book.",
            flush=True,
        )

    # Orphan cleanup (only on full / no-book run)
    if not book_filter:
        current_files_rel = {
            str(f.relative_to(TEXTBOOK_PATH)).replace("\\", "/") for f in md_files
        }
        orphan_cleanup(chunks_table, parents_table, current_files_rel)

    # Hash cache (load if incremental or book_filter; otherwise start fresh)
    hash_cache = load_hash_cache() if (incremental or book_filter) else {}
    error_log = load_error_log()
    error_log = archive_old_fixed_entries(error_log)
    save_error_log(error_log)

    # If --book, delete existing rows for that book first (preserves others)
    if book_filter:
        try:
            chunks_table.delete(f"book = '{book_filter}'")
            parents_table.delete(f"book = '{book_filter}'")
            print(f"[book] cleared existing rows for book='{book_filter}'", flush=True)
        except Exception as e:
            print(f"[WARN] could not clear existing rows for {book_filter}: {e}", flush=True)

    client = ollama.Client(host=OLLAMA_HOST)
    watchdog = LatencyWatchdog(WATCHDOG_BASELINE_BATCHES, WATCHDOG_MULTIPLIER, WATCHDOG_TRIGGER_CONSECUTIVE)

    pending_chunk_records: list[dict] = []
    pending_parent_records: list[dict] = []
    files_processed = 0
    files_skipped = 0
    chunks_total = 0
    parents_total = 0
    embed_failed = 0
    skipped_short = 0
    t_start = time.time()

    # === Producer thread does file I/O + chunking + tokenizer work in parallel ===
    # Main thread does GPU embed + LanceDB writes. The two pipeline naturally
    # because Ollama HTTP and LanceDB I/O release the GIL.
    import queue as _queue
    import threading as _threading

    work_queue: _queue.Queue = _queue.Queue(maxsize=4)  # buffer 4 chunked files ahead
    _SENTINEL = object()
    _producer_error: list = []

    def _produce():
        try:
            local_skipped_short = 0
            for md_file in md_files:
                rel_path = str(md_file.relative_to(TEXTBOOK_PATH)).replace("\\", "/")
                try:
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    work_queue.put(("read_error", rel_path, str(e)))
                    continue
                h = file_hash(content)
                if (incremental or book_filter) and hash_cache.get(rel_path) == h and not force:
                    work_queue.put(("skip_cached", rel_path))
                    continue
                book = rel_path.split("/")[0] if "/" in rel_path else "misc"
                parsed = parse_md_pages_and_headings(content)
                chapter = parsed["first_h1"] or md_file.stem.replace("_", " ")
                file_mtime = md_file.stat().st_mtime
                parents = build_parents(
                    parsed["lines"], parsed["headings"], book, rel_path, chapter, file_mtime
                )
                if not parents:
                    work_queue.put(("empty", rel_path, h))
                    continue
                all_children = []
                for p in parents:
                    all_children.extend(build_children_for_parent(p))
                kept_children = []
                short_here = 0
                for c in all_children:
                    if c["token_count"] < MIN_EMBED_TOKENS:
                        short_here += 1
                        continue
                    kept_children.append(c)
                parent_embed_texts = []
                for p in parents:
                    heading_prefix = f"[{p['book']} — {p['chapter']}"
                    if p["section_path"]:
                        heading_prefix += f" > {p['section_path']}"
                    heading_prefix += "]\n"
                    parent_embed_texts.append(heading_prefix + p["text"])
                work_queue.put((
                    "ok", rel_path, h, parents, kept_children, parent_embed_texts, short_here
                ))
        except Exception as e:
            _producer_error.append(e)
        finally:
            work_queue.put(_SENTINEL)

    producer_thread = _threading.Thread(target=_produce, daemon=True)
    producer_thread.start()

    # Cross-file batch accumulators (Phase 2 optimization).
    # We collect parents and children from multiple files until the buffer hits
    # BATCH_SIZE / PARENT_BATCH_SIZE, then send one big batch to Ollama.
    # This saturates the GPU (batch=300 children gives ~120 texts/s vs ~10/s
    # for tiny per-file batches).
    # Each pending entry tracks its source file so we can mark file-level
    # success/failure after all the file's entities are embedded.
    pending_parents: list[tuple[str, dict, str]] = []  # (rel_path, parent_dict, embed_text)
    pending_children: list[tuple[str, dict]] = []  # (rel_path, child_dict)
    # Per-file accounting: counts of expected vs successful entities
    file_state: dict[str, dict] = {}  # rel_path → {h, expected_p, expected_c, ok_p, ok_c, failed}

    def _flush_parents():
        if not pending_parents:
            return
        nonlocal embed_failed, parents_total
        batch_meta = [
            {"file": rp, "parent_id": p["parent_id"], "chunk_idx": -1, "text": p["text"]}
            for rp, p, _ in pending_parents
        ]
        texts = [t for _, _, t in pending_parents]
        embs = embed_texts(texts, client, error_log, batch_meta, watchdog)
        for (rp, p, _), emb in zip(pending_parents, embs):
            st = file_state[rp]
            st["done_p"] += 1
            if emb is None:
                embed_failed += 1
                st["failed"] = True
                continue
            pending_parent_records.append({
                "parent_id": p["parent_id"], "vector": emb, "text": p["text"],
                "file": p["file"], "book": p["book"], "chapter": p["chapter"],
                "section_path": p["section_path"], "section_path_raw": p["section_path_raw"],
                "page_start": p["page_start"], "page_end": p["page_end"],
                "token_count": p["token_count"], "heading_origin": p["heading_origin"],
                "figure_ids": p["figure_ids"], "has_figure_ref": p["has_figure_ref"],
                "chunking_version": CHUNKING_VERSION,
                "embedding_model": TEXTBOOK_EMBEDDING_MODEL,
                "query_template_version": QUERY_TEMPLATE_VERSION,
                "indexing_signature": sig, "mtime": p["mtime"],
            })
            st["ok_p"] += 1
            parents_total += 1
            mark_error_fixed(error_log, {"file": p["file"], "parent_id": p["parent_id"], "chunk_idx": -1})
        pending_parents.clear()

    def _flush_children():
        if not pending_children:
            return
        nonlocal embed_failed, chunks_total
        batch_meta = [
            {"file": rp, "parent_id": c["parent_id"], "chunk_idx": c["chunk_idx"], "text": c["text"]}
            for rp, c in pending_children
        ]
        texts = [c["text"] for _, c in pending_children]
        embs = embed_texts(texts, client, error_log, batch_meta, watchdog)
        for (rp, c), emb in zip(pending_children, embs):
            st = file_state[rp]
            st["done_c"] += 1
            if emb is None:
                embed_failed += 1
                st["failed"] = True
                continue
            cid = hashlib.md5(f"{c['file']}|{c['parent_id']}|{c['chunk_idx']}".encode("utf-8")).hexdigest()
            pending_chunk_records.append({
                "id": cid, "vector": emb, "text": c["text"],
                "parent_id": c["parent_id"], "file": c["file"],
                "book": c["book"], "chapter": c["chapter"],
                "section_path": c["section_path"], "section_path_raw": c["section_path_raw"],
                "page_start": c["page_start"], "page_end": c["page_end"],
                "chunk_kind": c["chunk_kind"], "chunk_idx": c["chunk_idx"],
                "n_siblings": c["n_siblings"], "token_count": c["token_count"],
                "heading_origin": c["heading_origin"],
                "figure_ids": c["figure_ids"], "has_figure_ref": c["has_figure_ref"],
                "chunking_version": CHUNKING_VERSION,
                "embedding_model": TEXTBOOK_EMBEDDING_MODEL,
                "query_template_version": QUERY_TEMPLATE_VERSION,
                "indexing_signature": sig, "mtime": c["mtime"],
            })
            st["ok_c"] += 1
            chunks_total += 1
            mark_error_fixed(error_log, {"file": c["file"], "parent_id": c["parent_id"], "chunk_idx": c["chunk_idx"]})
        pending_children.clear()

    def _finalize_completed_files():
        """For files whose all parents+children are accounted for (ok or failed),
        update hash cache if no failures and remove from state tracker."""
        done = []
        for rp, st in file_state.items():
            if st["done_p"] >= st["expected_p"] and st["done_c"] >= st["expected_c"]:
                if not st["failed"]:
                    hash_cache[rp] = st["h"]
                done.append(rp)
        for rp in done:
            del file_state[rp]

    # Main thread consumer loop
    while True:
        item = work_queue.get()
        if item is _SENTINEL:
            break
        kind = item[0]
        if kind == "read_error":
            print(f"  [SKIP] read error: {item[1]}: {item[2]}", flush=True)
            files_skipped += 1
            continue
        if kind == "skip_cached":
            files_skipped += 1
            continue
        if kind == "empty":
            rel_path, h = item[1], item[2]
            files_processed += 1
            hash_cache[rel_path] = h
            continue
        # kind == "ok"
        _, rel_path, h, parents, kept_children, parent_embed_texts, short_here = item
        skipped_short += short_here

        # Delete prior rows for this file (idempotent re-index)
        try:
            esc = rel_path.replace("'", "''")
            chunks_table.delete(f"file = '{esc}'")
            parents_table.delete(f"file = '{esc}'")
        except Exception:
            pass

        # Register file in state tracker
        file_state[rel_path] = {
            "h": h,
            "expected_p": len(parents),
            "expected_c": len(kept_children),
            "ok_p": 0, "ok_c": 0,
            "done_p": 0, "done_c": 0,
            "failed": False,
        }

        # Accumulate into cross-file batches
        for p, embed_text in zip(parents, parent_embed_texts):
            pending_parents.append((rel_path, p, embed_text))
            if len(pending_parents) >= PARENT_BATCH_SIZE:
                _flush_parents()
        for c in kept_children:
            pending_children.append((rel_path, c))
            if len(pending_children) >= BATCH_SIZE:
                _flush_children()

        files_processed += 1

        # Periodic table flush + cache save (NOTE: this is DB-write flush,
        # different from the embed-batch flush above)
        if files_processed % FLUSH_EVERY_FILES == 0:
            # Force embed flush so pending records are complete
            _flush_parents()
            _flush_children()
            _finalize_completed_files()
            if pending_chunk_records:
                chunks_table.add(pending_chunk_records)
                pending_chunk_records = []
            if pending_parent_records:
                parents_table.add(pending_parent_records)
                pending_parent_records = []
            elapsed = time.time() - t_start
            rate = files_processed / max(elapsed, 0.01)
            eta = (len(md_files) - files_processed) / max(rate, 0.001)
            print(
                f"  [{files_processed}/{len(md_files)}] "
                f"chunks={chunks_total} parents={parents_total} "
                f"failed={embed_failed} short={skipped_short} "
                f"rate={rate:.2f} files/s eta={eta/60:.0f} min",
                flush=True,
            )
        if files_processed % HASH_SAVE_EVERY_FILES == 0:
            save_hash_cache(hash_cache)
            save_error_log(error_log)

    # Final embed flush
    _flush_parents()
    _flush_children()
    _finalize_completed_files()

    # Wait for producer thread (it sent sentinel = it's done iterating)
    producer_thread.join(timeout=5)
    if _producer_error:
        print(f"[ERR] producer thread error: {_producer_error[0]}", flush=True)

    # Final flush
    if pending_chunk_records:
        chunks_table.add(pending_chunk_records)
    if pending_parent_records:
        parents_table.add(pending_parent_records)

    save_hash_cache(hash_cache)
    save_error_log(error_log)

    # Compact tables for retrieval performance
    try:
        chunks_table.compact_files()
        parents_table.compact_files()
    except Exception:
        pass

    # READY marker
    generation_id = sig + "_" + str(int(time.time()))
    write_ready_marker(generation_id)

    elapsed = time.time() - t_start
    print(f"\n[done] in {elapsed/60:.1f} min", flush=True)
    print(f"  Files: {files_processed} processed, {files_skipped} skipped", flush=True)
    print(f"  Chunks: {chunks_total} | Parents: {parents_total}", flush=True)
    print(f"  Embed failures: {embed_failed} (see {ERROR_LOG_PATH})", flush=True)
    print(f"  Skipped (< MIN_EMBED_TOKENS): {skipped_short}", flush=True)
    if embed_failed > 0:
        # These are REAL dropped chunks (not in the table), almost always shared-GPU
        # contention crashing the Ollama runner. Re-run with the GPU free to recover them:
        # incremental skips finished files and reprocesses only the ones that had failures.
        print(f"  ⚠ {embed_failed} chunks were dropped (NOT indexed) — usually shared-GPU contention.", flush=True)
        print(f"    Recover: ensure the lecture batch is paused (auto unless --no-gpu-lease), "
              f"then re-run `textbook_indexer.py --incremental`.", flush=True)


# =============================================================================
# Retry / reset CLI commands
# =============================================================================

def cmd_retry(status_filter: str):
    """Retry entries with given status."""
    error_log = load_error_log()
    pending_ids = [eid for eid, e in error_log.items() if e.get("status") == status_filter]
    print(f"[retry] {len(pending_ids)} entries with status={status_filter}", flush=True)
    if not pending_ids:
        return
    print(f"[retry] not yet fully implemented — entries listed below for manual review:")
    for eid in pending_ids[:20]:
        e = error_log[eid]
        print(f"  {eid}: retry_count={e.get('retry_count')} {e.get('error_type')} {e.get('text_preview','')[:80]}")


def cmd_reset(eid: str):
    error_log = load_error_log()
    if eid in error_log:
        error_log[eid]["status"] = "pending"
        error_log[eid]["retry_count"] = 0
        save_error_log(error_log)
        print(f"[reset] {eid} → status=pending", flush=True)
    else:
        print(f"[reset] entry not found: {eid}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def _gpu_lease(action: str) -> tuple[bool, str]:
    """Run `gpu_lease.py <action>` as a subprocess (acquire/release/status). Best-effort:
    a missing script or non-zero exit is reported, never fatal. Returns (ok, stdout)."""
    if not GPU_LEASE_PATH or not GPU_LEASE_PATH.exists():
        return False, ""
    try:
        # acquire may block until the runner exits and frees VRAM (lease default ~40 min)
        r = subprocess.run(
            [sys.executable, str(GPU_LEASE_PATH), action],
            capture_output=True, text=True, timeout=2700,
        )
        return r.returncode == 0, (r.stdout or "")
    except Exception as e:
        return False, f"error: {e}"


def _gpu_lease_state() -> str:
    """Current lecture-batch lease state: RUNNING / PAUSED / STOPPED / UNKNOWN."""
    ok, out = _gpu_lease("status")
    if not ok:
        return "UNKNOWN"
    for token in ("RUNNING", "PAUSED", "STOPPED"):
        if token in out:
            return token
    return "UNKNOWN"


def main():
    p = argparse.ArgumentParser(description="Textbook indexer v2 (parent-child + Qwen3)")
    p.add_argument("--incremental", action="store_true", help="Only changed files (mtime + hash)")
    p.add_argument("--book", help="Filter to one book folder (preserves others)")
    p.add_argument("--force", action="store_true", help="Bypass signature check")
    p.add_argument("--retry-failed", action="store_true", help="Retry status=pending entries")
    p.add_argument("--retry-permanent", action="store_true", help="Retry status=permanent entries")
    p.add_argument("--reset-error-status", metavar="ID", help="Reset one error entry to pending")
    p.add_argument("--no-gpu-lease", action="store_true",
                   help="Do not pause a competing GPU job (configured via VAULT_SEARCH_GPU_LEASE) "
                        "even if it is running (default: auto-pause to avoid shared-GPU OOM that drops chunks)")
    args = p.parse_args()

    if args.retry_failed:
        cmd_retry("pending")
    elif args.retry_permanent:
        cmd_retry("permanent")
    elif args.reset_error_status:
        cmd_reset(args.reset_error_status)
    else:
        # State-aware GPU lease: only borrow (pause→resume) the lecture batch if it is
        # actually RUNNING. If it is already PAUSED/STOPPED the GPU is free, so we leave the
        # batch untouched — never relaunch a batch the user deliberately stopped.
        managed_lease = False
        if not args.no_gpu_lease:
            if _gpu_lease_state() == "RUNNING":
                ok, out = _gpu_lease("acquire")
                managed_lease = ok
                print(f"[gpu-lease] acquire: {'ok — lecture batch paused' if ok else 'failed (continuing)'} "
                      f"{out.strip()[-160:]}", flush=True)
        try:
            index_textbooks(
                incremental=args.incremental,
                book_filter=args.book,
                force=args.force,
            )
        finally:
            if managed_lease:
                ok, out = _gpu_lease("release")
                print(f"[gpu-lease] release: {'ok — lecture batch resumed' if ok else 'FAILED — check gpu_lease status'} "
                      f"{out.strip()[-160:]}", flush=True)


if __name__ == "__main__":
    main()
