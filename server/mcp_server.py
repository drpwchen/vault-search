"""
Vault Semantic Search MCP Server
Provides semantic search over the Obsidian vault via LanceDB + Ollama embeddings.

Tools:
  - vault_search: Semantic search across all vault notes
  - vault_similar: Find notes similar to a given note
  - vault_stats: Show index statistics

Run as: python mcp_server.py
Register: claude mcp add vault-search -- python "/abs/path/to/server/mcp_server.py"
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path

import lancedb
import ollama

from scoring import rerank
from indexer import open_db, TABLE_NAME, EMBEDDING_MODEL, OLLAMA_HOST, QUERY_PREFIX, _escape_sql
from graph_builder import load_graph, get_neighbors
from ppr import personalized_pagerank, related_notes

# Textbook v2 constants — separate model from vault (vault stays bge-m3)
from textbook_indexer import (
    TEXTBOOK_EMBEDDING_MODEL,
    TEXTBOOK_QUERY_PREFIX,
    QUERY_TEMPLATE_VERSION,
    CHUNKING_VERSION,
    CHUNKS_TABLE,
    PARENTS_TABLE,
    OLLAMA_NUM_CTX,
    DB_PATH as TEXTBOOK_DB_PATH,
    CHILD_TARGET_TOKENS,
    CHILD_OVERLAP_TOKENS,
    get_tokenizer,
    tok_len,
)

# Lazy-loaded graph
_graph_cache = None

# Parent map cache (textbook v2)
_parent_map: dict[str, dict] = {}
_parent_map_mtime: float = 0.0
_parent_map_path = TEXTBOOK_DB_PATH / "textbook_parents.lance"
PARENT_CACHE_COLUMNS = [
    "parent_id", "text", "file", "book", "chapter",
    "section_path", "section_path_raw", "page_start", "page_end",
    "token_count", "heading_origin", "figure_ids", "has_figure_ref",
    "chunking_version",
]

# Search-side parameters
MAX_PARENT_RETURN_TOKENS = 800
ABSOLUTE_PARENT_MAX_RETURN_TOKENS = 4000
MAX_TOTAL_PARENT_RETURN_TOKENS = 12000
MAX_CHILDREN_PER_PARENT = 2
SOFT_BOOK_DIVERSITY_THRESHOLD = 0.5  # apply max_per_book only if >50% from one book
DEFAULT_MAX_PER_BOOK = 3

# Cowork exclusion patterns — substrings marking DERIVATIVE notes (drafts, shared
# co-notes, scratch material) that should be filtered out of search by default.
# Configure via VAULT_SEARCH_EXCLUDE_PATTERNS. Empty by default.
from config import COWORK_PATTERNS, SEARCH_LOG_PATH

def _is_cowork_path(s: str) -> bool:
    """Check if a book name, folder, or file path matches cowork exclusion patterns."""
    if not s:
        return False
    s_lower = s.lower()
    for pat in COWORK_PATTERNS:
        if pat.lower() in s_lower:
            return True
    return False

# Retrieval observability log (thread-safe via RotatingFileHandler)
_search_log_path = SEARCH_LOG_PATH
_search_log_path.parent.mkdir(parents=True, exist_ok=True)
_search_logger = logging.getLogger("textbook_search")
_search_logger.setLevel(logging.INFO)
if not _search_logger.handlers:
    _h = logging.handlers.RotatingFileHandler(
        str(_search_log_path), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _h.setFormatter(logging.Formatter("%(message)s"))
    _search_logger.addHandler(_h)
    _search_logger.propagate = False


def find_relations_for_notes(graph_relations: list[dict], note_names: set[str]) -> list[dict]:
    """Find extracted relations where source_note is in the result set.

    Returns compact relation dicts: {from, to, type, source_note}.
    Limits to 15 relations to avoid output bloat.
    """
    if not graph_relations:
        return []
    matched = []
    for rel in graph_relations:
        if rel.get("source_note") in note_names:
            matched.append({
                "from": rel.get("from", ""),
                "to": rel.get("to", ""),
                "type": rel.get("type", ""),
                "source_note": rel.get("source_note", ""),
            })
    return matched[:15]


def find_entities_for_notes(graph_entities: dict, note_names: set[str]) -> list[dict]:
    """Find extracted entities for result notes.

    Returns compact list: [{note, entities: [{name, type, canonical}]}].
    Limits to 15 entries to avoid output bloat.
    """
    if not graph_entities:
        return []
    matched = []
    for note in note_names:
        ents = graph_entities.get(note, [])
        if ents:
            matched.append({"note": note, "entities": ents[:10]})
    return matched[:15]


def get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = load_graph()
    return _graph_cache


def get_table():
    db = open_db()
    return db.open_table(TABLE_NAME)


def get_textbook_chunks_table():
    db = open_db()
    if CHUNKS_TABLE not in db.table_names():
        return None
    return db.open_table(CHUNKS_TABLE)


def get_textbook_parents_table():
    db = open_db()
    if PARENTS_TABLE not in db.table_names():
        return None
    return db.open_table(PARENTS_TABLE)


def _load_parent_map():
    """Build parent_map from textbook_parents table. Atomic swap (build new dict,
    assign at end) so concurrent searches never see a half-empty map."""
    global _parent_map, _parent_map_mtime
    parents_table = get_textbook_parents_table()
    if parents_table is None:
        return
    # CRITICAL: select metadata columns only — vector column would 4× memory spike.
    df = (
        parents_table.search()
        .select(PARENT_CACHE_COLUMNS)
        .limit(10**9)
        .to_pandas()
    )
    new_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        new_map[row["parent_id"]] = row.to_dict()
    # Atomic reference swap
    _parent_map = new_map
    try:
        _parent_map_mtime = _parent_map_path.stat().st_mtime
    except FileNotFoundError:
        _parent_map_mtime = 0.0


def _maybe_refresh_parent_map():
    """Reload parent_map if directory mtime newer AND a READY marker exists.
    READY marker file pattern: lance_db/READY.{generation_id}"""
    global _parent_map_mtime
    try:
        cur_mtime = _parent_map_path.stat().st_mtime
    except FileNotFoundError:
        return
    if cur_mtime <= _parent_map_mtime:
        return
    # Check for any READY.* marker (writer wrote it after final flush)
    ready_markers = list(TEXTBOOK_DB_PATH.glob("READY.*"))
    if not ready_markers:
        return  # Reindex still in progress, don't reload
    _load_parent_map()


def _truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, bool, int]:
    """Truncate text at sentence boundary within max_tokens. Returns (text, was_truncated, tokens_returned)."""
    tokenizer = get_tokenizer()
    full_tokens = tok_len(text)
    if full_tokens <= max_tokens:
        return (text, False, full_tokens)
    # Token-based truncation: encode + decode at limit
    ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_ids = ids[:max_tokens]
    truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    # Snap to last sentence boundary if reasonable
    for sep in ["。", "！", "？", ".", "!", "?", "\n\n", "\n"]:
        idx = truncated.rfind(sep)
        if idx > len(truncated) * 0.7:  # only snap if we don't lose too much
            truncated = truncated[: idx + len(sep)]
            break
    return (truncated, True, max_tokens)


def get_ollama():
    return ollama.Client(host=OLLAMA_HOST)


def embed_query(text: str) -> list[float]:
    """Embed a search query with instruction prefix."""
    client = get_ollama()
    response = client.embed(model=EMBEDDING_MODEL, input=[QUERY_PREFIX + text])
    return response["embeddings"][0]


def _clean(text: str) -> str:
    """Remove surrogate characters that break JSON serialization on Windows."""
    if not isinstance(text, str):
        return str(text)
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def _safe_json_dumps(obj, **kwargs) -> str:
    """json.dumps that guarantees no surrogate characters in output."""
    raw = json.dumps(obj, ensure_ascii=False, **kwargs)
    return _clean(raw)


def format_results(results_df, include_excerpt: bool = False) -> list[dict]:
    """Format LanceDB DataFrame results into readable output."""
    formatted = []
    for _, row in results_df.iterrows():
        entry = {
            "file": _clean(str(row.get("file", ""))),
            "note": _clean(str(row.get("note", ""))),
            "section": _clean(str(row.get("section", ""))),
            "folder": _clean(str(row.get("folder", ""))),
            "tags": _clean(str(row.get("tags", ""))),
            "similarity": round(max(0, 1 - float(row.get("_distance", 0))), 4) if "_distance" in row.index else None,
            "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
        }
        if include_excerpt:
            text = str(row.get("text", ""))
            entry["excerpt"] = _clean(text[:500]) if text else ""
        formatted.append(entry)
    return formatted


# --- MCP Protocol ---

TOOLS = [
    {
        "name": "vault_search",
        "description": "Semantic search across the Obsidian vault. Returns the most relevant note chunks for a natural language query. Supports Chinese, English, and mixed queries. Use this to find notes related to a topic, symptom, condition, or concept.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (Chinese, English, or mixed). E.g. '脊髓損傷的膀胱管理' or 'stroke rehabilitation upper limb'"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10, max: 30)",
                    "default": 10
                },
                "folder": {
                    "type": "string",
                    "description": "Optional: filter to a specific top-level folder of your vault"
                },
                "include_excerpt": {
                    "type": "boolean",
                    "description": "Include text excerpt in results (default: false). Set true when you need content for summarization.",
                    "default": False
                },
                "boost_recent": {
                    "type": "boolean",
                    "description": "Apply recency weighting — recently modified notes score higher (default: true)",
                    "default": True
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "When false (default), 89Archived results are deprioritized. Set true to treat archived notes equally.",
                    "default": False
                },
                "exclude_cowork": {
                    "type": "boolean",
                    "description": "When true (default), filter out derivative notes matching the configured exclude patterns (e.g. drafts, shared co-notes, scratch material). Set false only if you explicitly need to include them (rare — e.g., for cleanup). No-op when no exclude patterns are configured.",
                    "default": True
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "vault_related",
        "description": "Find notes RELATED to a given note via the wiki-link graph (personalized PageRank / LinearRAG-style propagation). Unlike vault_search (semantic), this seeds from a note's own links and surfaces multi-hop connected notes — including ones that share no text but are conceptually linked. Use to discover related notes, expand context around a note, or suggest links. Give the note's title (filename without .md).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note title (filename without .md) to find related notes for. E.g. 'Achilles tendinopathy' or 'CVA'."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of related notes to return (default: 12, max: 30)",
                    "default": 12
                }
            },
            "required": ["note"]
        }
    },
    {
        "name": "vault_similar",
        "description": "Find notes similar to a given note. Provide a note filename (without .md) to find semantically related notes in the vault.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "note_name": {
                    "type": "string",
                    "description": "The note name (without .md extension) to find similar notes for"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of similar notes to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["note_name"]
        }
    },
    {
        "name": "vault_stats",
        "description": "Show statistics about the vault semantic index: total chunks, files indexed, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "textbook_search",
        "description": "Semantic search across an optional secondary corpus of long-form reference documents (e.g. converted textbooks / manuals), indexed separately from your notes. Parent-child two-tier index — children for the precise hit, parent_text for the surrounding subsection context. Set full_parent=true to retrieve untruncated parent text (capped at 4K tokens per parent and 12K total).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (Chinese, English, or mixed)"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default: 10, max: 50). 10 is usually sufficient — avoid 20+ to save tokens.",
                    "default": 10
                },
                "book": {
                    "type": "string",
                    "description": "Optional: filter by source/book folder name within the reference corpus"
                },
                "full_parent": {
                    "type": "boolean",
                    "description": "If true, return untruncated parent_text (per-parent cap 4K tokens, total budget 12K). Default false (parent_text truncated to 800 tokens).",
                    "default": False
                },
                "exclude_cowork": {
                    "type": "boolean",
                    "description": "When true (default), filter out derivative sources matching the configured exclude patterns. Set false only if you explicitly need to inspect them (rare — e.g., for cleanup or comparison). No-op when no exclude patterns are configured.",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }
]


def _textbook_search_v2(arguments: dict) -> str:
    """v2 parent-child semantic search with dedup, parent context, observability."""
    t0 = time.time()
    query = _clean(arguments["query"])
    n = min(arguments.get("n_results", 10), 50)
    book_filter = arguments.get("book")
    full_parent = bool(arguments.get("full_parent", False))
    exclude_cowork = bool(arguments.get("exclude_cowork", True))

    chunks_table = get_textbook_chunks_table()
    if chunks_table is None:
        return json.dumps({"error": "Textbook v2 index not built. Run textbook_indexer.py first."})

    # Refresh parent map if needed (READY marker check)
    _maybe_refresh_parent_map()
    if not _parent_map:
        _load_parent_map()

    # Embed query with Qwen3 instruction prefix; L2 normalize
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.embed(
        model=TEXTBOOK_EMBEDDING_MODEL,
        input=[TEXTBOOK_QUERY_PREFIX + query],
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    query_embedding = response["embeddings"][0]
    # L2 normalize
    import math
    s = math.sqrt(sum(v * v for v in query_embedding))
    if s > 0:
        query_embedding = [v / s for v in query_embedding]

    # Fetch n*10 to give dedup room (Gemini r5 #3)
    fetch_limit = n * 10
    search_builder = chunks_table.search(query_embedding).metric("cosine").limit(fetch_limit)
    if book_filter:
        search_builder = search_builder.where(f"book = '{_escape_sql(book_filter)}'")
    df = search_builder.to_pandas()

    # Score: cosine similarity = 1 - distance (since vectors L2-normalized)
    raw_results = []
    for _, row in df.iterrows():
        raw_results.append({
            "id": _clean(str(row.get("id", ""))),
            "parent_id": _clean(str(row.get("parent_id", ""))),
            "file": _clean(str(row.get("file", ""))),
            "book": _clean(str(row.get("book", ""))),
            "chapter": _clean(str(row.get("chapter", ""))),
            "section_path": _clean(str(row.get("section_path", ""))),
            "page_start": int(row.get("page_start", 0) or 0),
            "page_end": int(row.get("page_end", 0) or 0),
            "chunk_kind": _clean(str(row.get("chunk_kind", ""))),
            "chunk_idx": int(row.get("chunk_idx", 0) or 0),
            "n_siblings": int(row.get("n_siblings", 0) or 0),
            "token_count": int(row.get("token_count", 0) or 0),
            "heading_origin": _clean(str(row.get("heading_origin", ""))),
            "figure_ids": [_clean(str(s)) for s in (row.get("figure_ids") if row.get("figure_ids") is not None else [])],
            "has_figure_ref": bool(row.get("has_figure_ref", False)),
            "text": _clean(str(row.get("text", ""))),
            "similarity": round(max(0.0, 1.0 - float(row.get("_distance", 0))), 4),
        })

    # Cowork exclusion (default true) — filter out 共筆/口試PPT/cowork before dedup
    if exclude_cowork:
        raw_results = [
            r for r in raw_results
            if not _is_cowork_path(r.get("book", "")) and not _is_cowork_path(r.get("file", ""))
        ]

    # Dedup: max children per parent
    parent_count: dict[str, int] = {}
    deduped: list[dict] = []
    for r in raw_results:
        pid = r["parent_id"]
        if parent_count.get(pid, 0) >= MAX_CHILDREN_PER_PARENT:
            continue
        parent_count[pid] = parent_count.get(pid, 0) + 1
        deduped.append(r)

    # Soft book diversity: only enforce max_per_book if >50% from one book
    if not book_filter and deduped:
        from collections import Counter
        bk_counter = Counter(r["book"] for r in deduped)
        top_book, top_count = bk_counter.most_common(1)[0]
        if top_count / len(deduped) > SOFT_BOOK_DIVERSITY_THRESHOLD:
            book_count: dict[str, int] = {}
            soft_filtered = []
            for r in deduped:
                if book_count.get(r["book"], 0) >= DEFAULT_MAX_PER_BOOK:
                    continue
                book_count[r["book"]] = book_count.get(r["book"], 0) + 1
                soft_filtered.append(r)
            deduped = soft_filtered

    deduped = deduped[:n]

    # Group by unique parent for budget allocation (Gemini r6 #2)
    # Top-1 unique parent gets full (up to ABSOLUTE_PARENT_MAX_RETURN_TOKENS),
    # subsequent shrink progressively.
    seen_parent_ids: list[str] = []
    parent_score: dict[str, float] = {}
    for r in deduped:
        pid = r["parent_id"]
        if pid not in parent_score:
            parent_score[pid] = r["similarity"]
            seen_parent_ids.append(pid)
        # else keep first (highest) score — already sorted

    # Budget allocation per unique parent
    budget_per_parent: dict[str, int] = {}
    remaining_budget = MAX_TOTAL_PARENT_RETURN_TOKENS if full_parent else (n * MAX_PARENT_RETURN_TOKENS)
    for rank, pid in enumerate(seen_parent_ids):
        if not full_parent:
            budget_per_parent[pid] = MAX_PARENT_RETURN_TOKENS
            continue
        if rank == 0:
            quota = ABSOLUTE_PARENT_MAX_RETURN_TOKENS
        elif rank == 1:
            quota = 2000
        else:
            # Progressively shrink for top-3..N
            quota = max(400, remaining_budget // max(1, len(seen_parent_ids) - rank))
        quota = min(quota, ABSOLUTE_PARENT_MAX_RETURN_TOKENS, remaining_budget)
        budget_per_parent[pid] = quota
        remaining_budget = max(0, remaining_budget - quota)

    # Apply budget to enrich results with parent_text
    formatted_results = []
    for r in deduped:
        pid = r["parent_id"]
        parent_data = _parent_map.get(pid)
        parent_text_full = ""
        parent_token_count = 0
        if parent_data:
            parent_text_full = str(parent_data.get("text", ""))
            parent_token_count = int(parent_data.get("token_count", 0) or 0)
        budget = budget_per_parent.get(pid, MAX_PARENT_RETURN_TOKENS)
        truncated_text, was_truncated, returned_tokens = _truncate_text_to_tokens(parent_text_full, budget)

        # API surface: keep legacy field 'page' as page_start; add new fields
        formatted_results.append({
            "file": r["file"],
            "book": r["book"],
            "chapter": r["chapter"],
            "section": r["section_path"],  # legacy field name
            "section_path": r["section_path"],
            "page": r["page_start"],  # legacy
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text": r["text"][:300],  # legacy 300-char excerpt
            "similarity": r["similarity"],
            "parent_id": r["parent_id"],
            "parent_text": _clean(truncated_text),
            "parent_truncated": was_truncated,
            "parent_return_tokens": returned_tokens,
            "parent_original_tokens": parent_token_count,
            "chunk_kind": r["chunk_kind"],
            "chunk_idx": r["chunk_idx"],
            "n_siblings": r["n_siblings"],
            "token_count": r["token_count"],
            "heading_origin": r["heading_origin"],
            "figure_ids": r["figure_ids"],
            "has_figure_ref": r["has_figure_ref"],
        })

    # Compute observability metrics
    latency_ms = int((time.time() - t0) * 1000)
    top_score = formatted_results[0]["similarity"] if formatted_results else 0
    score_gap = (
        formatted_results[0]["similarity"] - formatted_results[1]["similarity"]
        if len(formatted_results) >= 2 else 0
    )
    similarities = [r["similarity"] for r in formatted_results]
    if len(similarities) >= 2:
        mean_s = sum(similarities) / len(similarities)
        score_std = (sum((s - mean_s) ** 2 for s in similarities) / len(similarities)) ** 0.5
    else:
        score_std = 0.0
    raw_unique = len({r["parent_id"] for r in raw_results}) / max(1, len(raw_results))
    deduped_unique = len({r["parent_id"] for r in formatted_results}) / max(1, len(formatted_results)) if formatted_results else 0

    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query": query,
        "n_requested": n,
        "book_filter": book_filter,
        "full_parent": full_parent,
        "embedding_model": TEXTBOOK_EMBEDDING_MODEL,
        "chunking_version": CHUNKING_VERSION,
        "tokenizer_version": "Qwen3-Embedding-0.6B",
        "query_template_version": QUERY_TEMPLATE_VERSION,
        "embedding_prompt_variant": "no_doc_prefix",
        "overlap_tokens": CHILD_OVERLAP_TOKENS,
        "child_target_tokens": CHILD_TARGET_TOKENS,
        "top_score": round(top_score, 4),
        "score_gap_top1_top2": round(score_gap, 4),
        "score_std_topN": round(score_std, 4),
        "raw_parent_hit_rate": round(raw_unique, 3),
        "post_dedup_parent_hit_rate": round(deduped_unique, 3),
        "top_chunk_ids": [r["parent_id"] for r in formatted_results[:5]],
        "top_books": list({r["book"] for r in formatted_results[:5]}),
        "latency_ms": latency_ms,
    }
    try:
        _search_logger.info(_safe_json_dumps(log_entry))
    except Exception:
        pass  # never let logging break search

    return _safe_json_dumps({
        "query": query,
        "results": formatted_results,
        "total": len(formatted_results),
    }, indent=2)


def handle_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as text."""
    try:
        table = get_table()

        if name == "vault_search":
            query = _clean(arguments["query"])
            n = min(arguments.get("n_results", 10), 30)
            folder = arguments.get("folder")
            include_excerpt = arguments.get("include_excerpt", False)
            exclude_cowork = bool(arguments.get("exclude_cowork", True))

            query_embedding = embed_query(query)

            search_builder = table.search(query_embedding).metric("cosine")

            # Fetch extra chunks to compensate for dedup losing notes
            fetch_n = min(n * 5, 100)
            search_builder = search_builder.limit(fetch_n)

            if folder:
                search_builder = search_builder.where(f"folder = '{_escape_sql(folder)}'")

            results_df = search_builder.to_pandas()
            formatted = format_results(results_df, include_excerpt=include_excerpt)

            # Cowork exclusion (default true) — filter out 共筆/口試PPT/cowork notes
            if exclude_cowork:
                formatted = [
                    r for r in formatted
                    if not _is_cowork_path(r.get("file", "") or r.get("note", ""))
                    and not _is_cowork_path(r.get("folder", ""))
                ]

            # Deduplicate by note name, keeping highest similarity
            seen_notes = {}
            deduped = []
            for r in formatted:
                note = r["note"]
                if note not in seen_notes:
                    seen_notes[note] = True
                    deduped.append(r)

            # Rerank with path + recency + relation weighting
            boost_recent = arguments.get("boost_recent", True)
            include_archived = arguments.get("include_archived", False)
            graph = get_graph()
            graph_relations = graph.get("extracted_relations", [])
            deduped = rerank(deduped, boost_recent=boost_recent, include_archived=include_archived,
                             query=query, graph_relations=graph_relations)

            # Trim to requested n after dedup + rerank
            deduped = deduped[:n]

            # Graph expansion: PPR (personalized PageRank) seeded on the top semantic
            # hits ranks multi-hop linked notes by graph proximity — replaces the old
            # naive 1-hop neighbor expansion (sparse-bridge recall@10 ~60% -> ~84%).
            result_notes = {r["note"] for r in deduped}
            adjacency = graph.get("adjacency", {})
            ppr_ranked = personalized_pagerank(
                adjacency,
                [r["note"] for r in deduped[:5]],
                top_k=15,
                exclude=result_notes,
            )
            graph_suggested = [n for n, _ in ppr_ranked]

            # Remove internal mtime field from output
            for r in deduped:
                r.pop("mtime", None)
                r.pop("raw_similarity", None)

            result = {
                "query": query,
                "results": deduped,
                "total_results": len(deduped),
            }
            if graph_suggested:
                result["graph_linked"] = graph_suggested[:10]

            # Attach extracted entities for result notes
            result_note_names = {r["note"] for r in deduped}
            entities_data = find_entities_for_notes(
                graph.get("extracted_entities", {}), result_note_names
            )
            if entities_data:
                result["entities"] = entities_data

            # Legacy: attach extracted relations if available
            relations = find_relations_for_notes(
                graph_relations, result_note_names
            )
            if relations:
                result["relations"] = relations

            return _safe_json_dumps(result, indent=2)

        elif name == "vault_related":
            note = arguments["note"]
            n = min(arguments.get("n_results", 12), 30)
            graph = get_graph()
            adjacency = graph.get("adjacency", {})
            if note not in adjacency:
                return _safe_json_dumps(
                    {"note": note, "error": "note not found in wiki-link graph", "related": []},
                    indent=2,
                )
            exclude_cowork = bool(arguments.get("exclude_cowork", True))
            ranked = related_notes(adjacency, note, top_k=n * 3)
            meta = graph.get("metadata", {})
            related = []
            for rn, sc in ranked:
                folder = meta.get(rn, {}).get("folder", "")
                file_path = meta.get(rn, {}).get("file", "") or rn
                if exclude_cowork and (_is_cowork_path(file_path) or _is_cowork_path(folder)):
                    continue
                related.append({"note": rn, "score": round(sc, 4), "folder": folder})
                if len(related) >= n:
                    break
            return _safe_json_dumps(
                {"note": note, "related": related, "total": len(related)}, indent=2
            )

        elif name == "vault_similar":
            note_name = arguments["note_name"]
            n = min(arguments.get("n_results", 10), 30)

            # Find the note's first chunk
            note_rows = table.search().where(f"note = '{_escape_sql(note_name)}'").limit(1).to_pandas()
            if note_rows.empty:
                return json.dumps({"error": f"Note '{note_name}' not found in index"})

            query_embedding = note_rows.iloc[0]["vector"]
            cand = max(n * 3, 40)
            results_df = table.search(query_embedding).metric("cosine").limit(cand).to_pandas()
            formatted = format_results(results_df, include_excerpt=True)

            # Semantic candidates (note -> dict), preserving order, source note dropped
            sem, sem_order = {}, []
            for r in formatted:
                note = r["note"]
                if note == note_name or note in sem:
                    continue
                sem[note] = r
                sem_order.append(note)

            # Graph candidates via personalized PageRank over the wiki-link graph
            # (LinearRAG-style) — surfaces multi-hop related notes semantic misses.
            graph = get_graph()
            meta = graph.get("metadata", {})
            adjacency = graph.get("adjacency", {})
            ppr_order = [nn for nn, _ in related_notes(adjacency, note_name, top_k=cand)] \
                if note_name in adjacency else []

            # Reciprocal-rank fusion of graph + semantic rankings
            RRF_K = 60
            fused = {}
            for rank, nn in enumerate(sem_order):
                fused[nn] = fused.get(nn, 0.0) + 1.0 / (RRF_K + rank)
            for rank, nn in enumerate(ppr_order):
                fused[nn] = fused.get(nn, 0.0) + 1.0 / (RRF_K + rank)

            deduped = []
            for nn in sorted(fused, key=lambda k: fused[k], reverse=True):
                d = sem.get(nn) or {
                    "note": nn, "file": meta.get(nn, {}).get("file", ""),
                    "folder": meta.get(nn, {}).get("folder", ""), "section": "", "excerpt": "",
                }
                d["similarity"] = round(fused[nn], 6)
                deduped.append(d)

            # Rerank applies path + recency on the fused score, then trim
            deduped = rerank(deduped)[:n]

            # Remove internal mtime field from output
            for r in deduped:
                r.pop("mtime", None)
                r.pop("raw_similarity", None)

            return _safe_json_dumps({
                "similar_to": note_name,
                "results": deduped,
                "total_results": len(deduped),
            }, indent=2)

        elif name == "vault_stats":
            count = table.count_rows()
            # Get unique files
            all_data = table.to_pandas()
            files = set(all_data["file"].unique())
            folders = all_data["folder"].value_counts().to_dict()

            stats = {
                "total_chunks": count,
                "total_files": len(files),
                "chunks_by_folder": dict(sorted(folders.items(), key=lambda x: -x[1])),
            }

            # Include textbook stats if available (v2 schema: chunks + parents)
            db = open_db()
            if CHUNKS_TABLE in db.table_names():
                ch_table = db.open_table(CHUNKS_TABLE)
                ch_count = ch_table.count_rows()
                ch_books_df = ch_table.search().select(["book", "file"]).limit(10**9).to_pandas()
                ch_books = ch_books_df["book"].value_counts().to_dict()
                stats["textbook_chunks"] = ch_count
                stats["textbook_files"] = len(set(ch_books_df["file"].unique()))
                stats["textbook_books"] = dict(sorted(ch_books.items(), key=lambda x: -x[1]))
            if PARENTS_TABLE in db.table_names():
                p_table = db.open_table(PARENTS_TABLE)
                stats["textbook_parents"] = p_table.count_rows()

            return _safe_json_dumps(stats, indent=2)

        elif name == "textbook_search":
            return _textbook_search_v2(arguments)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        tb = traceback.format_exc()
        return json.dumps({"error": str(e), "traceback": tb})


def main():
    """MCP server main loop — reads JSON-RPC from stdin, writes to stdout."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=False)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "vault-search",
                        "version": "3.0.0",
                        "author": "P.W. Chen (drpwchen.com)",
                    },
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass  # No response needed

        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            arguments = msg["params"].get("arguments", {})
            result_text = handle_tool_call(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }
            sys.stdout.write(_safe_json_dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "ping":
            response = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        else:
            # Unknown method — return empty result
            if msg_id is not None:
                response = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
