"""
Vault Search HTTP API Server v3.0
Exposes LanceDB semantic search + Claude chat as a REST API for Obsidian plugin.

Usage: python api_server.py
Default: http://127.0.0.1:3789
"""

import os

# numpy ships its own OpenBLAS, which pre-commits one thread scratch buffer per CPU
# core the moment numpy is imported: 373 MB of private bytes uncapped versus 19 MB
# capped, measured on a 12-thread machine with numpy 2.5.1. This process runs no
# linear algebra of its own — embeddings come from Ollama, vector math from LanceDB
# — so cap it before anything below pulls numpy in. Export your own value to override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import asyncio
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import lancedb
import ollama
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from scoring import rerank
from ppr import personalized_pagerank, related_notes
from indexer import (
    parse_frontmatter, chunk_by_headings, file_hash, get_folder,
    embed_texts, load_hash_cache, save_hash_cache, collect_md_files,
    TABLE_NAME, EMBEDDING_MODEL, OLLAMA_HOST, QUERY_PREFIX,
    open_db, _escape_sql,
)
from graph_builder import load_graph, get_neighbors
from mcp_server import find_relations_for_notes, handle_tool_call, TOOLS as MCP_TOOLS
from textbook_indexer import (
    TEXTBOOK_EMBEDDING_MODEL, TEXTBOOK_QUERY_PREFIX,
    QUERY_TEMPLATE_VERSION, CHUNKING_VERSION,
    CHUNKS_TABLE, PARENTS_TABLE, OLLAMA_NUM_CTX,
    DB_PATH as TEXTBOOK_DB_PATH,
    CHILD_TARGET_TOKENS, CHILD_OVERLAP_TOKENS,
    get_tokenizer, tok_len,
)
import logging
import logging.handlers
import math

# Lazy-loaded graph
_graph_cache = None


def _get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = load_graph()
    return _graph_cache

# --- Config (resolved from environment / .env via config.py) ---
from config import (
    VAULT_PATH, CLAUDE_CMD, OLLAMA_VAULT_MODEL,
    API_HOST, API_PORT, SEARCH_LOG_PATH, TEXTBOOK_PATH,
    SOURCE_BOOST_PATH, ASSISTANT_PERSONA, ASSISTANT_LANGUAGE,
)

PORT = API_PORT
EXAM_BOOST_PATH = SOURCE_BOOST_PATH  # optional per-source ranking boost (None = disabled)

# --- Source Boost (hot-reloadable) ---
_exam_boost_cache: dict[str, float] | None = None
_exam_boost_mtime: float = 0


def _load_exam_boost() -> dict[str, float]:
    """Load per-source boost config, auto-reload if file changed. Disabled when unset."""
    global _exam_boost_cache, _exam_boost_mtime
    if EXAM_BOOST_PATH is None:
        return {}
    try:
        mtime = EXAM_BOOST_PATH.stat().st_mtime
        if _exam_boost_cache is None or mtime != _exam_boost_mtime:
            raw = json.loads(EXAM_BOOST_PATH.read_text(encoding="utf-8"))
            _exam_boost_cache = {k: v for k, v in raw.items() if not k.startswith("_")}
            _exam_boost_mtime = mtime
    except Exception:
        if _exam_boost_cache is None:
            _exam_boost_cache = {}
    return _exam_boost_cache


# --- Recency boost (year extracted from book name) ---
_year_cache: dict[str, int] = {}


def _recency_boost(book_name: str) -> float:
    """Small boost for newer editions: (year - 2000) * 0.001, max +0.025."""
    if book_name not in _year_cache:
        m = re.search(r'_(\d{4})', book_name)
        _year_cache[book_name] = int(m.group(1)) if m else 0
    year = _year_cache[book_name]
    if year < 2000:
        return 0.0
    return min((year - 2000) * 0.001, 0.025)

# --- API Key ---
_api_key_cache = None


def _load_api_key() -> str | None:
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache
    # 1. Environment variable
    key = os.environ.get("VAULT_API_KEY")
    if key:
        _api_key_cache = key.strip()
        return _api_key_cache
    # 2. File
    key_file = Path.home() / ".vault-search" / "api_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            _api_key_cache = key
            return _api_key_cache
    return None


# --- App ---
app = FastAPI(title="Vault Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Check X-API-Key header on all /api/* endpoints except /api/health."""
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        expected = _load_api_key()
        if expected:
            provided = request.headers.get("X-API-Key", "")
            if provided != expected:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )
    return await call_next(request)


# Lazy-loaded singletons
_table = None
_textbook_table = None
_ollama = None


def get_table():
    global _table
    if _table is None:
        db = open_db()
        _table = db.open_table(TABLE_NAME)
    return _table


def get_textbook_table():
    """Legacy: returns chunks table (now textbook_chunks). Kept name for compat."""
    global _textbook_table
    if _textbook_table is None:
        db = open_db()
        if CHUNKS_TABLE in db.table_names():
            _textbook_table = db.open_table(CHUNKS_TABLE)
        else:
            return None
    return _textbook_table


# --- Textbook v2: parent_map cache ---
_parent_map: dict[str, dict] = {}
_parent_map_mtime: float = 0.0
_parent_map_path = TEXTBOOK_DB_PATH / "textbook_parents.lance"
PARENT_CACHE_COLUMNS = [
    "parent_id", "text", "file", "book", "chapter",
    "section_path", "section_path_raw", "page_start", "page_end",
    "token_count", "heading_origin", "figure_ids", "has_figure_ref",
    "chunking_version",
]

# Search-side parameters (mirror mcp_server.py)
MAX_PARENT_RETURN_TOKENS = 800
ABSOLUTE_PARENT_MAX_RETURN_TOKENS = 4000
MAX_TOTAL_PARENT_RETURN_TOKENS = 12000
MAX_CHILDREN_PER_PARENT = 2
SOFT_BOOK_DIVERSITY_THRESHOLD = 0.5
DEFAULT_MAX_PER_BOOK = 3


def _get_textbook_parents_table():
    db = open_db()
    if PARENTS_TABLE not in db.table_names():
        return None
    return db.open_table(PARENTS_TABLE)


def _load_parent_map():
    """Atomic-swap reload. Excludes vector column to avoid memory spike."""
    global _parent_map, _parent_map_mtime
    table = _get_textbook_parents_table()
    if table is None:
        return
    df = (
        table.search().select(PARENT_CACHE_COLUMNS).limit(10**9).to_pandas()
    )
    new_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        new_map[row["parent_id"]] = row.to_dict()
    _parent_map = new_map
    try:
        _parent_map_mtime = _parent_map_path.stat().st_mtime
    except FileNotFoundError:
        _parent_map_mtime = 0.0


def _maybe_refresh_parent_map():
    global _parent_map_mtime
    try:
        cur_mtime = _parent_map_path.stat().st_mtime
    except FileNotFoundError:
        return
    if cur_mtime <= _parent_map_mtime:
        return
    if not list(TEXTBOOK_DB_PATH.glob("READY.*")):
        return
    _load_parent_map()


def _truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, bool, int]:
    tokenizer = get_tokenizer()
    full_tokens = tok_len(text)
    if full_tokens <= max_tokens:
        return (text, False, full_tokens)
    ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_ids = ids[:max_tokens]
    truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    for sep in ["。", "！", "？", ".", "!", "?", "\n\n", "\n"]:
        idx = truncated.rfind(sep)
        if idx > len(truncated) * 0.7:
            truncated = truncated[: idx + len(sep)]
            break
    return (truncated, True, max_tokens)


# Retrieval observability log (shared format with mcp_server.py)
_search_log_path = SEARCH_LOG_PATH
_search_log_path.parent.mkdir(parents=True, exist_ok=True)
_search_logger = logging.getLogger("textbook_search_api")
_search_logger.setLevel(logging.INFO)
if not _search_logger.handlers:
    _h = logging.handlers.RotatingFileHandler(
        str(_search_log_path), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _h.setFormatter(logging.Formatter("%(message)s"))
    _search_logger.addHandler(_h)
    _search_logger.propagate = False


@app.post("/api/admin/reload-parents")
def reload_parents():
    """Force-reload parent_map cache. Use after a known reindex."""
    _load_parent_map()
    return {"reloaded": True, "parent_count": len(_parent_map)}


def get_ollama():
    global _ollama
    if _ollama is None:
        _ollama = ollama.Client(host=OLLAMA_HOST)
    return _ollama


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


# --- Search & Similar endpoints ---

@app.get("/api/search")
def search(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(20, ge=1, le=30),
    folder: str | None = Query(None),
):
    query = _clean(query)
    client = get_ollama()
    response = client.embed(model=EMBEDDING_MODEL, input=[QUERY_PREFIX + query])
    query_embedding = response["embeddings"][0]

    search_builder = get_table().search(query_embedding).metric("cosine").limit(n_results * 3)
    if folder:
        search_builder = search_builder.where(f"folder = '{_escape_sql(folder)}'")
    results_df = search_builder.to_pandas()

    formatted = []
    seen = {}
    for _, row in results_df.iterrows():
        note = _clean(str(row.get("note", "")))
        if note in seen:
            continue
        seen[note] = True
        formatted.append({
            "file": _clean(str(row.get("file", ""))),
            "note": note,
            "section": _clean(str(row.get("section", ""))),
            "folder": _clean(str(row.get("folder", ""))),
            "tags": _clean(str(row.get("tags", ""))),
            "similarity": round(max(0, 1 - float(row.get("_distance", 0))), 4),
            "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
        })

    # Rerank with path + recency + relation weighting
    graph = _get_graph()
    graph_relations = graph.get("extracted_relations", [])
    formatted = rerank(formatted, query=query, graph_relations=graph_relations)
    formatted = formatted[:n_results]

    # Graph expansion: PPR (personalized PageRank) seeded on the top semantic hits
    # ranks multi-hop linked notes by graph proximity — replaces naive 1-hop
    # expansion (sparse-bridge recall@10 ~60% -> ~84%).
    result_notes = {r["note"] for r in formatted}
    adjacency = graph.get("adjacency", {})
    ppr_ranked = personalized_pagerank(
        adjacency, [r["note"] for r in formatted[:5]], top_k=15, exclude=result_notes
    )
    graph_linked = [n for n, _ in ppr_ranked]

    # Remove internal fields from output
    for r in formatted:
        r.pop("mtime", None)
        r.pop("raw_similarity", None)

    result = {"query": query, "results": formatted, "total": len(formatted)}
    if graph_linked:
        result["graph_linked"] = graph_linked[:10]

    # Attach extracted relations relevant to result notes
    relations = find_relations_for_notes(
        graph_relations, {r["note"] for r in formatted}
    )
    if relations:
        result["relations"] = relations
    return result


@app.get("/api/related")
def related(
    note: str = Query(..., description="Note title (filename without .md)"),
    n_results: int = Query(12, ge=1, le=30),
):
    """Find notes related to a given note via personalized PageRank over the
    wiki-link graph (LinearRAG-style). Seeds on the note's own links and surfaces
    multi-hop connected notes that semantic search may miss."""
    graph = _get_graph()
    adjacency = graph.get("adjacency", {})
    if note not in adjacency:
        return {"note": note, "error": "note not found in wiki-link graph", "related": []}
    meta = graph.get("metadata", {})
    ranked = related_notes(adjacency, note, top_k=n_results)
    out = [
        {"note": rn, "score": round(sc, 4), "folder": meta.get(rn, {}).get("folder", "")}
        for rn, sc in ranked
    ]
    return {"note": note, "related": out, "total": len(out)}


@app.get("/api/search-textbooks")
def search_textbooks(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(20, ge=1, le=50),
    book: str | None = Query(None, description="Filter by book folder name"),
    full_parent: bool = Query(False, description="Return untruncated parent_text (capped at 4K/parent, 12K total)"),
):
    """Semantic search v2 — parent-child schema, Qwen3 embedding, dedup, parent context."""
    t0 = time.time()
    chunks_table = get_textbook_table()  # now points to textbook_chunks
    if chunks_table is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Textbook v2 index not built. Run textbook_indexer.py first."},
        )

    _maybe_refresh_parent_map()
    if not _parent_map:
        _load_parent_map()

    query_clean = _clean(query)
    client = get_ollama()
    response = client.embed(
        model=TEXTBOOK_EMBEDDING_MODEL,
        input=[TEXTBOOK_QUERY_PREFIX + query_clean],
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    query_embedding = response["embeddings"][0]
    s = math.sqrt(sum(v * v for v in query_embedding))
    if s > 0:
        query_embedding = [v / s for v in query_embedding]

    # Fetch n*10 to give dedup room
    fetch_limit = n_results * 10
    search_builder = chunks_table.search(query_embedding).metric("cosine").limit(fetch_limit)
    if book:
        search_builder = search_builder.where(f"book = '{_escape_sql(book)}'")
    results_df = search_builder.to_pandas()

    raw_results = []
    for _, row in results_df.iterrows():
        b = str(row.get("book", ""))
        raw_sim = max(0, 1 - float(row.get("_distance", 0)))
        boost = _load_exam_boost().get(b, 0) + _recency_boost(b)
        boosted = raw_sim + boost
        raw_results.append({
            "row": row,
            "id": _clean(str(row.get("id", ""))),
            "parent_id": _clean(str(row.get("parent_id", ""))),
            "file": _clean(str(row.get("file", ""))),
            "book": _clean(b),
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
            "raw_similarity": round(raw_sim, 4),
            "similarity": round(boosted, 4),
        })
    raw_results.sort(key=lambda r: -r["similarity"])

    # Dedup: max children per parent
    parent_count: dict[str, int] = {}
    deduped = []
    for r in raw_results:
        pid = r["parent_id"]
        if parent_count.get(pid, 0) >= MAX_CHILDREN_PER_PARENT:
            continue
        parent_count[pid] = parent_count.get(pid, 0) + 1
        deduped.append(r)

    # Soft book diversity
    if not book and deduped:
        from collections import Counter
        bk = Counter(r["book"] for r in deduped)
        top_book, top_count = bk.most_common(1)[0]
        if top_count / len(deduped) > SOFT_BOOK_DIVERSITY_THRESHOLD:
            book_count: dict[str, int] = {}
            soft = []
            for r in deduped:
                if book_count.get(r["book"], 0) >= DEFAULT_MAX_PER_BOOK:
                    continue
                book_count[r["book"]] = book_count.get(r["book"], 0) + 1
                soft.append(r)
            deduped = soft

    deduped = deduped[:n_results]

    # Group unique parents for budget allocation
    seen_pids: list[str] = []
    for r in deduped:
        if r["parent_id"] not in seen_pids:
            seen_pids.append(r["parent_id"])

    budget_per_parent: dict[str, int] = {}
    remaining = MAX_TOTAL_PARENT_RETURN_TOKENS if full_parent else (n_results * MAX_PARENT_RETURN_TOKENS)
    for rank, pid in enumerate(seen_pids):
        if not full_parent:
            budget_per_parent[pid] = MAX_PARENT_RETURN_TOKENS
            continue
        if rank == 0:
            quota = ABSOLUTE_PARENT_MAX_RETURN_TOKENS
        elif rank == 1:
            quota = 2000
        else:
            quota = max(400, remaining // max(1, len(seen_pids) - rank))
        quota = min(quota, ABSOLUTE_PARENT_MAX_RETURN_TOKENS, remaining)
        budget_per_parent[pid] = quota
        remaining = max(0, remaining - quota)

    formatted = []
    for r in deduped:
        pid = r["parent_id"]
        parent_data = _parent_map.get(pid)
        parent_text_full = str(parent_data.get("text", "")) if parent_data else ""
        parent_token_count = int(parent_data.get("token_count", 0) or 0) if parent_data else 0
        budget = budget_per_parent.get(pid, MAX_PARENT_RETURN_TOKENS)
        truncated_text, was_truncated, returned_tokens = _truncate_text_to_tokens(parent_text_full, budget)

        formatted.append({
            "file": r["file"],
            "book": r["book"],
            "chapter": r["chapter"],
            "section": r["section_path"],
            "section_path": r["section_path"],
            "page": r["page_start"],
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text": r["text"][:500],  # legacy: 500-char excerpt
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

    # Observability log
    latency_ms = int((time.time() - t0) * 1000)
    top_score = formatted[0]["similarity"] if formatted else 0
    score_gap = formatted[0]["similarity"] - formatted[1]["similarity"] if len(formatted) >= 2 else 0
    sims = [r["similarity"] for r in formatted]
    if len(sims) >= 2:
        m = sum(sims) / len(sims)
        score_std = (sum((x - m) ** 2 for x in sims) / len(sims)) ** 0.5
    else:
        score_std = 0.0
    raw_unique = len({r["parent_id"] for r in raw_results}) / max(1, len(raw_results)) if raw_results else 0
    deduped_unique = len({r["parent_id"] for r in formatted}) / max(1, len(formatted)) if formatted else 0
    log_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "api_server",
        "query": query_clean,
        "n_requested": n_results,
        "book_filter": book,
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
        "top_parent_ids": [r["parent_id"] for r in formatted[:5]],
        "top_books": list({r["book"] for r in formatted[:5]}),
        "latency_ms": latency_ms,
    }
    try:
        _search_logger.info(json.dumps(log_entry, ensure_ascii=False))
    except Exception:
        pass

    return {"query": query_clean, "results": formatted, "total": len(formatted)}


@app.get("/api/similar")
def similar(
    note_name: str = Query(..., description="Note name without .md"),
    n_results: int = Query(10, ge=1, le=30),
):
    table = get_table()
    note_rows = table.search().where(f"note = '{_escape_sql(note_name)}'").limit(1).to_pandas()
    if note_rows.empty:
        return JSONResponse(status_code=404, content={"error": f"Note '{note_name}' not found"})

    query_embedding = note_rows.iloc[0]["vector"]
    cand = max(n_results * 3, 40)
    results_df = table.search(query_embedding).metric("cosine").limit(cand).to_pandas()

    # Semantic candidates (note -> metadata), preserving order
    sem, sem_order = {}, []
    for _, row in results_df.iterrows():
        note = _clean(str(row.get("note", "")))
        if note == note_name or note in sem:
            continue
        sem[note] = {
            "file": _clean(str(row.get("file", ""))),
            "note": note,
            "section": _clean(str(row.get("section", ""))),
            "folder": _clean(str(row.get("folder", ""))),
            "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
        }
        sem_order.append(note)

    # Graph candidates via personalized PageRank over the wiki-link graph (LinearRAG-style).
    graph = _get_graph()
    meta = graph.get("metadata", {})
    adjacency = graph.get("adjacency", {})
    ppr_order = [n for n, _ in related_notes(adjacency, note_name, top_k=cand)] \
        if note_name in adjacency else []

    # Reciprocal-rank fusion of the two rankings (graph + semantic).
    RRF_K = 60
    fused = {}
    for rank, n in enumerate(sem_order):
        fused[n] = fused.get(n, 0.0) + 1.0 / (RRF_K + rank)
    for rank, n in enumerate(ppr_order):
        fused[n] = fused.get(n, 0.0) + 1.0 / (RRF_K + rank)

    formatted = []
    for n in sorted(fused, key=lambda k: fused[k], reverse=True):
        d = sem.get(n) or {
            "file": meta.get(n, {}).get("file", ""), "note": n, "section": "",
            "folder": meta.get(n, {}).get("folder", ""), "mtime": None,
        }
        d["similarity"] = round(fused[n], 6)   # RRF score as base for rerank
        formatted.append(d)

    # Rerank applies path + recency on top of the fused score, then trim
    formatted = rerank(formatted)[:n_results]
    for r in formatted:
        r.pop("mtime", None)
        r.pop("raw_similarity", None)

    return {"similar_to": note_name, "results": formatted, "total": len(formatted)}


class ReindexRequest(BaseModel):
    file: str  # Relative path within vault, e.g. "52Medicine/某筆記.md"


@app.post("/api/reindex")
def reindex_note(req: ReindexRequest):
    """Re-index a single note in LanceDB."""
    import logging
    logger = logging.getLogger("vault-search")

    rel_path = req.file.replace("\\", "/")
    full_path = VAULT_PATH / rel_path

    if not full_path.exists():
        return JSONResponse(status_code=404, content={"error": f"File not found: {rel_path}"})

    try:
        content = full_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    note_name = full_path.stem
    meta, body = parse_frontmatter(content)
    if not body.strip():
        return JSONResponse(status_code=400, content={"error": "Note has no content"})

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    aliases = meta.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    folder = get_folder(rel_path)
    chunks = chunk_by_headings(body, note_name)

    table = get_table()

    # Delete old chunks for this file
    try:
        table.delete(f"file = '{_escape_sql(rel_path)}'")
    except Exception:
        pass

    # Build new chunks
    chunk_texts = []
    chunk_rows = []
    for ci, chunk in enumerate(chunks):
        chunk_text = chunk["text"][:8000]
        chunk_texts.append(chunk_text)
        chunk_rows.append({
            "id": f"{rel_path}::{ci}",
            "text": chunk_text,
            "file": rel_path,
            "note": note_name,
            "section": chunk["section"],
            "folder": folder,
            "tags": ", ".join(tags) if tags else "",
            "aliases": ", ".join(aliases) if aliases else "",
            "mtime": full_path.stat().st_mtime,
        })

    # Embed and store
    client = get_ollama()
    embeddings = embed_texts(chunk_texts, client)
    for row, emb in zip(chunk_rows, embeddings):
        row["vector"] = emb
    table.add(chunk_rows)

    # Update hash cache
    try:
        hash_cache = load_hash_cache()
        hash_cache[rel_path] = file_hash(content)
        save_hash_cache(hash_cache)
    except Exception:
        pass

    logger.info(f"[reindex] {note_name}: {len(chunks)} chunks")
    return {"status": "ok", "note": note_name, "chunks": len(chunks)}


# --- MCP passthrough ---------------------------------------------------------
# Registering mcp_server.py directly means every MCP client session starts its own
# copy, and each one loads lancedb plus — on first textbook use — the tokenizer and
# parent maps: 105 MB at startup, 367 MB after one vault_search and 2.2 GB after one
# textbook_search, measured here. Several concurrent sessions cost gigabytes for one
# index.
#
# mcp_thin.py forwards tool calls to these two endpoints instead, so this resident
# process is the only one holding the index and each session costs ~15 MB. The call
# below dispatches to the very same handle_tool_call() the standalone MCP server
# uses, so the two registration styles behave identically by construction.

@app.get("/api/mcp/tools")
def mcp_tools():
    return {"tools": MCP_TOOLS}


class McpCallBody(BaseModel):
    name: str
    arguments: dict = {}


@app.post("/api/mcp/call")
def mcp_call(body: McpCallBody):
    # handle_tool_call never raises — it returns error JSON as text
    return {"text": handle_tool_call(body.name, body.arguments)}


@app.get("/api/health")
def health():
    try:
        table = get_table()
        count = table.count_rows()
        return {"status": "ok", "chunks": count, "version": "3.0", "author": "P.W. Chen / drpwchen.com"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# --- Chat with Claude ---

# Persona and language are configurable via VAULT_SEARCH_PERSONA / VAULT_SEARCH_LANGUAGE.
_COMMON_SUFFIX = (
    "\n\nImportant constraints:\n"
    "1. You have no tools — you cannot search, read files, or run code. Answer ONLY from the text provided in this prompt.\n"
    "2. The system has already searched the vault and placed relevant notes in the 'vault notes' section above. If notes are present, use them directly.\n"
    "3. If there is not enough note content, or you think other notes may be relevant, suggest them with [[Note Name]] syntax "
    "(e.g. 'Consider adding [[Project Plan]], [[Meeting Notes]]'). The user can add them with one click and re-ask.\n"
    "4. Do not say 'I need permission' or 'I cannot search' — just answer from the available content or suggest notes to add."
)

SYSTEM_PROMPTS = {
    "vault": (
        ASSISTANT_PERSONA + " "
        "Answer ONLY from the provided vault note content. "
        "If the notes contain nothing relevant, say clearly that nothing relevant was found in the vault. "
        + ASSISTANT_LANGUAGE
        + _COMMON_SUFFIX
    ),
    "free": (
        ASSISTANT_PERSONA + " " + ASSISTANT_LANGUAGE
    ),
    "hybrid": (
        ASSISTANT_PERSONA + " "
        "Prefer the provided vault notes and cite the source note name. "
        "If the notes are insufficient, supplement with your own knowledge but mark it '(supplemented)'. "
        + ASSISTANT_LANGUAGE
        + _COMMON_SUFFIX
    ),
}



class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]
    use_vault_context: bool = True
    n_context: int = 5
    selected_notes: list[str] = []  # User-selected note names for context
    mode: str = "hybrid"  # "vault" | "free" | "hybrid"
    active_note: dict | None = None  # {"file": "...", "content": "...", "type": "selection|section|full"}
    mentioned_notes: list[str] = []  # Note names from @ mentions
    request_id: str = ""  # Client-generated ID for cancellation


# Active Claude subprocesses, keyed by request_id
_active_processes: dict[str, subprocess.Popen] = {}


def _read_full_note(file_path: str) -> str:
    """Read the full .md file from VAULT_PATH for complete note content."""
    try:
        full_path = VAULT_PATH / file_path
        if not full_path.exists():
            # Try with .md extension
            if not file_path.endswith(".md"):
                full_path = VAULT_PATH / (file_path + ".md")
        if full_path.exists() and full_path.is_file():
            content = full_path.read_text(encoding="utf-8")
            return content
    except Exception:
        pass
    return ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: Chinese chars * 1.5, English words * 0.25."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    # Remove Chinese chars to count English words
    english_text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]', ' ', text)
    english_words = len(english_text.split())
    return int(chinese_chars * 1.5 + english_words * 0.25)


def _compress_history(history: list[dict], keep_last_n: int = 4) -> list[dict]:
    """Compress older history turns, keeping last N turns complete."""
    if len(history) <= keep_last_n * 2:
        return history
    # Each "turn" is a user+assistant pair; keep_last_n turns = keep_last_n*2 messages
    keep_count = keep_last_n * 2
    old_msgs = history[:-keep_count]
    recent_msgs = history[-keep_count:]
    compressed = []
    for msg in old_msgs:
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "[...已壓縮...]"
        compressed.append({"role": msg["role"], "content": content})
    return compressed + recent_msgs


_TEXTBOOK_MD_PATH = TEXTBOOK_PATH if TEXTBOOK_PATH else Path("/nonexistent-textbook-corpus")


def _read_textbook_context(file_rel: str, page_num: int, context_pages: int = 2) -> str:
    """Read expanded context from textbook md file around the matched page.

    Returns text from page_num ± context_pages (e.g. pages 3-7 for match on page 5).
    Falls back to empty string if file not found.
    """
    try:
        md_path = _TEXTBOOK_MD_PATH / file_rel
        if not md_path.exists():
            return ""
        content = md_path.read_text(encoding="utf-8", errors="replace")

        # Extract pages around the match
        if "<!-- page " not in content:
            # No page markers (e.g. epub) — fallback: find the heading section
            # containing the chunk and return it with neighbors
            sections = re.split(r'\n(?=#{1,3} )', content)
            if page_num < len(sections):
                start = max(0, page_num - 1)
                end = min(len(sections), page_num + 2)
                text = "\n\n".join(s.strip() for s in sections[start:end] if s.strip())
                return text[:6000] if text else ""
            return ""

        pages = re.split(r'(<!-- page \d+ -->)', content)
        # Build page_num -> text mapping
        page_map: dict[int, str] = {}
        current_page = 0
        for part in pages:
            m = re.match(r'<!-- page (\d+) -->', part)
            if m:
                current_page = int(m.group(1))
            elif current_page > 0 and part.strip():
                page_map[current_page] = part.strip()

        # Collect context_pages before and after
        target_pages = range(max(1, page_num - context_pages), page_num + context_pages + 1)
        parts = []
        for pg in target_pages:
            if pg in page_map:
                parts.append(f"[p.{pg}] {page_map[pg]}")

        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def _search_vault_context(query: str, n: int = 5) -> str:
    """Search vault AND textbooks, return formatted context for Claude."""
    try:
        client = get_ollama()
        response = client.embed(model=EMBEDDING_MODEL, input=[QUERY_PREFIX + query])
        query_embedding = response["embeddings"][0]

        # --- Vault search ---
        results_df = get_table().search(query_embedding).metric("cosine").limit(n * 3).to_pandas()

        formatted = []
        seen = {}
        for _, row in results_df.iterrows():
            note = str(row.get("note", ""))
            if note in seen:
                continue
            seen[note] = True
            formatted.append({
                "note": note,
                "section": str(row.get("section", "")),
                "folder": str(row.get("folder", "")),
                "similarity": round(max(0, 1 - float(row.get("_distance", 0))), 4),
                "mtime": float(row["mtime"]) if row.get("mtime") is not None else None,
                "doc": str(row.get("text", "")),
            })

        formatted = rerank(formatted)
        context_parts = []
        for r in formatted[:n]:
            if r["similarity"] < 0.5:
                continue
            doc = r["doc"]
            if len(doc) > 1500:
                doc = doc[:1500] + "..."
            context_parts.append(f"【{r['note']} › {r['section']}��(sim={r['similarity']:.2f})\n{doc}")

        vault_ctx = ""
        if context_parts:
            vault_ctx = "以下是 vault 中與問題相關的筆記內容���\n\n" + "\n\n---\n\n".join(context_parts)

        # --- Textbook search (with exam boost + book-level dedup) ---
        textbook_ctx = ""
        tb = get_textbook_table()
        if tb is not None:
            tb_n = max(5, n)  # Match vault result count
            tb_df = tb.search(query_embedding).metric("cosine").limit(tb_n * 5).to_pandas()

            # Apply exam textbook boost + recency boost
            exam_boost = _load_exam_boost()
            rows_boosted = []
            for _, row in tb_df.iterrows():
                b = str(row.get("book", ""))
                raw_sim = max(0, 1 - float(row.get("_distance", 0)))
                boosted = raw_sim + exam_boost.get(b, 0) + _recency_boost(b)
                rows_boosted.append((row, b, boosted))
            rows_boosted.sort(key=lambda x: -x[2])

            # Book-level dedup: max 2 chunks per book in chat context
            book_counts: dict[str, int] = {}
            tb_parts = []
            for row, b, sim in rows_boosted:
                if sim < 0.5:
                    continue
                if book_counts.get(b, 0) >= 2:
                    continue
                book_counts[b] = book_counts.get(b, 0) + 1
                section = str(row.get("section", ""))
                file_name = str(row.get("file", ""))
                page_num = int(row.get("page", 0))

                # Try to read expanded context from the source md file
                text = _read_textbook_context(file_name, page_num)
                if not text:
                    text = str(row.get("text", ""))
                if len(text) > 3000:
                    text = text[:3000] + "..."

                tb_parts.append(f"【📖 {b} › {file_name} › {section}】(sim={sim:.2f})\n{text}")
                if len(tb_parts) >= tb_n:
                    break
            if tb_parts:
                textbook_ctx = "以下是教科書中與問題相關的內容：\n\n" + "\n\n---\n\n".join(tb_parts)

        # Combine
        parts = [p for p in [vault_ctx, textbook_ctx] if p]
        return "\n\n" + "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def _get_notes_context(note_names: list[str]) -> str:
    """Build context from user-selected note names."""
    table = get_table()
    context_parts = []
    for name in note_names:
        rows_df = table.search().where(f"note = '{_escape_sql(name)}'").limit(100).to_pandas()
        if rows_df.empty:
            continue
        # Combine all chunks of this note
        chunks = []
        for _, row in rows_df.iterrows():
            section = str(row.get("section", ""))
            text = str(row.get("text", ""))
            chunks.append(f"## {section}\n{text}" if section else text)
        combined = "\n\n".join(chunks)
        if len(combined) > 3000:
            combined = combined[:3000] + "..."
        context_parts.append(f"【{name}】\n{combined}")

    if not context_parts:
        return ""
    return "以下是使用者指定的 vault 筆記內容：\n\n" + "\n\n---\n\n".join(context_parts)


def _run_ollama_vault(system_prompt: str, user_prompt: str) -> dict:
    """Run Ollama gemma4 for vault-mode chat. Returns {reply, usage}."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = ollama.chat(
            model=OLLAMA_VAULT_MODEL,
            messages=messages,
            options={"temperature": 0.3, "num_predict": 4000, "num_ctx": 16384},
        )
        reply = resp["message"]["content"]
        # Strip <think>...</think> blocks (Gemma4 thinking output)
        reply = re.sub(r'<think>[\s\S]*?</think>', '', reply).strip()
        usage = {
            "input_tokens": resp.get("prompt_eval_count", 0),
            "output_tokens": resp.get("eval_count", 0),
            "cache_read": 0,
            "cache_creation": 0,
            "cost_usd": 0,
            "model": OLLAMA_VAULT_MODEL,
        }
        return {"reply": reply, "usage": usage}
    except Exception as e:
        return {"reply": "", "error": str(e)}


def _build_prompt(req: ChatRequest, vault_context: str, active_note_context: str = "") -> tuple[str, str]:
    """Build system prompt and user prompt separately for claude -p --system-prompt."""
    # System instruction based on mode
    mode = req.mode if req.mode in SYSTEM_PROMPTS else "hybrid"
    system_prompt = SYSTEM_PROMPTS[mode]

    # User prompt parts
    parts = []

    # Active note context
    if active_note_context:
        parts.append(active_note_context)

    # Vault context
    if vault_context:
        parts.append(vault_context)

    # Conversation history (with auto-compression)
    history = req.history
    total_text = system_prompt + vault_context + active_note_context + req.message
    for msg in history:
        total_text += msg.get("content", "")
    estimated_tokens = _estimate_tokens(total_text)

    if estimated_tokens > 80000:
        history = _compress_history(history, keep_last_n=4)
        # Re-estimate after compression
        compressed_text = system_prompt + vault_context + active_note_context + req.message
        for msg in history:
            compressed_text += msg.get("content", "")
        if _estimate_tokens(compressed_text) > 80000 and vault_context:
            # Truncate vault context: keep only first 3 notes
            context_notes = vault_context.split("\n\n---\n\n")
            if len(context_notes) > 3:
                vault_context = "\n\n---\n\n".join(context_notes[:3]) + "\n\n[...其餘 vault 內容已省略...]"
                parts = []
                if active_note_context:
                    parts.append(active_note_context)
                if vault_context:
                    parts.append(vault_context)

    for msg in history:
        role = "使用者" if msg["role"] == "user" else "助手"
        parts.append(f"{role}：{msg['content']}")

    # Current message
    parts.append(f"使用者：{req.message}")

    return system_prompt, "\n\n".join(parts)


# --- Async job store for polling-based chat ---
_chat_jobs: dict[str, dict] = {}


def _run_claude_async(cmd, user_prompt, request_id, vault_context, mode, context_notes):
    """Run Claude subprocess in background thread, store result in _chat_jobs."""
    import logging
    logger = logging.getLogger("vault-chat")

    _chat_jobs[request_id] = {"status": "processing"}

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).parent),
        )
        _active_processes[request_id] = proc

        try:
            stdout, stderr = proc.communicate(
                input=user_prompt.encode("utf-8"),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            _chat_jobs[request_id] = {"status": "error", "error": "Claude 回應超時（300秒）"}
            _active_processes.pop(request_id, None)
            return
        finally:
            _active_processes.pop(request_id, None)

        # Cancelled
        if proc.returncode and proc.returncode < 0:
            _chat_jobs[request_id] = {"status": "error", "error": "已取消"}
            return

        raw = stdout.decode("utf-8", errors="replace").strip()

        usage = {}
        try:
            data = json.loads(raw)
            reply = data.get("result") or ""
            raw_usage = data.get("usage", {})
            usage = {
                "input_tokens": raw_usage.get("input_tokens", 0),
                "output_tokens": raw_usage.get("output_tokens", 0),
                "cache_read": raw_usage.get("cache_read_input_tokens", 0),
                "cache_creation": raw_usage.get("cache_creation_input_tokens", 0),
                "cost_usd": data.get("total_cost_usd", 0),
            }
        except json.JSONDecodeError:
            reply = raw

        if proc.returncode != 0 and not reply:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            _chat_jobs[request_id] = {"status": "error", "error": error_msg or "Claude process failed"}
            return

        _chat_jobs[request_id] = {
            "status": "done",
            "reply": reply,
            "vault_context_used": bool(vault_context),
            "context_notes": context_notes,
            "mode": mode,
            "usage": usage,
        }
    except Exception as e:
        logger.error(f"[async-chat] error: {e}")
        _chat_jobs[request_id] = {"status": "error", "error": str(e)}


@app.get("/api/chat/poll")
async def poll_chat(request_id: str = Query(...)):
    """Poll for async chat job result."""
    job = _chat_jobs.get(request_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    if job["status"] == "processing":
        return {"status": "processing"}

    # Job is done or error — return result and clean up
    result = dict(job)
    _chat_jobs.pop(request_id, None)
    return result


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # --- Active note context ---
    active_note_context = ""
    if req.active_note:
        an = req.active_note
        an_file = an.get("file", "")
        an_type = an.get("type", "selection")
        if an_type == "full" and an_file:
            content = _read_full_note(an_file)
            if content:
                note_name = Path(an_file).stem
                active_note_context = f"以下是使用者目前開啟的筆記【{note_name}】完整內容：\n\n{content}"
        elif an.get("content"):
            note_name = Path(an_file).stem if an_file else "目前筆記"
            label = {"selection": "選取段落", "section": "章節", "full": "完整內容"}.get(an_type, "內容")
            active_note_context = f"以下是使用者目前開啟的筆記【{note_name}】的{label}：\n\n{an['content']}"

    # --- Merge mentioned_notes into selected_notes ---
    all_selected = list(req.selected_notes)
    for name in req.mentioned_notes:
        if name not in all_selected:
            all_selected.append(name)

    # --- Build vault context ---
    vault_context = ""
    skip_rag = False

    if req.mode == "free":
        skip_rag = True
    elif req.use_vault_context:
        if (len(req.message) <= 5
                and not all_selected
                and not req.active_note):
            skip_rag = True

    rag_query = req.message
    if req.active_note and req.active_note.get("content"):
        active_content = req.active_note["content"]
        rag_query = active_content[:300]

    if not skip_rag:
        if all_selected:
            vault_context = _get_notes_context(all_selected)
            if req.use_vault_context and not req.selected_notes:
                rag_context = _search_vault_context(rag_query, req.n_context)
                if rag_context:
                    vault_context = vault_context + "\n\n" + rag_context if vault_context else rag_context
        elif req.use_vault_context:
            vault_context = _search_vault_context(rag_query, req.n_context)

    system_prompt, user_prompt = _build_prompt(req, vault_context, active_note_context)

    import logging
    logger = logging.getLogger("vault-chat")
    logger.info(f"[chat] mode={req.mode} skip_rag={skip_rag} vault_ctx_len={len(vault_context)} active_note={'yes' if active_note_context else 'no'} mentioned={req.mentioned_notes} selected={req.selected_notes} msg_preview={req.message[:50]}")

    request_id = req.request_id
    context_notes = list(_extract_note_names(vault_context)) if vault_context else []

    # --- Per-mode tool permissions ---
    _ALWAYS_BLOCKED = (
        "Read,Write,Edit,Bash,Glob,Grep,Agent,NotebookEdit,"
        "Skill,ToolSearch,"
        "mcp__claude_ai_Gmail__*,"
        "mcp__claude_ai_Google_Calendar__*,"
        "mcp__semantic-scholar__*,"
        "mcp__vault-search__*"
    )
    _EXTERNAL_TOOLS = (
        "WebFetch,WebSearch,"
        "mcp__claude_ai_PubMed__search_articles,"
        "mcp__claude_ai_PubMed__get_article_metadata,"
        "mcp__claude_ai_PubMed__get_full_text_article,"
        "mcp__claude_ai_PubMed__find_related_articles,"
        "mcp__claude_ai_PubMed__lookup_article_by_citation,"
        "mcp__claude_ai_PubMed__convert_article_ids,"
        "mcp__claude_ai_PubMed__get_copyright_status,"
        "mcp__NotebookLM__*"
    )

    if req.mode == "vault":
        # --- Vault mode: use local Ollama gemma4 (zero Claude tokens) ---
        use_ollama = True
        try:
            ollama.show(OLLAMA_VAULT_MODEL)
        except Exception:
            use_ollama = False  # Model unavailable, fall back to Claude

        if use_ollama:
            import logging
            logger = logging.getLogger("vault-chat")
            logger.info(f"[chat] vault mode → Ollama {OLLAMA_VAULT_MODEL}")

            # For async polling
            if request.headers.get("x-async") == "1":
                def _run_ollama_async():
                    _chat_jobs[request_id] = {"status": "processing"}
                    result = _run_ollama_vault(system_prompt, user_prompt)
                    if result.get("error"):
                        _chat_jobs[request_id] = {"status": "error", "error": result["error"]}
                    else:
                        _chat_jobs[request_id] = {
                            "status": "done",
                            "reply": result["reply"],
                            "vault_context_used": bool(vault_context),
                            "context_notes": context_notes,
                            "mode": req.mode,
                            "usage": result["usage"],
                        }
                threading.Thread(target=_run_ollama_async, daemon=True).start()
                return {"status": "processing", "request_id": request_id}

            # Synchronous
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_ollama_vault, system_prompt, user_prompt
            )
            if result.get("error"):
                return JSONResponse(status_code=500, content={"error": result["error"]})
            return {
                "reply": result["reply"],
                "vault_context_used": bool(vault_context),
                "context_notes": context_notes,
                "mode": req.mode,
                "usage": result["usage"],
            }

        # Ollama unavailable — fall back to Claude
        disallowed = _ALWAYS_BLOCKED + ",WebFetch,WebSearch,mcp__claude_ai_PubMed__*,mcp__NotebookLM__*"
        cmd = [CLAUDE_CMD, "-p", "--output-format", "json",
               "--disallowedTools", disallowed,
               "--system-prompt", system_prompt]
    else:
        cmd = [CLAUDE_CMD, "-p", "--output-format", "json",
               "--disallowedTools", _ALWAYS_BLOCKED,
               "--allowedTools", _EXTERNAL_TOOLS,
               "--system-prompt", system_prompt]

    # --- Check if client wants async polling ---
    if request.headers.get("x-async") == "1":
        threading.Thread(
            target=_run_claude_async,
            args=(cmd, user_prompt, request_id, vault_context, req.mode, context_notes),
            daemon=True,
        ).start()
        return {"status": "processing", "request_id": request_id}

    # --- Synchronous (legacy) ---
    try:
        def _run_claude():
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent),
            )
            if request_id:
                _active_processes[request_id] = proc
            try:
                stdout, stderr = proc.communicate(
                    input=user_prompt.encode("utf-8"),
                    timeout=300,
                )
                return stdout, stderr, proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            finally:
                _active_processes.pop(request_id, None)

        stdout, stderr, returncode = await asyncio.get_event_loop().run_in_executor(None, _run_claude)

        if returncode < 0 or returncode == 1:
            if request_id not in _active_processes and returncode < 0:
                return JSONResponse(status_code=499, content={"error": "已取消"})

        raw = stdout.decode("utf-8", errors="replace").strip()

        usage = {}
        try:
            data = json.loads(raw)
            reply = data.get("result", raw)
            raw_usage = data.get("usage", {})
            usage = {
                "input_tokens": raw_usage.get("input_tokens", 0),
                "output_tokens": raw_usage.get("output_tokens", 0),
                "cache_read": raw_usage.get("cache_read_input_tokens", 0),
                "cache_creation": raw_usage.get("cache_creation_input_tokens", 0),
                "cost_usd": data.get("total_cost_usd", 0),
            }
        except json.JSONDecodeError:
            reply = raw

        if returncode != 0 and not reply:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            return JSONResponse(status_code=500, content={"error": error_msg or "Claude process failed"})

        return {
            "reply": reply,
            "vault_context_used": bool(vault_context),
            "context_notes": context_notes,
            "mode": req.mode,
            "usage": usage,
        }
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"error": "Claude 回應超時（300秒）"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat/cancel")
async def cancel_chat(request: Request):
    body = await request.json()
    request_id = body.get("request_id", "")
    if not request_id:
        return {"status": "no request_id"}
    proc = _active_processes.pop(request_id, None)
    if proc and proc.poll() is None:
        proc.kill()
        return {"status": "cancelled"}
    return {"status": "not_found"}


def _extract_note_names(context: str) -> set:
    """Extract note names from vault context string."""
    return set(re.findall(r"【(.+?) ›", context)) | set(re.findall(r"【(.+?)】", context))


if __name__ == "__main__":
    import sys
    _this_dir = str(Path(__file__).parent)
    # --reload: watch only api_server.py and exam_boost.json to avoid
    # spurious restarts when other .py files (mcp_server.py, indexer.py) change
    if "--no-reload" in sys.argv:
        uvicorn.run(app, host=API_HOST, port=PORT, log_level="info")
    else:
        uvicorn.run(
            "api_server:app",
            host=API_HOST,
            port=PORT,
            log_level="info",
            reload=True,
            reload_dirs=[_this_dir],
            reload_includes=["api_server.py", "scoring.py", "exam_boost.json"],
        )
