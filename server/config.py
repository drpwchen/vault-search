"""
vault-search — local-first semantic search for Obsidian.
Author: P.W. Chen · https://drpwchen.com · https://github.com/drpwchen

Central configuration for vault-search.

Every path, model name, and tunable is resolved here from environment variables,
with sensible defaults. Set them in a `.env` file (auto-loaded if python-dotenv is
installed) or export them in your shell. The ONLY value you must set is VAULT_PATH.

See `.env.example` in the repo root for the full list.
"""

import hashlib
import json
import os
from pathlib import Path

# Optionally load a .env file from the repo root (no hard dependency).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def _path(env: str, default: str | None) -> Path | None:
    val = os.environ.get(env)
    if val:
        return Path(val).expanduser()
    return Path(default).expanduser() if default else None


def _csv(env: str, default: str) -> list[str]:
    raw = os.environ.get(env, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def template_version(name: str, prefix: str) -> str:
    """Build the query-template identifier written to the search logs.

    The name alone is a promise nothing enforces: rename the template without
    editing the prefix (or edit the prefix and forget the name) and every log
    line afterwards attributes results to a template that was never used.
    Appending a short hash of the prefix itself makes that impossible — the
    name stays readable, the hash is what actually identifies the text.
    """
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}"


def _json_file(env: str, default: str | None) -> dict:
    p = _path(env, default)
    if p and p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            return {}
    return {}


# --- Data directory (all derived state lives here) ---------------------------
DATA_DIR = _path("VAULT_SEARCH_DATA_DIR", "~/.vault-search")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Vault ------------------------------------------------------------------
# REQUIRED: absolute path to your Obsidian vault (the folder holding your .md notes).
VAULT_PATH = _path("VAULT_PATH", None)
if VAULT_PATH is None:
    raise RuntimeError(
        "VAULT_PATH is not set. Point it at your Obsidian vault, e.g.\n"
        "  export VAULT_PATH=/home/you/Documents/MyVault   (Linux/macOS)\n"
        '  setx VAULT_PATH "C:\\Users\\you\\Documents\\MyVault"  (Windows)\n'
        "or put it in a .env file in the repo root."
    )

# Top-level folders never indexed (attachments, templates, system folders, etc.)
SKIP_FOLDERS = set(_csv(
    "VAULT_SEARCH_SKIP_FOLDERS",
    ".obsidian,.trash,.git,node_modules",
))

# --- LanceDB / Ollama -------------------------------------------------------
DB_PATH = _path("VAULT_SEARCH_DB_PATH", str(DATA_DIR / "lance_db"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Embedding model for vault notes. bge-m3 is bilingual (CN/EN), 1024-dim, runs on CPU.
EMBEDDING_MODEL = os.environ.get("VAULT_SEARCH_EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.environ.get("VAULT_SEARCH_EMBEDDING_DIM", "1024"))

# Incremental-index hash cache.
HASH_CACHE_PATH = _path("VAULT_SEARCH_HASH_CACHE", str(DATA_DIR / "file_hashes.json"))

# Shared observability log.
SEARCH_LOG_PATH = _path("VAULT_SEARCH_LOG_PATH", str(DATA_DIR / "search_log.jsonl"))

# --- Vault v2 index (parent-child chunking) ----------------------------------
# v2 splits every note into small child chunks (precise vector hit) plus
# subsection-sized parents (the context actually returned). It is a separate pair
# of tables, so v1 and v2 can coexist and be compared before you commit to one.
#
# Retrieval reads v2 only when this flag FILE exists — create or delete it to
# switch paths without restarting the server:
#   touch ~/.vault-search/vault_v2.enabled
VAULT_V2_FLAG = _path("VAULT_SEARCH_V2_FLAG", str(DATA_DIR / "vault_v2.enabled"))

# v2 embedding model. Defaults to the textbook model because the parent-child
# architecture came from there, but it is its own knob: a vault-only install
# never has to configure a textbook corpus to use v2.
VAULT_V2_EMBEDDING_MODEL = os.environ.get(
    "VAULT_SEARCH_V2_MODEL",
    os.environ.get("VAULT_SEARCH_TEXTBOOK_MODEL", "qwen3-embedding:0.6b"),
)

VAULT_V2_HASH_CACHE = _path("VAULT_SEARCH_V2_HASH", str(DATA_DIR / "vault_v2_hashes.json"))
VAULT_V2_ERROR_LOG = _path("VAULT_SEARCH_V2_ERRORS", str(DATA_DIR / "vault_v2_index_errors.json"))

# Observability log for vault searches (the textbook side has its own).
VAULT_SEARCH_LOG_PATH = _path(
    "VAULT_SEARCH_VAULT_LOG_PATH", str(DATA_DIR / "vault_search_log.jsonl")
)

# Instruction prefix prepended to every vault QUERY before embedding (Qwen3
# instruct format). Documents are embedded WITHOUT it, so changing this changes
# only how queries are phrased — never the stored vectors.
VAULT_QUERY_PREFIX = os.environ.get(
    "VAULT_SEARCH_V2_QUERY_PREFIX",
    "Instruct: Given a query, retrieve the most relevant passages "
    "from the user's personal notes.\nQuery: ",
)
# Human-readable name for the query template above, recorded in the search log.
# See `query_template_version()` in vault_indexer_v2.py: the logged value is this
# name plus a short hash of the prefix, so the name can never claim one template
# while a different prefix is actually in use.
VAULT_QUERY_TEMPLATE_NAME = os.environ.get(
    "VAULT_SEARCH_V2_TEMPLATE_VERSION", "qwen3-vault-v1"
)

# --- API server -------------------------------------------------------------
API_HOST = os.environ.get("VAULT_SEARCH_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("VAULT_SEARCH_API_PORT", "3789"))

# Path to the `claude` CLI used for Hybrid/Free chat (Claude subscription).
# Leave empty to disable Claude-backed chat (Vault mode still works via Ollama).
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

# Local Ollama model used for zero-cost Vault-mode chat. Any chat model you've pulled.
OLLAMA_VAULT_MODEL = os.environ.get("VAULT_SEARCH_CHAT_MODEL", "gemma2:9b")

# Assistant persona used in chat system prompts. Customize for your domain.
ASSISTANT_PERSONA = os.environ.get(
    "VAULT_SEARCH_PERSONA",
    "You are a knowledgeable assistant helping the user reason over their personal notes.",
)
# Reply language instruction appended to chat prompts.
ASSISTANT_LANGUAGE = os.environ.get(
    "VAULT_SEARCH_LANGUAGE",
    "Answer in the same language as the user's question.",
)

# Folder (relative to the vault) where the plugin saves chat transcripts.
CHAT_HISTORY_FOLDER = os.environ.get("VAULT_SEARCH_CHAT_HISTORY_FOLDER", "VaultChatHistory")

# --- Reranking weights ------------------------------------------------------
# Folder -> score multiplier. Boost the folders you trust most; demote archives.
# Default is empty (every folder weighted equally). Example:
#   VAULT_SEARCH_PATH_WEIGHTS='{"Reference":1.2,"Archive":0.8}'
PATH_WEIGHTS = (
    json.loads(os.environ["VAULT_SEARCH_PATH_WEIGHTS"])
    if os.environ.get("VAULT_SEARCH_PATH_WEIGHTS")
    else {}
)

# Folder name treated as "archived" (deprioritized unless include_archived=True).
ARCHIVE_FOLDER = os.environ.get("VAULT_SEARCH_ARCHIVE_FOLDER", "")

# --- Cowork / derivative exclusion ------------------------------------------
# Substrings (case-insensitive) marking derivative notes to filter out of search
# by default (e.g. shared exam-prep notes you should not cite as primary sources).
# Empty by default. Example: VAULT_SEARCH_EXCLUDE_PATTERNS="cowork,draft,scratch"
COWORK_PATTERNS = _csv("VAULT_SEARCH_EXCLUDE_PATTERNS", "")

# --- Knowledge graph (optional) ---------------------------------------------
GRAPH_PATH = _path("VAULT_SEARCH_GRAPH_PATH", str(DATA_DIR / "knowledge_graph.json"))
EXTRACT_PROGRESS_PATH = _path("VAULT_SEARCH_EXTRACT_PROGRESS", str(DATA_DIR / "extract_progress.json"))
# Optional dir of extra markdown (e.g. a glossary) used to canonicalize NER entities.
ENTITY_CANON_DIR = _path("VAULT_SEARCH_ENTITY_CANON_DIR", None)
# Folders whose notes get entity extraction FIRST. Extraction is slow and often
# time-boxed, so this decides what gets covered when a run cannot finish
# everything. Empty (the default) = no folder is favoured.
# Example: VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS="Reference,Projects"
ENTITY_PRIORITY_FOLDERS = set(_csv("VAULT_SEARCH_ENTITY_PRIORITY_FOLDERS", ""))

# --- Textbook corpus (optional add-on) --------------------------------------
# A second corpus of long-form reference docs, indexed separately with its own model.
# Leave TEXTBOOK_PATH unset to disable textbook search entirely.
TEXTBOOK_PATH = _path("VAULT_SEARCH_TEXTBOOK_PATH", None)
TEXTBOOK_EMBEDDING_MODEL = os.environ.get("VAULT_SEARCH_TEXTBOOK_MODEL", "qwen3-embedding:0.6b")
TEXTBOOK_HASH_CACHE = _path("VAULT_SEARCH_TEXTBOOK_HASH", str(DATA_DIR / "textbook_hashes.json"))
TEXTBOOK_ERROR_LOG = _path("VAULT_SEARCH_TEXTBOOK_ERRORS", str(DATA_DIR / "textbook_index_errors.json"))
TEXTBOOK_ERROR_ARCHIVE = _path("VAULT_SEARCH_TEXTBOOK_ERRORS_ARCHIVE", str(DATA_DIR / "textbook_index_errors.archive.json"))
# Per-source ranking boost for the textbook corpus (see examples/source_boost.example.json).
SOURCE_BOOST_PATH = _path("VAULT_SEARCH_SOURCE_BOOST", None)
# --- GPU lease (optional) ----------------------------------------------------
# Path to a machine-wide lease script that serializes GPU jobs. Unset (the
# default) means indexing just runs — right on a machine that has the GPU to
# itself. See server/gpu_lease_client.py for the interface it must expose.
GPU_LEASE_PATH = _path("VAULT_SEARCH_GPU_LEASE", None)

# Names this project registers under, so `<script> status` shows which indexer
# holds the card. One per indexer: they are separate jobs and must not be able
# to queue behind themselves.
GPU_LEASE_NAME_TEXTBOOK = os.environ.get(
    "VAULT_SEARCH_GPU_LEASE_NAME_TEXTBOOK", "vault_search_textbook_index")
GPU_LEASE_NAME_VAULT = os.environ.get(
    "VAULT_SEARCH_GPU_LEASE_NAME_VAULT", "vault_search_vault_index")

# How long to wait for a turn before giving up (default 6 h: a full index run
# ahead of us legitimately takes hours).
GPU_LEASE_ACQUIRE_TIMEOUT = float(
    os.environ.get("VAULT_SEARCH_GPU_LEASE_TIMEOUT", "21600"))
# Minimum time to hold the lease before yielding, so a burst of queued jobs
# cannot make the indexer thrash release/re-acquire without making progress.
GPU_LEASE_MIN_HOLD = float(os.environ.get("VAULT_SEARCH_GPU_LEASE_MIN_HOLD", "180"))
# Optional: the lease script's state directory. Knowing it lets the yield check
# peek at the queue with a directory listing instead of spawning a subprocess
# once per file. Without it the check still works, just time-throttled.
GPU_LEASE_STATE_DIR = _path("VAULT_SEARCH_GPU_LEASE_STATE", None)
GPU_LEASE_QUEUE_DIR = (GPU_LEASE_STATE_DIR / "queue") if GPU_LEASE_STATE_DIR else None
