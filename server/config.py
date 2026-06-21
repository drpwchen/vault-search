"""
Central configuration for vault-search.

Every path, model name, and tunable is resolved here from environment variables,
with sensible defaults. Set them in a `.env` file (auto-loaded if python-dotenv is
installed) or export them in your shell. The ONLY value you must set is VAULT_PATH.

See `.env.example` in the repo root for the full list.
"""

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
#   VAULT_SEARCH_PATH_WEIGHTS='{"52Medicine":1.2,"Archive":0.8}'
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
# Optional helper script that pauses a competing GPU job during heavy indexing.
GPU_LEASE_PATH = _path("VAULT_SEARCH_GPU_LEASE", None)
