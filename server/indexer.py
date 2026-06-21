"""
Vault Semantic Indexer
Reads all .md files from the Obsidian vault, chunks by headings,
embeds via Ollama (bge-m3), stores in LanceDB.

Usage:
    python indexer.py                  # Full re-index
    python indexer.py --incremental    # Only new/modified files
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa

# --- Config (resolved from environment / .env via config.py) ---
from config import (
    VAULT_PATH, DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIM,
    OLLAMA_HOST, HASH_CACHE_PATH, SKIP_FOLDERS,
)

TABLE_NAME = "vault"

# BGE-M3: no instruction prefix needed (handled internally by the model)
QUERY_PREFIX = ""
DOC_PREFIX = ""

# LanceDB table schema
SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("file", pa.utf8()),
    pa.field("note", pa.utf8()),
    pa.field("section", pa.utf8()),
    pa.field("folder", pa.utf8()),
    pa.field("tags", pa.utf8()),
    pa.field("aliases", pa.utf8()),
    pa.field("mtime", pa.float64()),
])


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end].strip()
    body = content[end + 4:].strip()

    # Simple YAML parsing for tags and aliases
    meta = {}
    current_key = None
    current_list = []
    for line in fm_text.split("\n"):
        # Key: value or Key:
        key_match = re.match(r'^(\w+):\s*(.*)', line)
        if key_match and not line.startswith("  ") and not line.startswith("\t"):
            if current_key and current_list:
                meta[current_key] = current_list
            current_key = key_match.group(1)
            val = key_match.group(2).strip()
            current_list = []
            if val and not val.startswith("["):
                meta[current_key] = val
                current_key = None
            elif val.startswith("[") and val.endswith("]"):
                meta[current_key] = [v.strip().strip('"').strip("'")
                                     for v in val[1:-1].split(",") if v.strip()]
                current_key = None
        elif current_key and line.strip().startswith("- "):
            current_list.append(line.strip()[2:].strip().strip('"').strip("'"))
    if current_key and current_list:
        meta[current_key] = current_list
    return meta, body


def chunk_by_headings(body: str, note_name: str) -> list[dict]:
    """Split note body into chunks by headings. Returns list of {text, section}."""
    lines = body.split("\n")
    chunks = []
    current_section = note_name  # Default section = note title
    current_lines = []

    for line in lines:
        heading_match = re.match(r'^(#{1,3})\s+(.+)', line)
        if heading_match:
            # Save previous chunk
            text = "\n".join(current_lines).strip()
            if text and len(text) > 20:  # Skip tiny fragments
                chunks.append({"text": text, "section": current_section})
            current_section = heading_match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last chunk
    text = "\n".join(current_lines).strip()
    if text and len(text) > 20:
        chunks.append({"text": text, "section": current_section})

    # If no headings found, return whole body as one chunk
    if not chunks:
        text = body.strip()
        if text and len(text) > 20:
            chunks.append({"text": text, "section": note_name})

    return chunks


def get_folder(rel_path: str) -> str:
    """Extract top-level folder from relative path."""
    parts = Path(rel_path).parts
    return parts[0] if len(parts) > 1 else ""


def file_hash(content: str) -> str:
    """SHA256 hash of file content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def embed_texts(texts: list[str], client: ollama.Client) -> list[list[float]]:
    """Embed a batch of texts via Ollama. Falls back to one-by-one on error."""
    import math
    try:
        response = client.embed(model=EMBEDDING_MODEL, input=texts)
        embeddings = response["embeddings"]
        # Check for NaN values
        if any(any(math.isnan(v) for v in emb) for emb in embeddings):
            raise ValueError("NaN in embeddings, retrying one-by-one")
        return embeddings
    except Exception as batch_err:
        print(f"  Batch embed failed ({batch_err}), retrying one-by-one...")
        # Fallback: embed one-by-one, skip failures
        results = []
        for text in texts:
            try:
                # Truncate aggressively for retry
                resp = client.embed(model=EMBEDDING_MODEL, input=[text[:4000]])
                emb = resp["embeddings"][0]
                if any(math.isnan(v) for v in emb):
                    print(f"    ⚠ NaN embedding, skipping (text: {text[:80]!r}...)")
                    results.append(None)
                else:
                    results.append(emb)
            except Exception as e:
                print(f"    [!] Embed failed: {e} (text: {text[:80]!r}...)")
                results.append(None)
        return results


def load_hash_cache() -> dict:
    """Load file hash cache for incremental indexing."""
    if HASH_CACHE_PATH.exists():
        return json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_hash_cache(cache: dict):
    """Save file hash cache."""
    HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def collect_md_files(vault_path: Path) -> list[Path]:
    """Collect all .md files, skipping excluded folders."""
    files = []
    for f in vault_path.rglob("*.md"):
        rel = f.relative_to(vault_path)
        # Skip if in excluded folder
        if any(part in SKIP_FOLDERS for part in rel.parts):
            continue
        # Skip hidden files/folders
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(f)
    return files


def open_db():
    """Connect to LanceDB."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def index_vault(incremental: bool = False):
    """Main indexing function."""
    db = open_db()

    # Full re-index: drop existing table first so we rebuild clean.
    # (Without this, _flush_batch just table.add()s onto the old rows → duplicates.)
    if not incremental:
        try:
            db.drop_table(TABLE_NAME)
            print(f"Full re-index: dropped existing '{TABLE_NAME}' table")
        except Exception:
            pass  # table doesn't exist yet — nothing to drop

    # Init Ollama client
    ollama_client = ollama.Client(host=OLLAMA_HOST)

    # Load hash cache for incremental mode
    hash_cache = load_hash_cache() if incremental else {}

    # Collect files
    md_files = collect_md_files(VAULT_PATH)
    print(f"Found {len(md_files)} .md files")

    # Process files
    skipped = 0
    errors = 0
    batch_size = 50  # Embed in batches

    pending_rows = []
    pending_texts = []
    new_hash_cache = {}
    files_to_delete = []

    for i, fpath in enumerate(md_files):
        try:
            content = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            errors += 1
            continue

        rel_path = str(fpath.relative_to(VAULT_PATH)).replace("\\", "/")
        note_name = fpath.stem
        content_hash = file_hash(content)
        new_hash_cache[rel_path] = content_hash

        # Skip unchanged files in incremental mode
        if incremental and hash_cache.get(rel_path) == content_hash:
            skipped += 1
            continue

        # Parse and chunk
        meta, body = parse_frontmatter(content)
        if not body.strip():
            continue

        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        aliases = meta.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        folder = get_folder(rel_path)
        chunks = chunk_by_headings(body, note_name)

        # Track files to delete old chunks (incremental mode)
        if incremental:
            files_to_delete.append(rel_path)

        for ci, chunk in enumerate(chunks):
            chunk_text = chunk["text"][:4000]
            pending_texts.append(chunk_text)
            pending_rows.append({
                "id": f"{rel_path}::{ci}",
                "text": chunk_text,
                "file": rel_path,
                "note": note_name,
                "section": chunk["section"],
                "folder": folder,
                "tags": ", ".join(tags) if tags else "",
                "aliases": ", ".join(aliases) if aliases else "",
                "mtime": fpath.stat().st_mtime,
            })

        # Embed and flush in batches
        if len(pending_texts) >= batch_size:
            embeddings = embed_texts(pending_texts, ollama_client)
            # Filter out failed embeddings (None)
            valid_rows = []
            for row, emb in zip(pending_rows, embeddings):
                if emb is not None:
                    row["vector"] = emb
                    valid_rows.append(row)
                else:
                    errors += 1
            pending_rows = valid_rows

            # Write batch
            if pending_rows:
                _flush_batch(db, pending_rows, files_to_delete, incremental)
            files_to_delete = []
            pending_rows, pending_texts = [], []

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(md_files)} files...")

    # Flush remaining
    if pending_texts:
        embeddings = embed_texts(pending_texts, ollama_client)
        valid_rows = []
        for row, emb in zip(pending_rows, embeddings):
            if emb is not None:
                row["vector"] = emb
                valid_rows.append(row)
            else:
                errors += 1
        pending_rows = valid_rows

    if pending_rows:
        _flush_batch(db, pending_rows, files_to_delete, incremental)

    # Save hash cache
    save_hash_cache(new_hash_cache)

    # Reconcile: drop chunks whose source file no longer exists on disk
    # (incremental only visits existing files, so deleted/renamed notes would
    # otherwise leave orphan rows = "ghost" search results)
    if incremental:
        try:
            table = db.open_table(TABLE_NAME)
            current = {str(p.relative_to(VAULT_PATH)).replace("\\", "/") for p in md_files}
            # Scan only the `file` column (avoid loading 1024-dim vectors)
            indexed = set(table.to_lance().to_table(columns=["file"]).column("file").to_pylist())
            gone = indexed - current
            for rel_path in gone:
                table.delete(f"file = '{_escape_sql(rel_path)}'")
            if gone:
                print(f"  Pruned {len(gone)} deleted file(s) from index")
        except Exception as e:
            print(f"  Reconcile skipped: {e}")

    total_chunks = _count_table(db)
    print(f"\nDone!")
    print(f"  Files processed: {len(md_files) - skipped}")
    print(f"  Files skipped (unchanged): {skipped}")
    print(f"  Total chunks in DB: {total_chunks}")
    print(f"  Errors: {errors}")


def _flush_batch(db, rows: list[dict], files_to_delete: list[str], incremental: bool):
    """Write a batch of rows to LanceDB."""
    try:
        table = db.open_table(TABLE_NAME)
    except Exception:
        # Table doesn't exist yet — create it
        db.create_table(TABLE_NAME, data=rows, schema=SCHEMA)
        return

    # Delete old chunks for modified files
    if incremental and files_to_delete:
        for file_path in files_to_delete:
            try:
                table.delete(f"file = '{_escape_sql(file_path)}'")
            except Exception:
                pass

    table.add(rows)


def _escape_sql(s: str) -> str:
    """Escape single quotes for LanceDB SQL filter."""
    return s.replace("'", "''")


def _count_table(db) -> int:
    """Count rows in the vault table."""
    try:
        table = db.open_table(TABLE_NAME)
        return table.count_rows()
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Obsidian vault into LanceDB")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-index new/modified files")
    args = parser.parse_args()

    start = time.time()
    index_vault(incremental=args.incremental)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")
