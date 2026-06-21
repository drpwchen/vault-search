"""
Vault Knowledge Graph Builder

Phase A: Parse [[wiki-links]] from all vault notes → adjacency graph (JSON)
Phase B: scispaCy NER entity extraction → entities per note (incremental)
Legacy Phase B: LLM extraction (frozen, replaced by NER)

Usage:
    python graph_builder.py              # Build/rebuild wiki-link graph
    python graph_builder.py --ner        # Run scispaCy NER on new/modified notes
    python graph_builder.py --ner --force # Re-NER all notes
    python graph_builder.py --normalize  # One-time: normalize legacy entities
    python graph_builder.py --bridges "Note Name"  # Query entity bridges for a note
    python graph_builder.py --stats      # Show graph statistics
    python graph_builder.py --extract N  # (Legacy) LLM extraction
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# --- Config (resolved from environment / .env via config.py) ---
from config import (
    VAULT_PATH, GRAPH_PATH, EXTRACT_PROGRESS_PATH, SKIP_FOLDERS,
    OLLAMA_HOST, OLLAMA_VAULT_MODEL, ENTITY_CANON_DIR,
)

# Regex for [[wiki-links]], handles [[Note]] and [[Note|alias]]
WIKI_LINK_RE = re.compile(r'\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]+?)?\]\]')


def collect_md_files(vault_path: Path) -> list[Path]:
    """Collect all .md files, skipping excluded folders."""
    files = []
    for f in vault_path.rglob("*.md"):
        rel = f.relative_to(vault_path)
        if any(part in SKIP_FOLDERS for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(f)
    return files


def build_link_graph() -> dict:
    """Phase A: Parse all [[wiki-links]] and build adjacency graph."""
    md_files = collect_md_files(VAULT_PATH)
    print(f"Scanning {len(md_files)} files for [[links]]...")

    # Map: filename (stem) → set of linked note names
    forward_links = defaultdict(set)   # note → {linked notes}
    backlinks = defaultdict(set)       # note → {notes that link to it}
    note_metadata = {}                 # note → {file, folder, tags}

    for fpath in md_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        note_name = fpath.stem
        rel_path = str(fpath.relative_to(VAULT_PATH)).replace("\\", "/")
        parts = Path(rel_path).parts
        folder = parts[0] if len(parts) > 1 else ""

        note_metadata[note_name] = {"file": rel_path, "folder": folder}

        # Extract all [[links]]
        links = set(WIKI_LINK_RE.findall(content))
        for link in links:
            link = link.strip()
            if link and link != note_name:
                forward_links[note_name].add(link)
                backlinks[link].add(note_name)

    # Build bidirectional adjacency
    adjacency = defaultdict(set)
    for note, targets in forward_links.items():
        for target in targets:
            adjacency[note].add(target)
            adjacency[target].add(note)

    # Preserve Phase B data from existing graph
    existing_graph = load_graph()
    existing_relations = existing_graph.get("extracted_relations", [])
    existing_bridges = existing_graph.get("entity_bridges", [])
    existing_entities = existing_graph.get("extracted_entities", {})

    # Convert sets to sorted lists for JSON
    graph = {
        "adjacency": {k: sorted(v) for k, v in adjacency.items()},
        "metadata": note_metadata,
        "stats": {
            "total_notes": len(md_files),
            "notes_with_links": len(forward_links),
            "total_edges": sum(len(v) for v in adjacency.values()) // 2,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "extracted_relations": existing_relations,
        "extracted_entities": existing_entities,
        "entity_bridges": existing_bridges,
    }

    return graph


def save_graph(graph: dict):
    """Save graph to JSON."""
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    size_kb = GRAPH_PATH.stat().st_size / 1024
    print(f"Graph saved: {GRAPH_PATH} ({size_kb:.0f} KB)")


def load_graph() -> dict:
    """Load graph from JSON."""
    if not GRAPH_PATH.exists():
        return {"adjacency": {}, "metadata": {}, "stats": {}, "extracted_relations": []}
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def get_neighbors(graph: dict, note_name: str, hops: int = 1) -> list[str]:
    """Get notes within N hops of a given note. Only returns actual indexed notes."""
    adjacency = graph.get("adjacency", {})
    metadata = graph.get("metadata", {})
    visited = {note_name}
    frontier = {note_name}

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    visited.discard(note_name)
    # Filter to only notes that exist in the vault (not images/attachments)
    return sorted(n for n in visited if n in metadata)


# --- Phase B: LLM Entity Extraction ---

def load_extract_progress() -> dict:
    """Track which notes have been processed by Phase B."""
    if EXTRACT_PROGRESS_PATH.exists():
        return json.loads(EXTRACT_PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"processed": [], "last_run": None}


def save_extract_progress(progress: dict):
    progress["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    EXTRACT_PROGRESS_PATH.write_text(
        json.dumps(progress, ensure_ascii=False),
        encoding="utf-8",
    )


EXTRACT_PROMPT = """Analyze this medical note and extract entities and relationships.

Note: {note_name}
Content:
{content}

Return a JSON object with:
{{
  "entities": [
    {{"name": "entity name", "type": "condition|treatment|anatomy|symptom|test|medication|device"}},
    ...
  ],
  "relationships": [
    {{"from": "entity1", "to": "entity2", "type": "causes|treats|diagnoses|complication_of|part_of|indicated_for|contraindicated_for"}},
    ...
  ]
}}

Rules:
- Only extract clinically meaningful relationships
- Use English entity names (keep Chinese in parentheses if helpful)
- Max 20 entities and 20 relationships per note
- Return ONLY the JSON, no explanation"""

EXTRACT_PROMPT_INFRA = """Analyze this system/infrastructure memory file and extract entities and relationships.

File: {note_name}
Content:
{content}

Return a JSON object with:
{{
  "entities": [
    {{"name": "entity name", "type": "service|tool|project|script|database|config|schedule|device"}},
    ...
  ],
  "relationships": [
    {{"from": "entity1", "to": "entity2", "type": "depends_on|uses|schedules|monitors|configures|stores_in|serves|triggers"}},
    ...
  ]
}}

Rules:
- Extract tools, services, scripts, databases, scheduled tasks, and their connections
- Use exact names (e.g., "session-indexer.py", "LanceDB", "vault-search MCP")
- Max 20 entities and 20 relationships per file
- Return ONLY the JSON, no explanation"""

# --- Optional supplementary markdown (e.g. a glossary) for entity canonicalization ---
MEMORY_DIR = ENTITY_CANON_DIR  # None unless VAULT_SEARCH_ENTITY_CANON_DIR is set


def _collect_memory_candidates(progress: dict) -> list[tuple]:
    """Collect supplementary files that need extraction (new or modified since last processed)."""
    processed_mtimes = progress.get("processed_mtimes", {})  # note → mtime when processed
    candidates = []

    if not MEMORY_DIR or not MEMORY_DIR.exists():
        return candidates

    for f in MEMORY_DIR.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        note_name = f"memory:{f.stem}"
        mtime = f.stat().st_mtime
        prev_mtime = processed_mtimes.get(note_name, 0)
        if mtime > prev_mtime:
            candidates.append((note_name, str(f), mtime, "memory"))

    return candidates


def extract_batch(n: int = 10, max_minutes: int = 0):
    """Phase B: Run LLM extraction on unprocessed notes.

    Args:
        n: Max number of notes to process (ignored if max_minutes > 0).
        max_minutes: Time limit in minutes. 0 = use n instead.
    """
    import ollama

    graph = load_graph()
    progress = load_extract_progress()
    processed = set(progress["processed"])

    # Collect vault candidates: new notes + modified notes needing re-extraction
    processed_mtimes = progress.get("processed_mtimes", {})
    candidates_med = []
    candidates_other = []
    reprocess = []
    now = time.time()
    for note, meta in graph.get("metadata", {}).items():
        file_path = meta["file"]
        if note in processed:
            # Skip stat for notes with recorded mtime that can't have changed
            prev_mtime = processed_mtimes.get(note, 0)
            if not prev_mtime and not processed_mtimes:
                continue  # Legacy batch with no mtimes at all — skip until next full run
            full_path = VAULT_PATH / file_path
            try:
                mtime = full_path.stat().st_mtime
            except (FileNotFoundError, OSError):
                continue
            if (prev_mtime and mtime > prev_mtime) or \
               (not prev_mtime and mtime > now - 7 * 86400):
                reprocess.append((note, file_path, mtime, "vault"))
            continue
        full_path = VAULT_PATH / file_path
        mtime = full_path.stat().st_mtime if full_path.exists() else 0
        entry = (note, file_path, mtime, "vault")
        if meta.get("folder") == "52Medicine":
            candidates_med.append(entry)
        else:
            candidates_other.append(entry)
    if reprocess:
        print(f"  Re-extracting {len(reprocess)} modified notes")

    # Sort by mtime descending (recently modified first)
    candidates_med.sort(key=lambda x: -x[2])
    candidates_other.sort(key=lambda x: -x[2])

    # Re-process modified first, then 52Medicine, then others
    vault_candidates = reprocess + candidates_med + candidates_other

    if not vault_candidates:
        print("All notes already processed!")
        return

    if max_minutes > 0:
        all_candidates = vault_candidates
        print(f"Extracting entities (time limit: {max_minutes} min, {len(vault_candidates)} notes)...")
    else:
        all_candidates = vault_candidates[:n]
        print(f"Extracting entities from {min(n, len(vault_candidates))} notes...")
    client = ollama.Client(host=OLLAMA_HOST)
    new_relations = []
    start_time = time.time()
    deadline = start_time + max_minutes * 60 if max_minutes > 0 else float("inf")

    # Live progress log — single file handle for the batch
    progress_log = EXTRACT_PROGRESS_PATH.parent / "phase-b-progress.log"
    _log_fh = open(progress_log, "a", encoding="utf-8")
    def _log(msg):
        _log_fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        _log_fh.flush()

    _log(f"Phase B started: {len(all_candidates)} candidates ({len(vault_candidates)} vault)")

    for i, (note_name, file_path, note_mtime, source) in enumerate(all_candidates):
        if time.time() > deadline:
            print(f"  Time limit reached ({max_minutes} min), stopping after {i} notes.")
            break

        # Resolve file path
        if source == "memory":
            full_path = Path(file_path)
        else:
            full_path = VAULT_PATH / file_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            processed.add(note_name)
            continue

        # Truncate for LLM
        content = content[:6000]

        # Use appropriate prompt
        if source == "memory":
            prompt = EXTRACT_PROMPT_INFRA.format(note_name=note_name, content=content)
        else:
            prompt = EXTRACT_PROMPT.format(note_name=note_name, content=content)

        try:
            response = client.chat(
                model=OLLAMA_VAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 4000},
            )
            reply = response["message"]["content"]

            # Strip <think>...</think> blocks (Gemma4 thinking output)
            reply = re.sub(r'<think>[\s\S]*?</think>', '', reply).strip()

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', reply)
            if json_match:
                raw_json = json_match.group()
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    # Fix common LLM JSON issues: trailing commas, unescaped quotes
                    fixed = re.sub(r',\s*([}\]])', r'\1', raw_json)  # trailing commas
                    fixed = fixed.replace('\n', ' ')  # newlines in strings
                    try:
                        data = json.loads(fixed)
                    except json.JSONDecodeError as e2:
                        print(f"  [{i+1}/{len(all_candidates)}] {note_name}: JSON parse failed after fix - {e2}")
                        print(f"    Raw (first 200 chars): {raw_json[:200]!r}")
                        # Don't mark as processed — will retry next run
                        continue

                entities = data.get("entities", [])
                relations = data.get("relationships", [])

                for rel in relations:
                    rel["source_note"] = note_name
                    new_relations.append(rel)

                print(f"  [{i+1}/{len(all_candidates)}] {note_name}: {len(entities)} entities, {len(relations)} relations")
                _log(f"[{i+1}/{len(all_candidates)}] {note_name}: {len(entities)}E {len(relations)}R")
            else:
                print(f"  [{i+1}/{len(all_candidates)}] {note_name}: no JSON in response")
                _log(f"[{i+1}/{len(all_candidates)}] {note_name}: no JSON")

        except Exception as e:
            print(f"  [{i+1}/{len(all_candidates)}] {note_name}: error - {e}")
            _log(f"[{i+1}/{len(all_candidates)}] {note_name}: ERROR {e}")

        processed.add(note_name)
        # Track mtime for all notes (enables re-extraction on update)
        progress.setdefault("processed_mtimes", {})[note_name] = note_mtime

        # Flush progress every 20 notes for live tracking
        if (i + 1) % 20 == 0:
            progress["processed"] = list(processed)
            save_extract_progress(progress)
            elapsed = (time.time() - start_time) / 60
            _log(f"--- Checkpoint: {len(processed)} total processed, {elapsed:.1f} min elapsed ---")

    # Merge new relations into graph (remove old relations from re-processed notes)
    existing = graph.get("extracted_relations", [])
    reprocessed_notes = {c[0] for c in reprocess}
    if reprocessed_notes:
        existing = [r for r in existing if r.get("source_note") not in reprocessed_notes]
    existing.extend(new_relations)
    graph["extracted_relations"] = existing
    save_graph(graph)

    progress["processed"] = sorted(processed)
    save_extract_progress(progress)

    elapsed_min = (time.time() - start_time) / 60
    notes_done_this_run = i + 1 if all_candidates else 0
    remaining = len(all_candidates) - notes_done_this_run
    print(f"\nDone! Added {len(new_relations)} relations (total: {len(existing)})")
    print(f"  Processed this run: {notes_done_this_run}")
    print(f"  Time: {elapsed_min:.1f} min")
    print(f"  Remaining unprocessed: {remaining}")
    _log(f"DONE: +{len(new_relations)} relations (total {len(existing)}), {notes_done_this_run} notes in {elapsed_min:.1f} min, {remaining} remaining")
    _log_fh.close()


# --- Relation Type Normalization ---
# Map LLM free-form types to canonical types
RELATION_TYPE_MAP = {
    # canonical → canonical
    "causes": "causes", "treats": "treats", "diagnoses": "diagnoses",
    "complication_of": "complication_of", "part_of": "part_of",
    "indicated_for": "indicated_for", "contraindicated_for": "contraindicated_for",
    # common LLM drift → canonical
    "treatment": "treats", "treated_by": "treats",
    "symptom": "symptom_of", "symptom_of": "symptom_of",
    "associated with": "associated_with", "associated_with": "associated_with",
    "related_to": "associated_with", "more strongly associated with": "associated_with",
    "type_of": "part_of", "subtype_of": "part_of",
    "recommended_for": "indicated_for", "useful_in": "indicated_for",
    "used_in": "indicated_for", "supports": "indicated_for",
    "assesses": "diagnoses", "finding_causing": "causes",
    "is harmful": "contraindicated_for", "procedure": "treats",
    "stimulator": "treats", "cut_at": "diagnoses",
}
CANONICAL_TYPES = {
    "causes", "treats", "diagnoses", "complication_of", "part_of",
    "indicated_for", "contraindicated_for", "symptom_of", "associated_with",
}


def normalize_relation_type(rel_type: str) -> str:
    """Normalize a relation type to canonical form."""
    return RELATION_TYPE_MAP.get(rel_type.lower().strip(), "associated_with")


# --- PMR Abbreviation Dictionary ---
PMR_ABBREVIATIONS = {
    "oa": "osteoarthritis", "ra": "rheumatoid arthritis",
    "sci": "spinal cord injury", "tbi": "traumatic brain injury",
    "cva": "cerebrovascular accident", "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism", "ms": "multiple sclerosis",
    "cp": "cerebral palsy", "als": "amyotrophic lateral sclerosis",
    "gbs": "guillain-barre syndrome", "cidp": "chronic inflammatory demyelinating polyneuropathy",
    "cts": "carpal tunnel syndrome", "mps": "myofascial pain syndrome",
    "crps": "complex regional pain syndrome", "lbp": "low back pain",
    "afo": "ankle foot orthosis", "kafo": "knee ankle foot orthosis",
    "hkafo": "hip knee ankle foot orthosis", "tlso": "thoracolumbosacral orthosis",
    "rom": "range of motion", "mmt": "manual muscle testing",
    "emg": "electromyography", "ncs": "nerve conduction study",
    "mri": "magnetic resonance imaging", "ct": "computed tomography",
    "us": "ultrasound", "eswt": "extracorporeal shock wave therapy",
    "rtms": "repetitive transcranial magnetic stimulation",
    "tdcs": "transcranial direct current stimulation",
    "vns": "vagus nerve stimulation", "fes": "functional electrical stimulation",
    "nmes": "neuromuscular electrical stimulation", "tens": "transcutaneous electrical nerve stimulation",
    "abi": "ankle brachial index", "fim": "functional independence measure",
    "mmse": "mini mental state examination", "moca": "montreal cognitive assessment",
    "nihss": "national institutes of health stroke scale",
    "asia": "american spinal injury association", "isncsci": "international standards for neurological classification of sci",
    "botox": "botulinum toxin", "bont": "botulinum toxin",
    "prp": "platelet rich plasma", "ha": "hyaluronic acid",
    "nsaid": "nonsteroidal anti-inflammatory drug", "nsaids": "nonsteroidal anti-inflammatory drug",
    "ssri": "selective serotonin reuptake inhibitor", "snri": "serotonin norepinephrine reuptake inhibitor",
    "tca": "tricyclic antidepressant", "dtr": "deep tendon reflex",
    "cmap": "compound muscle action potential", "snap": "sensory nerve action potential",
    "muap": "motor unit action potential", "nrb": "nerve root block",
    "si": "sacroiliac", "acl": "anterior cruciate ligament",
    "pcl": "posterior cruciate ligament", "rct": "rotator cuff tear",
    "avn": "avascular necrosis", "thr": "total hip replacement",
    "tkr": "total knee replacement", "ddd": "degenerative disc disease",
    "hld": "herniated lumbar disc", "hcd": "herniated cervical disc",
    "cts": "carpal tunnel syndrome", "cog": "center of gravity",
    "bos": "base of support", "grf": "ground reaction force",
}


def _rule_normalize(entity: str) -> str:
    """Rule-based entity normalization: lowercase, strip articles, singularize."""
    s = entity.lower().strip()
    # Strip leading articles
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    # Strip trailing parenthetical
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # Expand known abbreviations
    if s in PMR_ABBREVIATIONS:
        s = PMR_ABBREVIATIONS[s]
    # Simple singularize (English)
    if len(s) > 3:
        if s.endswith("ies") and not s.endswith("series"):
            s = s[:-3] + "y"
        elif s.endswith("ses") and not s.endswith("ases"):
            s = s[:-2]
        elif s.endswith("s") and not s.endswith(("ss", "us", "is")):
            s = s[:-1]
    return s


# Entities too generic to be useful for bridging
NER_STOPWORDS = {
    # Markdown/frontmatter noise
    "tag", "tags", "title", "alias", "aliase", "aliases", "day", "month", "year",
    "priority", "unscheduled", "daily", "weekday", "homepage", "hub", "post",
    "pasted image", "resource", "level", "type", "note", "file", "source",
    "section", "table", "image", "link", "page", "list", "item", "content",
    "category", "group", "index", "header", "block", "label", "field",
    "vault", "prompt", "free", "calendar", "project", "model", "tool",
    "command", "script", "config", "setting", "template", "query", "search",
    "folder", "path", "key", "value", "string", "number", "code", "text",
    "user", "session", "hook", "server", "client", "api", "url", "json",
    "claude", "claude.md", "obsidian", "plugin", "extension", "notebooklm",
    "slash command", "highlight==", "wikilink",
    # Too generic medical terms
    "patient", "patients", "treatment", "management", "evaluation", "diagnosis",
    "symptom", "symptoms", "sign", "signs", "disease", "disorder", "condition",
    "therapy", "medication", "drug", "procedure", "test", "study", "result",
    "finding", "examination", "assessment", "intervention", "outcome",
    "clinical", "chronic", "acute", "severe", "mild", "moderate",
    "normal", "abnormal", "positive", "negative", "risk", "factor",
    "male", "female", "age", "adult", "child", "elderly",
    "case", "report", "review", "article", "evidence", "data",
    "method", "analysis", "measure", "score", "scale", "rate", "ratio",
    "increase", "decrease", "improvement", "reduction", "change",
    "effect", "side effect", "complication", "prognosis", "recovery",
    "function", "activity", "exercise", "muscle", "nerve", "bone", "joint",
    "week", "weeks", "medicine", "surgical", "conservative",
    "history", "physical", "lateral", "medial", "anterior", "posterior",
    "upper", "lower", "left", "right", "bilateral", "unilateral",
    "common", "rare", "typical", "high", "low", "large", "small",
    "initial", "final", "primary", "secondary", "early", "late",
    "body", "tissue", "cell", "organ", "system", "region", "area",
    "development", "guideline", "mortality", "weak", "pediatric",
    "mechanism", "process", "pattern", "phase", "stage", "form",
}


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from note text."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:]
    return text


def _is_cjk_heavy(text: str, threshold: float = 0.5) -> bool:
    """Check if text is predominantly CJK characters."""
    if not text:
        return False
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk_count / len(text) > threshold


# --- UMLS Semantic Type → Entity Type Mapping ---
UMLS_TYPE_MAP = {
    # Condition
    "T047": "condition",  # Disease or Syndrome
    "T048": "condition",  # Mental or Behavioral Dysfunction
    "T191": "condition",  # Neoplastic Process
    "T046": "condition",  # Pathologic Function
    "T019": "condition",  # Congenital Abnormality
    "T190": "condition",  # Anatomical Abnormality
    "T020": "condition",  # Acquired Abnormality
    # Symptom
    "T184": "symptom",    # Sign or Symptom
    "T033": "symptom",    # Finding
    # Treatment
    "T061": "treatment",  # Therapeutic or Preventive Procedure
    "T058": "treatment",  # Health Care Activity
    "T065": "treatment",  # Educational Activity (rehab context)
    # Medication
    "T121": "medication", # Pharmacologic Substance
    "T200": "medication", # Clinical Drug
    "T109": "medication", # Organic Chemical (pharm use)
    "T195": "medication", # Antibiotic
    "T131": "medication", # Hazardous or Poisonous Substance
    # Anatomy
    "T023": "anatomy",    # Body Part, Organ, or Organ Component
    "T024": "anatomy",    # Tissue
    "T030": "anatomy",    # Body Space or Junction
    "T029": "anatomy",    # Body Location or Region
    "T025": "anatomy",    # Cell
    # Test
    "T059": "test",       # Laboratory Procedure
    "T060": "test",       # Diagnostic Procedure
    "T034": "test",       # Laboratory or Test Result
    # Device
    "T074": "device",     # Medical Device
    "T075": "device",     # Research Device
    "T203": "device",     # Drug Delivery Device
}


def normalize_entities_batch(graph: dict) -> dict:
    """One-time normalization of legacy extracted_relations into extracted_entities.

    Collects entities from existing relations, normalizes names, and builds
    the extracted_entities dict keyed by note.
    """
    relations = graph.get("extracted_relations", [])
    if not relations:
        print("No extracted_relations to normalize.")
        return {"normalized": 0, "notes": 0}

    # Collect raw entities per note
    note_entities_raw = defaultdict(list)  # note → [(name, type)]
    for rel in relations:
        note = rel.get("source_note", "")
        if not note:
            continue
        for key in ("from", "to"):
            raw = rel.get(key, "")
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if name and len(name) >= 2:
                note_entities_raw[note].append((name, rel.get("type", "associated_with")))

    # Normalize and deduplicate per note
    extracted_entities = {}
    total_normalized = 0
    for note, raw_ents in note_entities_raw.items():
        seen_canonical = set()
        note_ents = []
        for name, rel_type in raw_ents:
            canonical = _rule_normalize(name)
            if canonical in seen_canonical or len(canonical) < 2:
                continue
            seen_canonical.add(canonical)
            # Infer entity type from relation context
            ent_type = "condition"  # default
            if rel_type in ("treats", "indicated_for"):
                ent_type = "treatment"
            elif rel_type == "symptom_of":
                ent_type = "symptom"
            elif rel_type == "diagnoses":
                ent_type = "test"
            note_ents.append({
                "name": name,
                "type": ent_type,
                "canonical": canonical,
            })
            total_normalized += 1
        if note_ents:
            extracted_entities[note] = note_ents

    # Merge with any existing NER entities (don't overwrite)
    existing = graph.get("extracted_entities", {})
    for note, ents in extracted_entities.items():
        if note not in existing:
            existing[note] = ents
    graph["extracted_entities"] = existing

    print(f"Normalized {total_normalized} entities across {len(extracted_entities)} notes")
    return {"normalized": total_normalized, "notes": len(extracted_entities)}


def ner_batch(force: bool = False):
    """Phase B replacement: scispaCy NER on all notes.

    Args:
        force: If True, re-process all notes. Otherwise only new/modified.
    """
    import spacy

    print("Loading scispaCy model (en_core_sci_lg)...")
    nlp = spacy.load("en_core_sci_lg")
    # Disable unused pipes for speed
    nlp.disable_pipes("tagger", "attribute_ruler", "lemmatizer", "parser")
    print(f"Model loaded. Active pipes: {nlp.pipe_names}")

    graph = load_graph()
    progress = load_extract_progress()

    ner_processed = set(progress.get("ner_processed", []))
    ner_mtimes = progress.get("ner_processed_mtimes", {})
    extracted_entities = graph.get("extracted_entities", {})

    # Collect candidates
    candidates = []
    for note, meta in graph.get("metadata", {}).items():
        full_path = VAULT_PATH / meta["file"]
        if not full_path.exists():
            continue
        mtime = full_path.stat().st_mtime

        if not force and note in ner_processed:
            prev_mtime = ner_mtimes.get(note, 0)
            if prev_mtime and mtime <= prev_mtime:
                continue  # unchanged

        candidates.append((note, meta["file"], mtime))

    if not candidates:
        print("All notes already NER-processed!")
        return

    print(f"Processing {len(candidates)} notes with scispaCy NER...")
    start_time = time.time()

    for i, (note_name, file_path, mtime) in enumerate(candidates):
        full_path = VAULT_PATH / file_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            ner_processed.add(note_name)
            continue

        # Strip frontmatter, filter CJK-heavy lines
        content = _strip_frontmatter(content)
        lines = content.split('\n')
        english_lines = [ln for ln in lines if not _is_cjk_heavy(ln)]
        text = '\n'.join(english_lines)[:10000]  # cap at 10K chars

        if len(text.strip()) < 20:
            ner_processed.add(note_name)
            ner_mtimes[note_name] = mtime
            continue

        doc = nlp(text)
        note_ents = []
        seen = set()

        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 3 or name.startswith('#') or name.startswith('!'):
                continue
            # Skip pure numbers or single-char
            if re.match(r'^[\d\s.,%]+$', name):
                continue

            canonical = _rule_normalize(name)
            if canonical in seen or len(canonical) < 2:
                continue
            if canonical in NER_STOPWORDS:
                continue
            seen.add(canonical)

            note_ents.append({
                "name": name,
                "type": "condition",  # default; will be refined by UMLS if available
                "canonical": canonical,
            })

        extracted_entities[note_name] = note_ents
        ner_processed.add(note_name)
        ner_mtimes[note_name] = mtime

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(candidates)}] {elapsed:.1f}s elapsed...")

    graph["extracted_entities"] = extracted_entities
    save_graph(graph)

    progress["ner_processed"] = sorted(ner_processed)
    progress["ner_processed_mtimes"] = ner_mtimes
    save_extract_progress(progress)

    elapsed = time.time() - start_time
    total_ents = sum(len(v) for v in extracted_entities.values())
    print(f"\nNER done! {len(candidates)} notes in {elapsed:.1f}s")
    print(f"Total extracted entities: {total_ents} across {len(extracted_entities)} notes")


def get_entity_bridges_for_note(note_name: str) -> list[dict]:
    """Get notes sharing canonical entities with the given note, ranked by count."""
    graph = load_graph()
    extracted_entities = graph.get("extracted_entities", {})

    target_ents = extracted_entities.get(note_name, [])
    if not target_ents:
        # Try fuzzy match on note name
        for k in extracted_entities:
            if k.lower() == note_name.lower():
                target_ents = extracted_entities[k]
                note_name = k
                break
    if not target_ents:
        return []

    target_canonicals = {e["canonical"] for e in target_ents}

    note_scores = defaultdict(list)
    for other_note, entities in extracted_entities.items():
        if other_note == note_name:
            continue
        for ent in entities:
            if ent["canonical"] in target_canonicals:
                note_scores[other_note].append(ent["canonical"])

    results = [
        {"note": note, "shared_entities": sorted(set(ents)), "count": len(set(ents))}
        for note, ents in note_scores.items()
        if len(set(ents)) >= 2  # require at least 2 shared entities
    ]
    results.sort(key=lambda x: -x["count"])
    return results[:20]


def resolve_entities(graph: dict) -> dict:
    """Cross-note entity resolution: find shared entities across notes → add adjacency edges.

    Uses canonical names from extracted_entities (NER) + legacy extracted_relations.
    Returns dict with stats about new edges added.
    """
    adjacency = graph.get("adjacency", {})

    # Collect entities per note using canonical names
    entity_to_notes = defaultdict(set)  # canonical → {source_notes}

    # Primary: extracted_entities (NER or normalized legacy)
    for note, entities in graph.get("extracted_entities", {}).items():
        for ent in entities:
            canonical = ent.get("canonical", ent["name"]).lower().strip()
            if len(canonical) >= 3:  # skip noise
                entity_to_notes[canonical].add(note)

    # Fallback: legacy extracted_relations (for notes not yet NER-processed)
    ner_notes = set(graph.get("extracted_entities", {}).keys())
    for rel in graph.get("extracted_relations", []):
        note = rel.get("source_note", "")
        if not note or note in ner_notes:
            continue  # skip if already covered by NER
        for entity in [rel.get("from", ""), rel.get("to", "")]:
            if isinstance(entity, list):
                entity = entity[0] if entity else ""
            if not isinstance(entity, str) or not entity:
                continue
            canonical = _rule_normalize(entity)
            if len(canonical) >= 3:
                entity_to_notes[canonical].add(note)

    # Find entities appearing in 2+ notes (but not too many — noise filter)
    MAX_BRIDGE_NOTES = 50  # entities in 50+ notes are too generic to be useful
    entity_bridges = []
    for entity, notes in entity_to_notes.items():
        if len(notes) < 2 or len(notes) > MAX_BRIDGE_NOTES:
            continue
        if entity in NER_STOPWORDS:
            continue
        note_list = sorted(notes)
        entity_bridges.append({"entity": entity, "notes": note_list})

    # Entity bridges are stored separately — NOT added to adjacency
    # (adjacency is for wiki-links only; bridges are queried via --bridges)
    graph["entity_bridges"] = entity_bridges

    # Also normalize legacy relation types
    for rel in graph.get("extracted_relations", []):
        rel["type"] = normalize_relation_type(rel.get("type", ""))

    return {"bridges": len(entity_bridges)}


def show_stats():
    """Show graph statistics."""
    graph = load_graph()
    stats = graph.get("stats", {})
    adj = graph.get("adjacency", {})
    relations = graph.get("extracted_relations", [])
    progress = load_extract_progress()

    print("=== Vault Knowledge Graph ===")
    print(f"Built: {stats.get('built_at', 'never')}")
    print(f"Notes: {stats.get('total_notes', 0)}")
    print(f"Notes with links: {stats.get('notes_with_links', 0)}")
    print(f"Wiki-link edges: {stats.get('total_edges', 0)}")
    print(f"Avg connections: {sum(len(v) for v in adj.values()) / max(len(adj), 1):.1f}")
    # NER stats
    extracted_entities = graph.get("extracted_entities", {})
    if extracted_entities:
        total_ents = sum(len(v) for v in extracted_entities.values())
        print(f"\n--- NER (scispaCy) ---")
        print(f"Notes with entities: {len(extracted_entities)}")
        print(f"Total entities: {total_ents}")
        print(f"NER processed: {len(progress.get('ner_processed', []))}")

    print(f"\n--- Legacy Phase B (LLM extraction, frozen) ---")
    print(f"Notes processed: {len(progress.get('processed', []))}")
    print(f"Extracted relations: {len(relations)}")
    print(f"Last run: {progress.get('last_run', 'never')}")

    # Relation type distribution
    type_counts = defaultdict(int)
    for rel in relations:
        type_counts[normalize_relation_type(rel.get("type", ""))] += 1
    if type_counts:
        print(f"Relation types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

    # Entity bridges
    bridges = graph.get("entity_bridges", [])
    if bridges:
        print(f"\n--- Entity Bridges (cross-note) ---")
        print(f"Shared entities: {len(bridges)}")
        for b in bridges[:10]:
            print(f"  '{b['entity']}' → {b['notes']}")

    # Top connected notes
    if adj:
        top = sorted(adj.items(), key=lambda x: -len(x[1]))[:10]
        print(f"\nTop 10 most connected notes:")
        for note, neighbors in top:
            print(f"  {len(neighbors):3d} connections: {note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vault Knowledge Graph Builder")
    parser.add_argument("--extract", type=int, metavar="N", nargs="?", const=0,
                        help="(Legacy) Run LLM extraction (N notes, or use --max-minutes)")
    parser.add_argument("--max-minutes", type=int, default=0, metavar="M",
                        help="Time limit for --extract in minutes (0=use N)")
    parser.add_argument("--ner", action="store_true",
                        help="Run scispaCy NER on new/modified notes")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing (use with --ner)")
    parser.add_argument("--normalize", action="store_true",
                        help="One-time: normalize legacy extracted_relations into extracted_entities")
    parser.add_argument("--bridges", type=str, metavar="NOTE",
                        help="Query entity bridges for a note (output JSON)")
    parser.add_argument("--resolve", action="store_true",
                        help="Run entity resolution across notes")
    parser.add_argument("--count-pending", action="store_true",
                        help="Print number of notes needing NER (new + modified)")
    parser.add_argument("--stats", action="store_true",
                        help="Show graph statistics")
    args = parser.parse_args()

    if args.bridges is not None:
        results = get_entity_bridges_for_note(args.bridges)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.normalize:
        graph = load_graph()
        result = normalize_entities_batch(graph)
        save_graph(graph)
        print(f"Normalization complete: {result['normalized']} entities, {result['notes']} notes")
    elif args.ner:
        ner_batch(force=args.force)
    elif args.count_pending:
        graph = load_graph()
        progress = load_extract_progress()
        ner_processed = set(progress.get("ner_processed", []))
        ner_mtimes = progress.get("ner_processed_mtimes", {})
        pending = 0
        for note, meta in graph.get("metadata", {}).items():
            if note not in ner_processed:
                pending += 1
            else:
                prev_mtime = ner_mtimes.get(note, 0)
                if not prev_mtime:
                    continue
                try:
                    mtime = (VAULT_PATH / meta["file"]).stat().st_mtime
                except (FileNotFoundError, OSError):
                    continue
                if mtime > prev_mtime:
                    pending += 1
        print(pending)
    elif args.stats:
        show_stats()
    elif args.resolve:
        graph = load_graph()
        result = resolve_entities(graph)
        save_graph(graph)
        print(f"Entity resolution: {result['bridges']} shared entities")
    elif args.extract is not None:
        print("WARNING: --extract uses legacy LLM extraction. Consider --ner instead.")
        extract_batch(n=args.extract or 10, max_minutes=args.max_minutes)
    else:
        graph = build_link_graph()
        save_graph(graph)
        print(f"\nStats:")
        print(f"  Notes: {graph['stats']['total_notes']}")
        print(f"  Notes with links: {graph['stats']['notes_with_links']}")
        print(f"  Edges: {graph['stats']['total_edges']}")
