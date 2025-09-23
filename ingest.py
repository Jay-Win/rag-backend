# ingest.py — minimal & robust for PDF/DOCX/TXT only

import os
import re
import json
import shutil
import yaml
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

from langchain_core.documents import Document

# Loaders (only the three we support in this minimal build)
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.txt_loader import load_txt

from chunking.text_chunker import chunk_text
from vectordb.chroma_client import get_chroma


# -------------------------
# Constants / Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = (PROJECT_ROOT / "chroma/.ingest_manifest.json").resolve()

# -------------------------
# Manifest helpers
# -------------------------

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(m: dict) -> None:
    """Atomic write to avoid corruption on crash."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)

# -------------------------
# Signature / IDs
# -------------------------

def file_signature(p: Path) -> str:
    """Stable signature based on size + mtime_ns + content hash (head/tail for large)."""
    stat = p.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    try:
        if size <= 16 * 1024 * 1024:  # <= 16MB full hash
            h = hashlib.sha1()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            fp = h.hexdigest()
        else:
            with open(p, "rb") as f:
                head = f.read(1024 * 1024)
                f.seek(max(0, size - 1024 * 1024))
                tail = f.read(1024 * 1024)
            fp = hashlib.sha1(head + tail).hexdigest()
    except Exception:
        fp = "nofp"
    return f"{size}:{mtime_ns}:{fp}"

def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]

def build_locator(md: dict) -> str:
    parts = []
    for key in ("page", "slide", "sheet", "row", "section", "element_id"):
        v = md.get(key)
        if v not in (None, "", "none"):
            parts.append(f"{key}={v}")
    return ";".join(parts)

def assign_ids(chunks: List[Document]) -> List[Document]:
    """Stable, readable IDs per (source, locator, index) + short content hash."""
    counters: Dict[str, int] = {}
    for d in chunks:
        md = d.metadata or {}
        source = md.get("source", "unknown")
        locator = build_locator(md)
        key = f"{source}|{locator}"
        idx = counters.get(key, 0)
        h = short_hash((d.page_content or "") + locator)
        d.metadata["id"] = f"{source}:{locator}:{idx}:{h}"
        counters[key] = idx + 1
    return chunks

# -------------------------
# Metadata utils
# -------------------------

def sanitize_metadata(md: dict) -> dict:
    """Keep simple, JSON-serializable metadata."""
    clean = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = v[0] if len(v) == 1 and isinstance(v[0], (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            clean[k] = str(v)
    return clean

def normalize_basic_metadata(doc: Document, abs_path: Path, typ: str) -> None:
    """Ensure required fields used by filters are present."""
    if doc.metadata is None:
        doc.metadata = {}
    doc.metadata.setdefault("source", str(abs_path))
    doc.metadata.setdefault("doc_name", abs_path.name)
    doc.metadata.setdefault("type", typ)  # 'pdf'|'docx'|'txt'

# -------------------------
# Chroma helpers
# -------------------------

def delete_docs_for_source(db, source_path_str: str):
    try:
        db._collection.delete(where={"source": {"$eq": source_path_str}})
        print(f"🧹 Removed old chunks for {Path(source_path_str).name}")
    except Exception as e:
        print(f"⚠️ Could not delete old chunks for {source_path_str}: {e}")

# -------------------------
# Loader map (PDF/DOCX/TXT only)
# -------------------------

def loader_for_ext(ext: str):
    if ext == ".pdf":
        return load_pdf, "pdf"
    if ext == ".docx":
        return load_docx, "docx"
    if ext == ".txt":
        return load_txt, "txt"
    return None, None



def build_loaders_map(loaders_cfg):
    """Return only the enabled, minimal loaders (pdf/docx/txt) without importing others."""
    m = {}
    if loaders_cfg.get("pdf", False):
        m[".pdf"] = load_pdf
    if loaders_cfg.get("docx", False):
        m[".docx"] = load_docx
    if loaders_cfg.get("txt", False):
        m[".txt"] = load_txt
    return m


from pathlib import Path
import shutil
import os

def _clear_dir(path: str | Path):
    p = Path(path)
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            try: child.unlink()
            except Exception: pass
        else:
            try: shutil.rmtree(child)
            except Exception: pass


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--rescan", action="store_true", help="Ignore manifest cache and rescan all files.")
    args = parser.parse_args()

    # --- Load config ---
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data_path"])
    chroma_path = cfg["chroma_path"]
    chunk_cfg = cfg["chunking"]["text"]
    loaders_cfg = cfg["loaders"]

    # --- Reset DB if requested ---

    if args.reset:
        print("=== 🚨 Ingest Mode: RESET (clear DB contents) ===")
        print("✨ Clearing Database contents")
        _clear_dir(chroma_path)
        # Remove manifest file if present
        try:
            Path("chroma/.ingest_manifest.json").unlink()
        except FileNotFoundError:
            pass

    # --- Build extension → loader map ---
    loaders_map = build_loaders_map(loaders_cfg)

    # --- Manifest ---
    manifest = load_manifest()
    current_seen = set()

    # --- Diagnostics collectors ---
    unsupported = []
    empty_or_whitespace = []
    ingested_files = 0

    # --- Walk data folder ---
    all_docs: List[Document] = []
    SKIP_SUFFIXES = {".docx#", ".backup"}

    for path in data_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES or path.name.endswith("~"):
            continue

        loader = loaders_map.get(path.suffix.lower())
        if not loader:
            unsupported.append(str(path))
            continue

        sig = file_signature(path)
        key = str(path.resolve())

        if not args.rescan:
            rec = manifest.get(key)
            if rec and rec.get("sig") == sig:
                current_seen.add(key)
                continue

        try:
            docs = loader(path)

            if path.suffix.lower() == ".txt" and not docs:
                try:
                    if path.stat().st_size == 0:
                        empty_or_whitespace.append(f"{path.name} (zero bytes)")
                    else:
                        empty_or_whitespace.append(f"{path.name} (empty/whitespace or decode-failed)")
                except Exception:
                    empty_or_whitespace.append(f"{path.name} (unknown size)")

            docs = [d for d in docs if (d.page_content or "").strip()]
            manifest[key] = {"sig": sig}
            current_seen.add(key)

            if docs:
                ingested_files += 1
                all_docs.extend(docs)
                print(f"Loaded {len(docs):3d} docs from {path.name}")
            else:
                print(f"ℹ️  No content extracted from {path.name}")

        except Exception as e:
            print(f"⚠️ Skipping {path.name}: {e}")

    # --- Clean up removed files ---
    removed = set(manifest.keys()) - current_seen
    if removed:
        db = get_chroma(chroma_path)
        for dead in removed:
            try:
                db._collection.delete(where={"source": {"$eq": dead}})
                print(f"🗑️  Removed all chunks for deleted file: {Path(dead).name}")
                manifest.pop(dead, None)
            except Exception as e:
                print(f"⚠️ Could not delete chunks for {dead}: {e}")

    # --- Report diagnostics ---
    print(f"Scan summary: {ingested_files} candidate files, {len(unsupported)} unsupported, {len(empty_or_whitespace)} empty/whitespace")

    if unsupported:
        print("↪ Unsupported examples:")
        for p in unsupported[:10]:
            print("  -", p)

    if empty_or_whitespace:
        print("↪ TXT with no usable content:")
        for p in empty_or_whitespace[:10]:
            print("  -", p)

    if not all_docs:
        print("No new/changed documents to (re)chunk. Saving manifest and exiting.")
        save_manifest(manifest)
        return

    # --- Chunking & assign IDs ---
    chunks = chunk_text(all_docs, chunk_cfg["chunk_size"], chunk_cfg["overlap"])
    chunks = assign_ids(chunks)

    by_source = Counter(d.metadata.get("source", "unknown") for d in chunks)
    for src, n in by_source.items():
        print(f"{n}: Chunks for {Path(src).name}")
    print(f"Total chunks: {len(chunks)}")

    # --- Upsert to Chroma ---
    db = get_chroma(chroma_path)
    existing = db.get(include=[])
    existing_ids = set(existing.get("ids", []))
    new_docs = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if not new_docs:
        print("✅ No new documents to add")
        save_manifest(manifest)
        return

    from collections import defaultdict
    by_source_new = defaultdict(list)
    for d in new_docs:
        by_source_new[d.metadata.get("source", "unknown")].append(d)

    print(f"👉 Adding new documents: {len(new_docs)} (grouped across {len(by_source_new)} sources)")

    for src, docs_for_src in by_source_new.items():
        delete_docs_for_source(db, src)
        for d in docs_for_src:
            d.metadata = sanitize_metadata(d.metadata)
        db.add_documents(docs_for_src, ids=[d.metadata["id"] for d in docs_for_src])
        print(f"✅ Upserted {len(docs_for_src)} chunks for {Path(src).name}")

    save_manifest(manifest)

if __name__ == "__main__":
    main()
