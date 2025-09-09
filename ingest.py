import os
import shutil
import yaml
import argparse
from pathlib import Path
from collections import Counter
from typing import List
from langchain_core.documents import Document

from chunking.text_chunker import chunk_text
from vectordb.chroma_client import get_chroma

# our new utils
from ingest_utils.manifest import load_manifest, save_manifest, file_signature
from ingest_utils.ids import assign_ids
from ingest_utils.meta import sanitize_metadata, delete_docs_for_source
from ingest_utils.loaders_map import build_loaders_map


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
    if args.reset and os.path.exists(chroma_path):
        print("✨ Clearing Database")
        shutil.rmtree(chroma_path)
        try:
            Path("chroma/.ingest_manifest.json").unlink()
        except FileNotFoundError:
            pass

    # --- Build extension → loader map ---
    loaders_map = build_loaders_map(loaders_cfg)

    # --- Manifest ---
    manifest = load_manifest()
    current_seen = set()

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
            docs = [d for d in docs if (d.page_content or "").strip()]
            manifest[key] = {"sig": sig}
            current_seen.add(key)
            all_docs.extend(docs)
            print(f"Loaded {len(docs):3d} docs from {path.name}")
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
