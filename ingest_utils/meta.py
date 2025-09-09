from pathlib import Path

def sanitize_metadata(md: dict) -> dict:
    import json
    clean = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], (str, int, float, bool)):
                clean[k] = v[0]
            else:
                clean[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            clean[k] = str(v)
    return clean

def delete_docs_for_source(db, source_path_str: str):
    try:
        db._collection.delete(where={"source": {"$eq": source_path_str}})
        print(f"🧹 Removed old chunks for {Path(source_path_str).name}")
    except Exception as e:
        print(f"⚠️ Could not delete old chunks for {source_path_str}: {e}")
