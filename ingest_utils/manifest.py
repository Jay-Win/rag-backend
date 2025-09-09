from pathlib import Path
import json
import hashlib

MANIFEST_PATH = Path("chroma/.ingest_manifest.json")

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.load(open(MANIFEST_PATH, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(MANIFEST_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def file_signature(p: Path) -> str:
    """Stable signature that changes on any content change.
    - For small/medium files: full SHA1
    - For large files: SHA1(head+tail) to stay fast
    """
    stat = p.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    try:
        if size <= 16 * 1024 * 1024:
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
