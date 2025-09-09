# api/main.py  — replace entire file with this

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, UploadFile, File, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import subprocess, os, json, time, shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

import yaml
from vectordb.chroma_client import get_chroma

# -------- Settings --------
API_KEY = os.getenv("OPAL_API_KEY", "")  # optional API key
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (PROJECT_ROOT / "data").resolve()
MANIFEST_PATH = (PROJECT_ROOT / "chroma/.ingest_manifest.json").resolve()

# For uploads, save into the same folder your ingest uses:
DATA_PATH = DATA_DIR

# CORS origins (dev Vite/Svelte)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
# Optionally extend via env var FRONTEND_ORIGINS="http://myapp.com,https://staging.myapp.com"
extra = os.getenv("FRONTEND_ORIGINS")
if extra:
    ALLOWED_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

# -------- App --------
app = FastAPI(title="OPAL RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helpers --------
def check_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _safe_in_data(p: Path) -> Path:
    """Ensure path stays inside data/."""
    rp = (DATA_DIR / p).resolve()
    if not str(rp).startswith(str(DATA_DIR)):
        raise HTTPException(400, "Path escapes data directory")
    return rp

def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

# light ingest "job" status
INGEST_STATE = {"last_started": None, "last_finished": None, "args": None, "running": False}
def _bg_ingest(args: List[str]):
    def run():
        try:
            INGEST_STATE["running"] = True
            INGEST_STATE["last_started"] = time.time()
            INGEST_STATE["args"] = args[:]
            subprocess.run(args, cwd=str(PROJECT_ROOT))
        finally:
            INGEST_STATE["running"] = False
            INGEST_STATE["last_finished"] = time.time()
    return run

# -------- Models --------
class IngestRequest(BaseModel):
    reset: bool = False
    rescan: bool = False

class QueryRequest(BaseModel):
    query: str
    k: int = 12
    fetch_k: int = 48
    per_source_limit: int = 2
    max_context_chars: int = 12000
    type: Optional[str] = None
    file: Optional[str] = None
    model: str = "mistral"
    show_snippets: bool = True

# -------- Routes --------
@app.get("/health")
def health(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    return {"ok": True}

@app.get("/files")
def files(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    if not MANIFEST_PATH.exists():
        return {"files": []}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    items = [{"path": k, "sig": v.get("sig")} for k, v in data.items()]
    return {"files": items}

@app.post("/ingest")
def ingest(req: IngestRequest, x_api_key: Optional[str] = Header(None), background_tasks: BackgroundTasks = None):
    check_key(x_api_key)
    args = ["py", "ingest.py"]
    if req.reset: args.append("--reset")
    if req.rescan: args.append("--rescan")
    # Run in background so the HTTP call returns immediately
    if background_tasks:
        background_tasks.add_task(_bg_ingest(args))
        return {"started": True, "args": args}
    # fallback sync
    subprocess.run(args, cwd=str(PROJECT_ROOT))
    return {"started": True, "args": args}

@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    args = ["py", "query_data.py", req.query,
            "--k", str(req.k),
            "--fetch-k", str(req.fetch_k),
            "--per-source-limit", str(req.per_source_limit),
            "--max-context-chars", str(req.max_context_chars),
            "--model", req.model]
    if req.type: args += ["--type", req.type]
    if req.file: args += ["--file", req.file]
    proc = subprocess.run(args, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"Query failed: {proc.stderr}")
    return {"stdout": proc.stdout}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    dest = DATA_PATH / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    # optional: trigger an incremental rescan in the background
    # background trigger intentionally omitted here; call /ingest from FE after upload.
    return {"filename": file.filename, "saved_to": str(dest)}

@app.delete("/files")
def delete_file(
    path: str = Query(..., description="Path relative to data/, e.g. 'folder/doc.pdf'"),
    reingest: bool = Query(True, description="Kick ingest after deletion"),
    x_api_key: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = None,
):
    check_key(x_api_key)
    abs_path = _safe_in_data(Path(path))
    if not abs_path.exists():
        raise HTTPException(404, f"Not found: {abs_path.relative_to(DATA_DIR)}")

    # 1) delete on disk
    try:
        abs_path.unlink()
    except IsADirectoryError:
        shutil.rmtree(abs_path)

    # 2) delete vectors for this source
    try:
        db = get_chroma()  # reads config.yaml internally
        db._collection.delete(where={"source": {"$eq": str(abs_path.resolve())}})
    except Exception as e:
        print("Vector delete error:", e)  # non-fatal

    # 3) drop manifest entry
    m = _load_manifest()
    m.pop(str(abs_path.resolve()), None)
    _save_manifest(m)

    # 4) optional background ingest
    started = False
    if reingest and background_tasks:
        args = ["py", "ingest.py", "--rescan"]
        background_tasks.add_task(_bg_ingest(args))
        started = True

    return {"deleted": str(abs_path.relative_to(DATA_DIR)), "reingest_started": started}

@app.get("/files/index-status")
def files_index_status(x_api_key: Optional[str] = Header(None)):
    """
    Return a mapping of doc_name -> count of chunks in Chroma.
    Also include counts by absolute source path.
    """
    check_key(x_api_key)
    try:
        db = get_chroma()
        # grab ids+metas; limit to metadata only for speed
        res = db._collection.get(include=["metadatas", "ids"])
        counts_by_doc = {}
        counts_by_source = {}
        for md in res.get("metadatas", []):
            if not md: 
                continue
            doc_name = md.get("doc_name")
            source = md.get("source")
            if doc_name:
                counts_by_doc[doc_name] = counts_by_doc.get(doc_name, 0) + 1
            if source:
                counts_by_source[source] = counts_by_source.get(source, 0) + 1
        return {"by_doc_name": counts_by_doc, "by_source": counts_by_source}
    except Exception as e:
        raise HTTPException(500, f"Index status failed: {e}")