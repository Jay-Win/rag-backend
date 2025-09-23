# api/main.py
import os
import sys
import subprocess
import json
import time
import shutil
import re
import html
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.init_db import init_db
from db.session import SessionLocal
from db.models import Chat, Message
from api.chats import router as chats_router
from api.security import check_key
from vectordb.chroma_client import get_chroma

# --- Python executable to use for subprocesses (works in Docker, Linux, Mac, Windows)
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable or "python")

# -------- Settings --------
API_KEY = os.getenv("OPAL_API_KEY", "my-secret-key")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (PROJECT_ROOT / "data").resolve()
MANIFEST_PATH = (PROJECT_ROOT / "chroma/.ingest_manifest.json").resolve()
DATA_PATH = DATA_DIR

# CORS origins (dev Vite/Svelte)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
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

@app.on_event("startup")
def _startup():
    init_db()

# mount chats routes
app.include_router(chats_router)

# -------- Helpers --------
def _safe_in_data(p: Path) -> Path:
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
INGEST_STATE: Dict[str, Any] = {"last_started": None, "last_finished": None, "args": None, "running": False}

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
    chat_id: Optional[str] = None

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
def ingest(req: IngestRequest, background_tasks: BackgroundTasks, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    args = [PYTHON_BIN, "ingest.py"]
    if req.reset:  args.append("--reset")
    if req.rescan: args.append("--rescan")
    background_tasks.add_task(_bg_ingest(args))
    return {"started": True, "args": args}


from fastapi import Depends
QUERY_SCRIPT = os.getenv("QUERY_SCRIPT", "query_data2.py")

@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)

    args = [PYTHON_BIN, QUERY_SCRIPT, req.query, "--k", str(req.k), "--model", req.model]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    chat_id = req.chat_id

    # record USER message first
    if chat_id:
        db = SessionLocal()
        try:
            chat = db.get(Chat, chat_id)
            if not chat:
                chat = Chat(id=chat_id, title="New chat")
                db.add(chat)
                db.commit()
                db.refresh(chat)

            try:
                payload_dict = req.model_dump()
            except AttributeError:
                payload_dict = req.dict()

            db.add(Message(
                chat_id=chat.id,
                role="user",
                content=req.query,
                payload=payload_dict
            ))
            chat.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    # run model subprocess
    try:
        proc = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60,
        )
    except subprocess.TimeoutExpired as te:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Query timed out after 60s.\nArgs: {args}\n"
                f"Partial stdout:\n{te.stdout or ''}\nPartial stderr:\n{te.stderr or ''}\n"
                f"Tip: ensure Ollama is reachable and the model '{req.model}' is available."
            ),
        )
    except Exception as e:
        raise HTTPException(500, f"Query subprocess failed to start: {e}")

    if proc.returncode != 0:
        raise HTTPException(
            500,
            f"Query failed.\nArgs: {args}\nStderr:\n{proc.stderr}\n"
            f"Tip: ensure Ollama is reachable and 'ollama pull {req.model}'.",
        )

    # persist ASSISTANT message and auto-name on first turn
    chat_snapshot = None
    if chat_id:
        db = SessionLocal()
        try:
            chat = db.get(Chat, chat_id)
            if chat:
                db.add(Message(
                    chat_id=chat.id,
                    role="assistant",
                    content=proc.stdout,
                    raw=proc.stdout,
                    payload={"args": args, "code": proc.returncode},
                ))
                chat.updated_at = datetime.utcnow()
                db.commit()

                try:
                    default_names = {"new chat", "imported chat"}
                    is_default = (chat.title or "").strip().lower() in default_names
                    msg_count = db.query(Message).filter(Message.chat_id == chat.id).count()
                    if is_default and msg_count <= 2:
                        final_answer = _extract_final_answer(proc.stdout)
                        new_title = _derive_title(req.query, final_answer)
                        if new_title and new_title.strip().lower() not in default_names:
                            chat.title = new_title
                            chat.updated_at = datetime.utcnow()
                            db.commit()
                except Exception:
                    pass

                chat_snapshot = {
                    "id": chat.id,
                    "title": chat.title,
                    "archived": chat.archived,
                    "created_at": chat.created_at.isoformat(),
                    "updated_at": chat.updated_at.isoformat(),
                }
        finally:
            db.close()

    return {
        "args": args,
        "code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "chat": chat_snapshot,
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    check_key(x_api_key)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    dest = DATA_PATH / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "saved_to": str(dest)}

@app.get("/ingest/status")
def ingest_status(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    return {
        "running": INGEST_STATE.get("running", False),
        "last_started": INGEST_STATE.get("last_started"),
        "last_finished": INGEST_STATE.get("last_finished"),
        "args": INGEST_STATE.get("args"),
    }

@app.delete("/files")
def delete_file(
    path: str = Query(..., description="Path relative to data/, e.g. 'folder/doc.pdf'"),
    reingest: bool = Query(True, description="Kick ingest after deletion"),
    background_tasks: BackgroundTasks = None,
    x_api_key: Optional[str] = Header(None),
):
    check_key(x_api_key)

    bg = background_tasks if isinstance(background_tasks, BackgroundTasks) else None

    abs_path = _safe_in_data(Path(path))
    if not abs_path.exists():
        raise HTTPException(404, f"Not found: {abs_path.relative_to(DATA_DIR)}")

    try:
        abs_path.unlink()
    except IsADirectoryError:
        shutil.rmtree(abs_path)

    try:
        db = get_chroma()
        db._collection.delete(where={"source": {"$eq": str(abs_path.resolve())}})
    except Exception as e:
        print("Vector delete error:", e)

    m = _load_manifest()
    m.pop(str(abs_path.resolve()), None)
    _save_manifest(m)

    started = False
    if reingest and bg is not None:
        args = [PYTHON_BIN, "ingest.py", "--rescan"]
        bg.add_task(_bg_ingest(args))
        started = True

    return {"deleted": str(abs_path.relative_to(DATA_DIR)), "reingest_started": started}

@app.get("/files/index-status")
def files_index_status(x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    try:
        db = get_chroma()
        res = db._collection.get(include=["metadatas", "ids"])
        counts_by_doc: Dict[str, int] = {}
        counts_by_source: Dict[str, int] = {}
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
    except Exception:
        return {"by_doc_name": {}, "by_source": {}}

# --- Helpers for chat auto-naming -------------------------------------------
RESP_RE = re.compile(r"Response:\s*([\s\S]*?)(?:\n+Sources:|$)", re.IGNORECASE)

def _extract_final_answer(stdout: str) -> str:
    if not stdout:
        return ""
    m = RESP_RE.search(stdout)
    if m:
        return m.group(1).strip()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""

def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()

def _derive_title(user_text: str, answer_text: str, max_len: int = 60) -> str:
    base = _strip_html(answer_text) or (user_text or "").strip()
    if not base:
        return "New chat"
    first = re.split(r"(?<=[.!?])\s+", base, maxsplit=1)[0].strip()
    cand = (first or base).lstrip("-• ").rstrip(" .!?")
    if len(cand) > max_len:
        cand = cand[:max_len].rstrip() + "…"
    return cand[:1].upper() + cand[1:]
