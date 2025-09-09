from pathlib import Path
from typing import List, Optional
import subprocess
import tempfile
import shutil
import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

def _get_soffice_cmd() -> str:
    """
    Resolve soffice executable. You can set LIBREOFFICE_PATH env var to the full exe path.
    Falls back to 'soffice' on PATH.
    """
    env_path = os.environ.get("LIBREOFFICE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    return "soffice"

def _convert_doc_to_docx(src: Path, out_dir: Path) -> Path:
    """
    Convert legacy .doc to .docx using LibreOffice (soffice) in headless mode.
    Returns the path to the converted .docx.
    Raises RuntimeError if conversion fails or output file is missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    soffice = _get_soffice_cmd()
    cmd = [
        soffice,
        "--headless",
        "--convert-to", "docx",   # don't specify a filter; LO picks correct one
        str(src),
        "--outdir", str(out_dir),
    ]
    # Capture output to prevent noisy stdout/stderr in logs
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # LibreOffice returns 0 on success; but we also verify the file physically exists
    out_docx = out_dir / (src.stem + ".docx")
    if proc.returncode != 0 or not out_docx.exists():
        raise RuntimeError(
            "LibreOffice conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return out_docx

def load_doc(path: Path) -> List[Document]:
    """
    Convert .doc -> .docx, then load via Docx2txtLoader.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="doc2docx_"))
    try:
        out_docx = _convert_doc_to_docx(path, tmpdir)
        docs = Docx2txtLoader(str(out_docx)).load()
        for d in docs:
            d.metadata.update({
                "type": "doc",
                "source": str(path),     # keep original .doc as source
                "doc_name": path.name,
                "converted_from": str(out_docx),
            })
        return docs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
