from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(path: Path) -> List[Document]:
    # One loader instance per file keeps page metadata intact
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["type"] = "pdf"
        d.metadata["source"] = str(path)
        d.metadata["doc_name"] = path.name
    return docs
