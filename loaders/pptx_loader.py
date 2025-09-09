from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document

def load_pptx(path: Path) -> List[Document]:
    loader = UnstructuredPowerPointLoader(str(path))
    docs = loader.load()
    # Try to keep slide-level provenance if present
    for i, d in enumerate(docs):
        d.metadata["type"] = "pptx"
        d.metadata["source"] = str(path)
        d.metadata.setdefault("slide", d.metadata.get("section", i))
        d.metadata["doc_name"] = path.name
    return docs
